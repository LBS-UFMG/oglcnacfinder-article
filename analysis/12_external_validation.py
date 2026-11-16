# %%
import os
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))

import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# External validation of the final model (decision 12b; our model only)
#   Inputs: the external sites and their sequences (NB01, external section) and the final
#   model's score for every S/T of every external protein (NB05, last cell).
#   Labels, per evaluation unit:
#     positive  S/T localized to one residue by the study;
#     masked    S/T reported without single-residue localization (Lei "ambiguous" positions;
#               every candidate position of a NOTISE "(1 of ...)" region), excluded from scoring;
#     negative  every other S/T of the unit's proteins (unverified, as in the training data).
#   Units: each study (Fu 2026 = its three cell lines combined) and all studies pooled (a site
#   is positive if any study localized it; masked if reported only without localization).
#   Every protein the study reports (localized or not) enters the study's unit.
# ---------------------------------------------------------------------------
OUT_DIR = 'data_processed/external_validation'
os.makedirs(OUT_DIR, exist_ok=True)

ext_sites = pd.read_parquet('data_processed/external_sites_mapped.parquet')
ext_proteins = pd.read_parquet('data_processed/external_proteins.parquet')
scores = pd.read_parquet('data_processed/final_model/external_scores.parquet')
seq_of = dict(zip(ext_proteins['seq_id'], ext_proteins['sequence']))
assert ext_sites['seq_id'].notna().all()

# Reported positions, one row per study, protein and position, with their status
rows = []
for r in ext_sites.itertuples(index=False):
    seq = seq_of[r.seq_id]
    if r.localized:
        positions, status = [int(r.position)], 'positive'
    elif pd.notna(r.position):                                   # Lei: unlocalized, best position given
        positions, status = [int(r.position)], 'masked'
    else:                                                        # NOTISE region "(1 of a,b,c)"
        positions, status = [int(p) for p in re.findall(r'\d+', r.detail.split('of', 1)[1])], 'masked'
    for p in positions:
        if 1 <= p <= len(seq) and seq[p - 1] in 'ST':          # Tyr and other residues are not scored
            rows.append({'study': r.study, 'seq_id': r.seq_id, 'position': p, 'status': status})
reported = pd.DataFrame(rows)
# Within a study, a localized report wins over an unlocalized one at the same position
reported['rank'] = (reported['status'] == 'positive').astype(int)
reported = (reported.sort_values('rank', ascending=False)
            .drop_duplicates(['study', 'seq_id', 'position']).drop(columns='rank'))

UNITS = list(ext_sites['study'].unique()) + ['All studies pooled']
labels = []
for unit in UNITS:
    rep = reported if unit == 'All studies pooled' else reported[reported['study'] == unit]
    rep = (rep.assign(rank=(rep['status'] == 'positive').astype(int)).sort_values('rank', ascending=False)
           .drop_duplicates(['seq_id', 'position']).drop(columns=['rank', 'study']))
    proteins = rep['seq_id'].unique()
    unit_scores = scores[scores['seq_id'].isin(proteins)]
    lab = unit_scores.merge(rep, on=['seq_id', 'position'], how='left')
    lab['label'] = lab['status'].fillna('negative')
    labels.append(lab.drop(columns='status').assign(unit=unit))
labels = pd.concat(labels, ignore_index=True)
labels.to_parquet(f'{OUT_DIR}/labels_and_scores.parquet', index=False)

summary = (labels.groupby('unit', sort=False)
           .agg(proteins=('seq_id', 'nunique'), sites=('label', 'size'),
                positive=('label', lambda x: (x == 'positive').sum()),
                negative=('label', lambda x: (x == 'negative').sum()),
                masked=('label', lambda x: (x == 'masked').sum())))
summary['positive rate'] = summary['positive'] / (summary['positive'] + summary['negative'])
print(f'Reported sites: {len(ext_sites):,} rows; positions on S/T after expanding regions and removing repeats: '
      f'{len(reported):,}')
print(summary.to_string(formatters={'positive rate': '{:.4f}'.format}))

# %%
import subprocess
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

# ---------------------------------------------------------------------------
# Overlap of the external proteins with the proteins the final model has seen, and metrics
#   Seen proteins: the training and validation splits of our dataset (validation was used for
#   early stopping, model selection, Platt calibration and the cut-off).
#   Two definitions of overlap, each removing whole external proteins (as for the competitors):
#     same protein: an accession of the external protein (isoform suffix ignored) is a seen
#                   protein ID, or its sequence is identical to a seen sequence;
#     homolog:      identity >= 30% to any seen protein, with identity = identical residues /
#                   length of the shorter protein, from an exhaustive MMseqs2 alignment (fake
#                   prefilter, --alignment-mode 3, E <= 0.001; E-values relative to the seen proteins).
#   Metrics per unit and subset (masked sites excluded): AUROC, AUPRC, and MCC, precision and
#   recall at the validation cut-off (calibrated probability >= cut-off).
# ---------------------------------------------------------------------------
atlas = pd.read_parquet('data_processed/atlas5_proteins.parquet')
seen = atlas[atlas['split'].isin(['train', 'val'])][['protein_id', 'sequence']]
print(f'Seen proteins (training + validation splits): {len(seen):,}')

MMSEQS = '/content/mmseqs/bin/mmseqs'
if not os.path.exists(MMSEQS):
    subprocess.run('wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -O /content/mmseqs.tar.gz '
                   '&& tar xzf /content/mmseqs.tar.gz -C /content', shell=True, check=True)
print('MMseqs2 version:', subprocess.run([MMSEQS, 'version'], capture_output=True, text=True).stdout.strip())

def run_streamed(cmd, cwd):
    """Run a command and stream its output (MMseqs2 prints its own progress)."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end='')
    assert proc.wait() == 0, f'{cmd} failed'

def write_fasta(path, ids, seqs):
    with open(path, 'w') as f:
        for i, s in zip(ids, seqs):
            f.write(f'>{i}\n{s}\n')

ALN_DIR = '/content/external_overlap'
os.makedirs(ALN_DIR, exist_ok=True)
ALN_FILE = os.path.abspath(f'{OUT_DIR}/external_vs_seen_alignments.tsv')   # MMseqs2 runs in ALN_DIR
if not os.path.exists(ALN_FILE):
    write_fasta(f'{ALN_DIR}/external.fasta', ext_proteins['seq_id'], ext_proteins['sequence'])
    write_fasta(f'{ALN_DIR}/seen.fasta', seen['protein_id'], seen['sequence'])
    run_streamed([MMSEQS, 'createdb', f'{ALN_DIR}/external.fasta', f'{ALN_DIR}/qdb'], cwd=ALN_DIR)
    run_streamed([MMSEQS, 'createdb', f'{ALN_DIR}/seen.fasta', f'{ALN_DIR}/tdb'], cwd=ALN_DIR)
    # Fake prefilter (MMseqs2 user guide): every query points to the full list of targets
    subprocess.run(f'ln -sf {ALN_DIR}/tdb.index {ALN_DIR}/pref', shell=True, check=True)
    subprocess.run(f"""awk -v size=$(wc -c < {ALN_DIR}/tdb.index) '{{ print $1"\\t0\\t"size; }}' """
                   f'{ALN_DIR}/qdb.index > {ALN_DIR}/pref.index', shell=True, check=True)
    subprocess.run(f"""awk 'BEGIN {{ printf("%c%c%c%c",7,0,0,0); exit; }}' > {ALN_DIR}/pref.dbtype""",
                   shell=True, check=True)
    run_streamed([MMSEQS, 'align', f'{ALN_DIR}/qdb', f'{ALN_DIR}/tdb', f'{ALN_DIR}/pref', f'{ALN_DIR}/aln',
                  '--alignment-mode', '3', '-e', '0.001', '--min-seq-id', '0', '-c', '0',
                  '--max-accept', '100000', '--threads', str(os.cpu_count())], cwd=ALN_DIR)
    run_streamed([MMSEQS, 'convertalis', f'{ALN_DIR}/qdb', f'{ALN_DIR}/tdb', f'{ALN_DIR}/aln', ALN_FILE,
                  '--format-output', 'query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits'], cwd=ALN_DIR)

aln = pd.read_csv(ALN_FILE, sep='\t', header=None,
                  names=['query', 'target', 'fident', 'alnlen', 'qlen', 'tlen', 'qcov', 'tcov', 'evalue', 'bits'])
aln['id_shortest'] = (aln['fident'] * aln['alnlen']).round() / aln[['qlen', 'tlen']].min(axis=1)

# Per external protein: closest seen protein, and both overlap flags
best = aln.sort_values('id_shortest', ascending=False).drop_duplicates('query').set_index('query')
overlap = ext_proteins[['seq_id', 'accessions', 'sequence']].copy()
overlap['max_identity'] = overlap['seq_id'].map(best['id_shortest']).fillna(0.0)
overlap['closest_seen_protein'] = overlap['seq_id'].map(best['target'])
seen_ids = set(seen['protein_id'].astype(str).str.split('-').str[0])
overlap['same_protein'] = (np.array([any(a.split('-')[0] in seen_ids for a in accs) for accs in overlap['accessions']])
                           | overlap['sequence'].isin(set(seen['sequence'])).to_numpy())
overlap['homolog_30'] = overlap['max_identity'] >= 0.30
assert (overlap.loc[overlap['sequence'].isin(set(seen['sequence'])), 'max_identity'] >= 0.99).all()
overlap.drop(columns='sequence').to_parquet(f'{OUT_DIR}/protein_overlap.parquet', index=False)

flags = overlap.set_index('seq_id')
scored = labels[labels['label'] != 'masked'].copy()
scored['y'] = (scored['label'] == 'positive').astype(int)
scored['same_protein'] = scored['seq_id'].map(flags['same_protein']).to_numpy(bool)
scored['homolog_30'] = scored['seq_id'].map(flags['homolog_30']).to_numpy(bool)
SUBSETS = {'all proteins': lambda d: np.ones(len(d), bool),
           'same protein removed': lambda d: ~d['same_protein'].to_numpy(),
           'homologs >= 30% removed': lambda d: ~d['homolog_30'].to_numpy()}

rows = []
for unit in UNITS:
    d_unit = scored[scored['unit'] == unit]
    for subset, keep_of in SUBSETS.items():
        d = d_unit[keep_of(d_unit)]
        y, p, called = d['y'].to_numpy(), d['probability'].to_numpy(), d['called'].to_numpy()
        tp = int((called & (y == 1)).sum())
        both = 0 < y.sum() < len(y)
        rows.append({'unit': unit, 'subset': subset, 'proteins': d['seq_id'].nunique(), 'sites': len(d),
                     'positive': int(y.sum()), 'positive rate': y.mean() if len(y) else np.nan,
                     'AUROC': roc_auc_score(y, p) if both else np.nan,
                     'AUPRC': average_precision_score(y, p) if both else np.nan,
                     'MCC': matthews_corrcoef(y, called) if both else np.nan,
                     'precision': tp / max(called.sum(), 1), 'recall': tp / max(y.sum(), 1)})
metrics = pd.DataFrame(rows)
metrics.to_csv(f'{OUT_DIR}/metrics.csv', index=False)

for unit in UNITS:
    ids = labels.loc[labels['unit'] == unit, 'seq_id'].unique()
    print(f'{unit}: {len(ids)} proteins   same protein as a seen protein: {int(flags.loc[ids, "same_protein"].sum())}   '
          f'homolog >= 30%: {int(flags.loc[ids, "homolog_30"].sum())}')
print()
print(metrics.to_string(index=False, float_format='{:.4f}'.format))

# %%
import io
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

# ---------------------------------------------------------------------------
# External validation of the final model (figure number set with the manuscript)
#   (A) ROC curves, all studies pooled, for the three subsets: all external proteins; proteins
#   seen in training/validation removed; proteins with a >= 30% homolog in training/validation
#   removed. Markers: the validation MCC cut-off. Grey diagonal: no skill.
#   (B) AUROC per study and pooled, same three subsets; orange vertical line: AUROC on the
#   internal test set (0.809), a reference, not a no-skill line (grey is kept for no skill).
#   Subsets are ordered (each filter stricter than the previous), so they share one blue hue,
#   light to dark, and are also told apart by marker shape.
#   Frontiers figure rules: 180 mm wide, text >= 8 pt and lines >= 2 pt at final size,
#   solid lines only, panels labelled outside the plot area, no red or green; PDF and RGB
#   TIFF at 300 dpi.
# ---------------------------------------------------------------------------
FIG_DIR = 'results/figures/external_validation'
os.makedirs(FIG_DIR, exist_ok=True)
GREY = '#8a8984'
REF = '#d9730d'   # internal test-set AUROC (reference)
TEST_AUROC = 0.8089   # NB05, internal test set
STYLE = {'all proteins': ('#6f9fd8', 'o', 'All proteins', 'All'),
         'same protein removed': ('#2f6dbf', 's', 'Seen proteins removed', 'Seen removed'),
         'homologs >= 30% removed': ('#173f7a', '^', 'Homologs (≥ 30%) removed', 'Homologs removed')}

def save_figure(fig, name, width_mm):
    """PDF (vector, for the LaTeX manuscript) and RGB TIFF at 300 dpi (production format)."""
    assert abs(fig.get_size_inches()[0] * 25.4 - width_mm) < 1e-6
    fig.savefig(f'{FIG_DIR}/{name}.pdf')
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=300)
    Image.open(buffer).convert('RGB').save(f'{FIG_DIR}/{name}.tif', dpi=(300, 300))

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'lines.linewidth': 2,
                     'axes.linewidth': 2, 'xtick.major.width': 2, 'ytick.major.width': 2,
                     'axes.spines.top': False, 'axes.spines.right': False})

fig, (roc_ax, auc_ax) = plt.subplots(1, 2, figsize=(180 / 25.4, 72 / 25.4), layout='constrained',
                                     gridspec_kw={'width_ratios': [1, 1.25]})
pooled = scored[scored['unit'] == 'All studies pooled']
roc_ax.plot([0, 1], [0, 1], color=GREY)
roc_text = []
for subset, keep_of in SUBSETS.items():
    colour, marker, _, label = STYLE[subset]
    d = pooled[keep_of(pooled)]
    y, p, called = d['y'].to_numpy(), d['probability'].to_numpy(), d['called'].to_numpy()
    fpr, tpr, _ = roc_curve(y, p)
    rec_cut = (called & (y == 1)).sum() / y.sum()
    roc_ax.plot(fpr, tpr, color=colour)
    roc_ax.plot((called & (y == 0)).sum() / (y == 0).sum(), rec_cut, marker=marker, color=colour, markersize=6,
                markeredgecolor='white', markeredgewidth=0.8, linestyle='', zorder=4)
    roc_text.append((label, roc_auc_score(y, p), colour))

def value_block(ax, header, values, top):
    """Metric name and one coloured line per subset, right-aligned from the top position."""
    ax.text(0.97, top, header, ha='right', va='top', transform=ax.transAxes, fontweight='bold')
    for k, (name, value, colour) in enumerate(values):
        ax.text(0.97, top - 0.09 * (k + 1), f'{name} {value:.3f}', color=colour, ha='right', va='top',
                transform=ax.transAxes)

value_block(roc_ax, 'AUROC', roc_text, top=0.12 + 0.09 * len(roc_text))

roc_ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='False positive rate', ylabel='True positive rate (recall)')
roc_ax.set_aspect('equal')
roc_ax.set_title('(A) All studies pooled', loc='left')

# (B) AUROC per study and pooled
units = [u for u in UNITS if u != 'All studies pooled'] + ['All studies pooled']
ypos = {u: len(units) - 1 - i for i, u in enumerate(units)}
auc_ax.axvline(TEST_AUROC, color=REF)
auc_ax.text(TEST_AUROC + 0.003, -0.95, f'internal test set ({TEST_AUROC:.3f})', color='#4a4945', ha='left', va='bottom')
offsets = dict(zip(SUBSETS, (0.2, 0.0, -0.2)))
for subset in SUBSETS:
    colour, marker, _, _ = STYLE[subset]
    m = metrics[metrics['subset'] == subset].set_index('unit')
    auc_ax.plot([m.at[u, 'AUROC'] for u in units], [ypos[u] + offsets[subset] for u in units], marker=marker,
                color=colour, markersize=6, markeredgecolor='white', markeredgewidth=0.8, linestyle='')
auc_ax.set_yticks([ypos[u] for u in units])
auc_ax.set_yticklabels([u.replace(' 2025', '').replace(' 2026', '').replace('All studies pooled', 'Pooled')
                        for u in units])
auc_ax.set_ylim(-1.05, len(units) - 0.4)
auc_ax.set_xlim(0.75, 0.95)
auc_ax.set_xlabel('AUROC')
auc_ax.set_title('(B) By study', loc='left')
auc_ax.tick_params(axis='y', length=0)

handles = [Line2D([], [], color=c, marker=m, markersize=6, markeredgecolor='white', markeredgewidth=0.8)
           for c, m, _, _ in STYLE.values()]
handles.append(Line2D([], [], color=GREY))   # the orange reference line is labelled directly in (B)
fig.legend(handles, [s[2] for s in STYLE.values()] + ['No skill'], loc='outside lower center', ncol=4,
           frameon=False)

save_figure(fig, 'figure_external_validation', 180)
plt.show()

# %%
# ---------------------------------------------------------------------------
# Per-site and per-protein results of the external validation (final model)
#   Sites table: every positive site (localized by a study), with its calibrated probability,
#   whether it is called at the validation cut-off, and its rank among the S/T of its protein
#   (1 = highest probability; masked sites excluded).
#   Proteins table: per study and protein, scored S/T, positives, sites called, true
#   positives, false negatives, false positives (unverified negatives called positive),
#   recall, precision and the protein's own AUROC (when it has positives and negatives).
#   Both tables carry the overlap with training/validation proteins (same protein, highest
#   identity, homolog >= 30%). Saved as CSV and as one Excel file with two sheets.
# ---------------------------------------------------------------------------
per_unit = scored[scored['unit'] != 'All studies pooled'].copy()
per_unit['rank_in_protein'] = (per_unit.groupby(['unit', 'seq_id'])['probability']
                               .rank(ascending=False, method='min').astype(int))
per_unit['st_in_protein'] = per_unit.groupby(['unit', 'seq_id'])['position'].transform('size')
info = overlap.set_index('seq_id')[['accessions', 'max_identity', 'closest_seen_protein', 'same_protein', 'homolog_30']]
info['accessions'] = info['accessions'].apply(', '.join)
length_of = ext_proteins.set_index('seq_id')['length']

site_table = (per_unit[per_unit['y'] == 1]
              [['unit', 'seq_id', 'position', 'residue', 'probability', 'called', 'rank_in_protein', 'st_in_protein']]
              .join(info, on='seq_id')
              .rename(columns={'unit': 'study'})
              .sort_values(['study', 'accessions', 'position']))

def protein_row(g):
    y, called = g['y'].to_numpy(), g['called'].to_numpy()
    tp, fp = int((called & (y == 1)).sum()), int((called & (y == 0)).sum())
    return pd.Series({'st_scored': len(g), 'positive': int(y.sum()), 'called': int(called.sum()),
                      'true_positive': tp, 'false_negative': int(y.sum()) - tp, 'false_positive': fp,
                      'recall': tp / y.sum() if y.sum() else np.nan,
                      'precision': tp / called.sum() if called.sum() else np.nan,
                      'protein_AUROC': roc_auc_score(y, g['probability']) if 0 < y.sum() < len(y) else np.nan})

protein_table = (per_unit.groupby(['unit', 'seq_id']).apply(protein_row).reset_index()
                 .join(info, on='seq_id').rename(columns={'unit': 'study'}))
protein_table.insert(2, 'length', protein_table['seq_id'].map(length_of))
protein_table = protein_table.sort_values(['study', 'accessions'])

site_table.to_csv(f'{OUT_DIR}/positive_sites_predictions.csv', index=False)
protein_table.to_csv(f'{OUT_DIR}/protein_results.csv', index=False)
with pd.ExcelWriter(f'{OUT_DIR}/external_validation_details.xlsx') as xl:
    site_table.to_excel(xl, sheet_name='positive sites', index=False)
    protein_table.to_excel(xl, sheet_name='proteins', index=False)

print('Positive sites called at the validation cut-off, per study:')
print(site_table.groupby('study').agg(sites=('called', 'size'), called=('called', 'sum'),
                                      top_ranked_in_protein=('rank_in_protein', lambda r: (r == 1).sum()))
      .to_string())
with_pos = protein_table[protein_table['positive'] > 0]
print('\nProteins with at least one positive site, per study:')
print(with_pos.groupby('study').agg(proteins=('seq_id', 'size'),
                                    all_found=('false_negative', lambda x: (x == 0).sum()),
                                    none_found=('true_positive', lambda x: (x == 0).sum()),
                                    median_protein_AUROC=('protein_AUROC', 'median')).to_string())

# %%
protein_results = pd.read_csv(f'{OUT_DIR}/protein_results.csv')
cols = ['accessions', 'length', 'st_scored', 'positive', 'called', 'true_positive', 'false_negative',
        'false_positive', 'recall', 'precision', 'protein_AUROC', 'same_protein', 'max_identity']
for study, g in protein_results.groupby('study', sort=False):
    print(f'\n===== {study}: {len(g)} proteins')
    print(g[cols].to_string(index=False, float_format='{:.3f}'.format))

# %%
protein_results = pd.read_csv(f'{OUT_DIR}/protein_results.csv')
pr = protein_results.copy()
pr['negative'] = pr['st_scored'] - pr['positive']
pr['true_negative'] = pr['negative'] - pr['false_positive']
pct = lambda a, b: (100 * pr[a] / pr[b]).where(pr[b] > 0)
pr['TP %'] = pct('true_positive', 'positive')    # positives correctly predicted (recall)
pr['FN %'] = pct('false_negative', 'positive')   # positives missed
pr['TN %'] = pct('true_negative', 'negative')    # negatives correctly predicted (specificity)
pr['FP %'] = pct('false_positive', 'negative')   # negatives predicted as positive
pr.to_csv(f'{OUT_DIR}/protein_results_percent.csv', index=False)

cols = ['accessions', 'length', 'positive', 'negative', 'TP %', 'FN %', 'TN %', 'FP %', 'protein_AUROC',
        'same_protein', 'max_identity']
for study, g in pr.groupby('study', sort=False):
    print(f'\n===== {study}: {len(g)} proteins')
    print(g[cols].to_string(index=False, float_format='{:.1f}'.format))

# %%
protein_results = pd.read_csv(f'{OUT_DIR}/protein_results.csv')
pr = protein_results.copy()
pr['negative'] = pr['st_scored'] - pr['positive']
pr['true_negative'] = pr['negative'] - pr['false_positive']
pct = lambda a, b: (100 * pr[a] / pr[b]).where(pr[b] > 0)
pr['TP %'] = pct('true_positive', 'positive')    # positives correctly predicted (recall)
pr['FN %'] = pct('false_negative', 'positive')   # positives missed
pr['TN %'] = pct('true_negative', 'negative')    # negatives correctly predicted (specificity)
pr['FP %'] = pct('false_positive', 'negative')   # negatives predicted as positive
pr.to_csv(f'{OUT_DIR}/protein_results_percent.csv', index=False)

cols = ['accessions', 'length', 'positive', 'negative', 'TP %', 'FN %', 'TN %', 'FP %', 'protein_AUROC',
        'same_protein', 'max_identity']
for study, g in pr.groupby('study', sort=False):
    g = g.sort_values(['positive', 'accessions'], ascending=[False, True])
    print(f'\n===== {study}: {len(g)} proteins')
    print(g[cols].to_string(index=False, float_format='{:.1f}'.format))

# %%
SEED = 0
sample = pr.sample(7, random_state=SEED)
print(f'Random sample of 3 study-protein rows (seed {SEED}):')
print(sample[['study'] + cols].to_string(index=False, float_format='{:.1f}'.format))

# %%
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Specificity and recall vs highest identity to a training/validation protein, one point per protein
#   One column per study (shared axes). x: highest identity of the protein to any training or
#   validation protein (MMseqs2, identical residues / shorter length).
#   Top row: specificity (true negatives / negatives); proteins without negatives omitted.
#   Bottom row: recall (true positives / positives); proteins without positives omitted.
#   Both at the validation cut-off.
# ---------------------------------------------------------------------------
FIG_DIR = 'results/figures/external_validation'
os.makedirs(FIG_DIR, exist_ok=True)
STUDY_COLOURS = {'Lei 2025': '#2a78d6', 'Mouse brain 2025': '#eb6834', 'HCC liver 2026': '#4a3aa7',
                 'NOTISE 2026': '#e87ba4', 'Fu 2026': '#eda100'}
ROWS = [('TN %', 'negative', 'Specificity (%)'), ('TP %', 'positive', 'Recall (%)')]
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False})

fig, axes = plt.subplots(2, len(STUDY_COLOURS), figsize=(180 / 25.4, 110 / 25.4), sharex=True, sharey='row',
                         layout='constrained')
for c, (study, colour) in enumerate(STUDY_COLOURS.items()):
    for r, (metric, denominator, label) in enumerate(ROWS):
        ax = axes[r, c]
        g = pr[(pr['study'] == study) & (pr[denominator] > 0)]
        ax.scatter(g['max_identity'], g[metric], s=10, color=colour, alpha=0.6, linewidths=0)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-3, 103)
        ax.text(0.03, 0.03, f'{len(g)} proteins', transform=ax.transAxes, ha='left', va='bottom', color='#4a4945')
        if c == 0:
            ax.set_ylabel(label)
        if r == 0:
            ax.set_title(study, loc='left')
        else:
            ax.set_xlabel('Highest identity to\ntraining/validation')

fig.savefig(f'{FIG_DIR}/identity_vs_specificity_recall.pdf')
fig.savefig(f'{FIG_DIR}/identity_vs_specificity_recall.png', dpi=300)
plt.show()

# %%
import subprocess

# ---------------------------------------------------------------------------
# Overlap of the external proteins with each existing predictor's training data
#   For the comparison of our model with each predictor on proteins homologous to the training data
#   of neither. Training proteins of each predictor, as in its test-set leakage filter:
#     LM-OGlcNAc-Site      the 5,431 proteins of its data snapshot (NB06);
#     O-GlcNAcPRED-DL      the proteins traced from its development windows, per model (NB07);
#     Seber RNNs           the 4,541 proteins of OVSlab_allSpecies_O-GlcNAcome_PS.csv (NB08);
#     Stack-OglyPred-PLM   the proteins of its training windows (NB10);
#     BSB 2025             its training and validation splits (NB11).
#   Our model: protein_overlap.parquet (cell 1; training and validation proteins). A protein counts as a
#   homolog of a model's training data if an accession (isoform suffix ignored) or its exact sequence is
#   among the model's training proteins, or its identity to one is >= 30% (identical residues / length
#   of the shorter protein; exhaustive MMseqs2 alignment as in cell 1).
#   New files only (tool_overlap.parquet, external_vs_<model>_alignments.tsv).
# ---------------------------------------------------------------------------
TOOLS_DIR = 'data_processed/other_tools'
REPOS = {'seber': ('https://github.com/PedroSeber/O-GlcNAcylation_Prediction', '8d97344c70ab08be008c6bfd9aa507c42d6c279d'),
         'bsb2025': ('https://github.com/LBS-UFMG/OGlcNAc', 'c1d6d351b1339562337fe1d7fc6b68462158d2ba')}
for name, (url, commit) in REPOS.items():
    path = f'/content/repo_{name}'
    if not os.path.exists(path):
        subprocess.run(f'git clone -q {url} {path}', shell=True, check=True)
    subprocess.run(f'git -C {path} checkout -q {commit}', shell=True, check=True)

def read_fasta(path):
    seqs, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith('>'):
            name = line[1:]
            seqs[name] = ''
        elif name:
            seqs[name] += line
    return seqs

def accession_of(header):
    token = header.split()[0]
    return token.split('|')[1] if token.count('|') >= 2 else token

# Training proteins of each model: (ids, sequences, accessions)
training_sets = {}
snapshot = read_fasta(f'{TOOLS_DIR}/LM-OGlcNAc-Site/training_data_git/snapshot_proteins.fasta')
training_sets['LM-OGlcNAc-Site'] = (list(snapshot), list(snapshot.values()), set(snapshot))
server = read_fasta(f'{TOOLS_DIR}/O-GlcNAcPRED-DL/training_data_server/seen_proteins_server.fasta')   # id|species|accessions
for species in ('human', 'mouse'):
    hits = {h: s for h, s in server.items() if species in h.split('|')[1]}
    accs = {a for h in hits for a in h.split('|')[2].split(';') if a}
    training_sets[f'O-GlcNAcPRED-DL {species}'] = ([h.split('|')[0] for h in hits], list(hits.values()), accs)
seber = pd.read_csv('/content/repo_seber/OVSlab_allSpecies_O-GlcNAcome_PS.csv', usecols=['UniprotKB ID', 'sequence'])
training_sets['Seber'] = (list(seber['UniprotKB ID']), list(seber['sequence']), set(seber['UniprotKB ID']))
stack = read_fasta(f'{TOOLS_DIR}/Stack-OglyPred-PLM/their_training_proteins_uniprot.fasta')
training_sets['Stack-OglyPred-PLM'] = ([accession_of(h) for h in stack], list(stack.values()), {accession_of(h) for h in stack})
bsb = pd.concat([pd.read_csv(f'/content/repo_bsb2025/data_processed/02_{s}_dataset_balanced.csv', usecols=['accession', 'sequence'])
                 for s in ('training', 'validation')]).drop_duplicates()
training_sets['BSB 2025'] = (list(bsb['accession']), list(bsb['sequence']), set(bsb['accession']))
for tool, (ids, seqs, accs) in training_sets.items():
    print(f'{tool}: {len(seqs):,} training sequences, {len(accs):,} accessions')

MMSEQS = '/content/mmseqs/bin/mmseqs'
if not os.path.exists(MMSEQS):
    subprocess.run('wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -O /content/mmseqs.tar.gz '
                   '&& tar xzf /content/mmseqs.tar.gz -C /content', shell=True, check=True)
print('MMseqs2 version:', subprocess.run([MMSEQS, 'version'], capture_output=True, text=True).stdout.strip())

def run_streamed(cmd, cwd):
    """Run a command and stream its output (MMseqs2 prints its own progress)."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end='')
    assert proc.wait() == 0, f'{cmd} failed'

def write_fasta(path, ids, seqs):
    with open(path, 'w') as f:
        for i, s in zip(ids, seqs):
            f.write(f'>{i}\n{s}\n')

def max_identity_to(tool, ids, seqs):
    """Highest identity of each external protein to the model's training proteins (exhaustive MMseqs2)."""
    stem = tool.replace(' ', '_')
    aln_dir = f'/content/external_overlap_{stem}'
    aln_file = os.path.abspath(f'{OUT_DIR}/external_vs_{stem}_alignments.tsv')
    os.makedirs(aln_dir, exist_ok=True)
    if not os.path.exists(aln_file):
        target_ids = [f't{k}' for k in range(len(seqs))]   # unique target names
        write_fasta(f'{aln_dir}/external.fasta', ext_proteins['seq_id'], ext_proteins['sequence'])
        write_fasta(f'{aln_dir}/training.fasta', target_ids, seqs)
        run_streamed([MMSEQS, 'createdb', f'{aln_dir}/external.fasta', f'{aln_dir}/qdb'], cwd=aln_dir)
        run_streamed([MMSEQS, 'createdb', f'{aln_dir}/training.fasta', f'{aln_dir}/tdb'], cwd=aln_dir)
        # Fake prefilter (MMseqs2 user guide): every query points to the full list of targets
        subprocess.run(f'ln -sf {aln_dir}/tdb.index {aln_dir}/pref', shell=True, check=True)
        subprocess.run(f"""awk -v size=$(wc -c < {aln_dir}/tdb.index) '{{ print $1"\\t0\\t"size; }}' """
                       f'{aln_dir}/qdb.index > {aln_dir}/pref.index', shell=True, check=True)
        subprocess.run(f"""awk 'BEGIN {{ printf("%c%c%c%c",7,0,0,0); exit; }}' > {aln_dir}/pref.dbtype""",
                       shell=True, check=True)
        run_streamed([MMSEQS, 'align', f'{aln_dir}/qdb', f'{aln_dir}/tdb', f'{aln_dir}/pref', f'{aln_dir}/aln',
                      '--alignment-mode', '3', '-e', '0.001', '--min-seq-id', '0', '-c', '0',
                      '--max-accept', '100000', '--threads', str(os.cpu_count())], cwd=aln_dir)
        run_streamed([MMSEQS, 'convertalis', f'{aln_dir}/qdb', f'{aln_dir}/tdb', f'{aln_dir}/aln', aln_file,
                      '--format-output', 'query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits'], cwd=aln_dir)
    aln = pd.read_csv(aln_file, sep='\t', header=None,
                      names=['query', 'target', 'fident', 'alnlen', 'qlen', 'tlen', 'qcov', 'tcov', 'evalue', 'bits'])
    aln['id_shortest'] = (aln['fident'] * aln['alnlen']).round() / aln[['qlen', 'tlen']].min(axis=1)
    return ext_proteins['seq_id'].map(aln.groupby('query')['id_shortest'].max()).fillna(0.0).to_numpy()

base = lambda acc: str(acc).split('-')[0]
ext_accs = [{base(a) for a in accs} for accs in ext_proteins['accessions']]
tool_overlap = ext_proteins[['seq_id']].copy()
ours = pd.read_parquet(f'{OUT_DIR}/protein_overlap.parquet').set_index('seq_id')
tool_overlap['max_identity_ours'] = tool_overlap['seq_id'].map(ours['max_identity']).to_numpy()
tool_overlap['homolog_ours'] = (tool_overlap['seq_id'].map(ours['same_protein']).to_numpy(bool)
                             | tool_overlap['seq_id'].map(ours['homolog_30']).to_numpy(bool))
for tool, (ids, seqs, accs) in training_sets.items():
    print(f'\n--- {tool}')
    identity = max_identity_to(tool, ids, seqs)
    acc_base = {base(a) for a in accs}
    same = (np.array([bool(a & acc_base) for a in ext_accs]) | ext_proteins['sequence'].isin(set(seqs)).to_numpy())
    tool_overlap[f'max_identity_{tool}'] = identity
    tool_overlap[f'homolog_{tool}'] = same | (identity >= 0.30)
tool_overlap.to_parquet(f'{OUT_DIR}/tool_overlap.parquet', index=False)

# Proteins per unit homologous to each model's training data (labels from cell 0)
homolog_cols = [c for c in tool_overlap.columns if c.startswith('homolog_')]
flags = tool_overlap.set_index('seq_id')[homolog_cols]
rows = []
for unit in UNITS:
    ids = labels.loc[labels['unit'] == unit, 'seq_id'].unique()
    rows.append({'unit': unit, 'proteins': len(ids), **{c[8:]: int(flags.loc[ids, c].sum()) for c in homolog_cols}})
print("\nExternal proteins homologous to each model's training data (same accession or sequence, or identity >= 30%):")
print(pd.DataFrame(rows).to_string(index=False))

# %%
import io
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

# ---------------------------------------------------------------------------
# Comparison of all models on every protein of each external set (main external result)
#   No filtering: a model that has seen a protein (or a homolog) in training keeps that advantage,
#   as it would on a new experiment. Units and labels as in cell 0 (masked sites excluded).
#   Scores (each tool's external_scores.parquet; NB06-NB11) and yes/no calls at each tool's own
#   cut-off: ours, calibrated probability >= 0.0821 (validation MCC); LM-OGlcNAc-Site (ESM-2 layer
#   33), mean of its three probabilities, call = >= 2 of 3 networks; O-GlcNAcPRED-DL, >= 0.5, the
#   organism-matched model run on each study's own file (pooled set: mean of a site's study scores);
#   Stack-OglyPred-PLM, second output, call = their rule; Seber RNNs, >= 0.60; BSB 2025, >= 0.5 (every
#   protein at full length). A model whose scores are not on Drive yet is left out.
#   Outputs: metrics_all_tools.csv and a figure of AUROC (one panel per unit).
# ---------------------------------------------------------------------------
TOOLS_DIR = 'data_processed/other_tools'
FIG_DIR = 'results/figures/external_validation'
os.makedirs(FIG_DIR, exist_ok=True)
METRIC = 'AUROC'

scored = labels[labels['label'] != 'masked'][['unit', 'seq_id', 'position', 'label', 'probability', 'called']].copy()
scored['y'] = (scored['label'] == 'positive').astype(int)
key = ['seq_id', 'position']

# Each model: unit -> (score, call) aligned with `scored`
def merged(df, cols):
    out = scored[['unit'] + key].merge(df[key + cols], on=key, how='left', validate='many_to_one')
    assert out[cols].notna().all().all(), 'sites without a score'
    return out

model_cols = {'Our model': (scored['probability'].to_numpy(), scored['called'].to_numpy(bool))}
lm_file = f'{TOOLS_DIR}/LM-OGlcNAc-Site/external_scores.parquet'
if os.path.exists(lm_file):
    lm = merged(pd.read_parquet(lm_file), ['score_layer33', 'call_layer33'])
    model_cols['LM-OGlcNAc-Site'] = (lm['score_layer33'].to_numpy(), lm['call_layer33'].to_numpy() == 1)
pdl = pd.read_parquet(f'{TOOLS_DIR}/O-GlcNAcPRED-DL/external_scores.parquet')
pdl_score = np.full(len(scored), np.nan)
for unit in UNITS:
    rows = (scored['unit'] == unit).to_numpy()
    runs = pdl.groupby(key, as_index=False)['score'].mean() if unit == 'All studies pooled' else pdl[pdl['study'] == unit]
    pdl_score[rows] = scored.loc[rows, key].merge(runs[key + ['score']], on=key, how='left', validate='one_to_one')['score'].to_numpy()
assert np.isfinite(pdl_score).all()
model_cols['O-GlcNAcPRED-DL'] = (pdl_score, pdl_score >= 0.5)
stack = merged(pd.read_parquet(f'{TOOLS_DIR}/Stack-OglyPred-PLM/external_scores.parquet'), ['score', 'called'])
model_cols['Stack-OglyPred-PLM'] = (stack['score'].to_numpy(), stack['called'].to_numpy(bool))
seber = merged(pd.read_parquet(f'{TOOLS_DIR}/Seber/external_scores.parquet'), ['score_rnn600_75', 'score_rnn225'])
model_cols['RNN-[600,75] (Seber)'] = (seber['score_rnn600_75'].to_numpy(), seber['score_rnn600_75'].to_numpy() >= 0.60)
model_cols['RNN-225 (Seber and Braatz)'] = (seber['score_rnn225'].to_numpy(), seber['score_rnn225'].to_numpy() >= 0.60)
bsb = merged(pd.read_parquet(f'{TOOLS_DIR}/BSB2025/external_scores.parquet'), ['score', 'called'])
model_cols['BSB 2025 (Arcanjo et al.)'] = (bsb['score'].to_numpy(), bsb['called'].to_numpy(bool))
MODELS = list(model_cols)

rows = []
for unit in UNITS:
    in_unit = (scored['unit'] == unit).to_numpy()
    y = scored.loc[in_unit, 'y'].to_numpy()
    for model, (score, called) in model_cols.items():
        s, c = score[in_unit], called[in_unit]
        tp = int((c & (y == 1)).sum())
        rows.append({'unit': unit, 'model': model, 'proteins': scored.loc[in_unit, 'seq_id'].nunique(), 'sites': len(y),
                     'positive': int(y.sum()), 'positive rate': y.mean(), 'AUROC': roc_auc_score(y, s),
                     'AUPRC': average_precision_score(y, s), 'MCC': matthews_corrcoef(y, c),
                     'precision': tp / max(c.sum(), 1), 'recall': tp / y.sum(), 'called positive': int(c.sum())})
all_tools = pd.DataFrame(rows)
all_tools.to_csv(f'{OUT_DIR}/metrics_all_tools.csv', index=False)
for m in ('AUROC', 'AUPRC', 'MCC'):
    print(f'\n{m}:')
    print(all_tools.pivot(index='model', columns='unit', values=m).loc[MODELS, UNITS].to_string(float_format='{:.3f}'.format))

# Figure: one panel per unit, models in a fixed order; ours blue, the other tools grey
# (Frontiers: 180 mm wide, text >= 8 pt, panels labelled (A)-(F), no red or green; PDF and RGB TIFF at 300 dpi)
# Model names as written in the manuscript (display only; the CSV keeps the names above)
DISPLAY_NAMES = {'Our model': 'O-GlcNAcFinder', 'RNN-[600,75] (Seber)': 'RNN-[600,75]',
                 'RNN-225 (Seber and Braatz)': 'RNN-225', 'BSB 2025 (Arcanjo et al.)': 'Arcanjo et al. (2025)'}
# Study names as cited in the manuscript (first author and year; display only, the CSV keeps the units above)
STUDY_NAMES = {'Lei 2025': 'Lei et al. (2025)', 'Mouse brain 2025': 'Hou et al. (2025)',
               'HCC liver 2026': 'Hou et al. (2026)', 'NOTISE 2026': 'Griffin et al. (2026)',
               'Fu 2026': 'Fu et al. (2026)', 'All studies pooled': 'All studies pooled'}
assert set(UNITS) == set(STUDY_NAMES), set(UNITS) ^ set(STUDY_NAMES)
OUR_COLOUR, TOOL_COLOUR = '#2a78d6', '#8a8984'
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False})
ypos = np.arange(len(MODELS))[::-1]
values = all_tools.set_index(['unit', 'model'])
fig, axes = plt.subplots(2, 3, figsize=(180 / 25.4, 135 / 25.4), sharex=True, sharey=True, layout='constrained')
hi = min(1.0, values[METRIC].max() + 0.08)
for k, (ax, unit) in enumerate(zip(axes.flat, UNITS)):
    v = values.loc[unit].loc[MODELS]
    colours = [OUR_COLOUR if m == 'Our model' else TOOL_COLOUR for m in MODELS]
    ax.hlines(ypos, 0.5, v[METRIC], color=colours, linewidth=1.5)
    ax.scatter(v[METRIC], ypos, color=colours, s=22, zorder=3)
    best = v[METRIC].round(3).max()   # highest AUROC of the panel, in bold
    for yp, val in zip(ypos, v[METRIC]):
        ax.annotate(f'{val:.3f}', (val, yp), xytext=(5, 0), textcoords='offset points', va='center', fontsize=8,
                    fontweight='bold' if round(val, 3) == best else 'normal')
    ax.set_title(f'({chr(65 + k)}) {STUDY_NAMES[unit]}\n{int(v["proteins"].iloc[0]):,} proteins\n'
                 f'{int(v["positive"].iloc[0]):,} positive sites', loc='left')
    ax.set_xlim(0.5, hi)
    ax.set_yticks(ypos, [DISPLAY_NAMES.get(m, m) for m in MODELS])
    ax.tick_params(axis='y', length=0)
for ax in axes[-1]:
    ax.set_xlabel(f'{METRIC} (0.5 = no skill)')

fig.savefig(f'{FIG_DIR}/figure_external_all_tools_{METRIC}.pdf')
buffer = io.BytesIO()
fig.savefig(buffer, format='png', dpi=300)
Image.open(buffer).convert('RGB').save(f'{FIG_DIR}/figure_external_all_tools_{METRIC}.tif', dpi=(300, 300))
plt.show()

# %%
# ---------------------------------------------------------------------------
# Comparison of our model with each predictor on external proteins with < 30% identity to both
# models' training data
#   For each predictor and unit: the proteins homologous to the training data of neither model are kept
#   (tool_overlap.parquet: homolog = same accession or sequence as, or identity >= 30% to, a training
#   protein), and both models are scored on the same sites. O-GlcNAcPRED-DL: the training proteins of
#   the model applied to the unit (mouse model for the mouse study, human model for the others; pooled
#   set: either model). Uses the scores, calls, units and display names of the previous cell. Units with
#   fewer than MIN_POSITIVES positive sites are listed in the table but not drawn.
#   Outputs: metrics_pairwise_homologs_removed.csv and a figure of AUROC (one panel per predictor).
# ---------------------------------------------------------------------------
METRIC = 'AUROC'
MIN_POSITIVES = 15
overlap = pd.read_parquet(f'{OUT_DIR}/tool_overlap.parquet').set_index('seq_id')
HOMOLOG = {'LM-OGlcNAc-Site': ['homolog_LM-OGlcNAc-Site'], 'Stack-OglyPred-PLM': ['homolog_Stack-OglyPred-PLM'],
           'RNN-[600,75] (Seber)': ['homolog_Seber'], 'RNN-225 (Seber and Braatz)': ['homolog_Seber'],
           'BSB 2025 (Arcanjo et al.)': ['homolog_BSB 2025']}
def homolog_columns(tool, unit):
    if tool != 'O-GlcNAcPRED-DL':
        return HOMOLOG[tool]
    if unit == 'All studies pooled':
        return ['homolog_O-GlcNAcPRED-DL human', 'homolog_O-GlcNAcPRED-DL mouse']
    return ['homolog_O-GlcNAcPRED-DL mouse'] if unit == 'Mouse brain 2025' else ['homolog_O-GlcNAcPRED-DL human']

homolog_ours = scored['seq_id'].map(overlap['homolog_ours']).to_numpy(bool)
TOOLS = [m for m in MODELS if m != 'Our model']
rows = []
for tool in TOOLS:
    for unit in UNITS:
        homolog_tool = scored['seq_id'].map(overlap[homolog_columns(tool, unit)].any(axis=1)).to_numpy(bool)
        keep = (scored['unit'] == unit).to_numpy() & ~homolog_ours & ~homolog_tool
        y = scored.loc[keep, 'y'].to_numpy()
        both = 0 < y.sum() < len(y)
        row = {'tool': tool, 'unit': unit, 'proteins': scored.loc[keep, 'seq_id'].nunique(), 'sites': int(keep.sum()),
               'positive': int(y.sum()), 'drawn': bool(y.sum() >= MIN_POSITIVES)}
        for who, (score, called) in [('tool', model_cols[tool]), ('ours', model_cols['Our model'])]:
            s, c = score[keep], called[keep]
            tp = int((c & (y == 1)).sum())
            row.update({f'AUROC {who}': roc_auc_score(y, s) if both else np.nan,
                        f'AUPRC {who}': average_precision_score(y, s) if both else np.nan,
                        f'MCC {who}': matthews_corrcoef(y, c) if both else np.nan,
                        f'precision {who}': tp / max(c.sum(), 1), f'recall {who}': tp / max(y.sum(), 1)})
        rows.append(row)
pairwise = pd.DataFrame(rows)
pairwise.to_csv(f'{OUT_DIR}/metrics_pairwise_homologs_removed.csv', index=False)
cols = ['unit', 'proteins', 'positive'] + [f'{m} {w}' for m in ('AUROC', 'AUPRC', 'MCC') for w in ('tool', 'ours')]
for tool in TOOLS:
    print(f'\n{tool} vs our model, proteins with < 30% identity to both training sets:')
    print(pairwise.loc[pairwise['tool'] == tool, cols].to_string(index=False, float_format='{:.3f}'.format))
drawn = pairwise[pairwise['drawn']]
for m in ('AUROC', 'AUPRC', 'MCC'):
    print(f'{m}: ours higher in {(drawn[f"{m} ours"] > drawn[f"{m} tool"]).sum()} of {len(drawn)} '
          f'comparisons with >= {MIN_POSITIVES} positive sites')

# Figure: one panel per predictor; per unit, the predictor (grey) and our model (blue) on the same sites
# (Frontiers: 180 mm wide, text >= 8 pt, panels labelled (A)-(F), no red or green; PDF and RGB TIFF at 300 dpi)
n_cols = 3
n_rows = -(-len(TOOLS) // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(180 / 25.4, 52 * n_rows / 25.4), sharex=True, sharey=True,
                         layout='constrained', squeeze=False)
ypos = np.arange(len(UNITS))[::-1]
lo = 0.5 if METRIC == 'AUROC' else 0.0
hi = min(1.0, np.nanmax(drawn[[f'{METRIC} tool', f'{METRIC} ours']].to_numpy()) + 0.05)
for k, (ax, tool) in enumerate(zip(axes.flat, TOOLS)):
    d = pairwise[pairwise['tool'] == tool].set_index('unit').loc[UNITS]
    for yp, (unit, r) in zip(ypos, d.iterrows()):
        if not r['drawn']:
            ax.text(lo + 0.01, yp, f'< {MIN_POSITIVES} positive sites', va='center', fontsize=8, color=TOOL_COLOUR)
            continue
        ax.plot([r[f'{METRIC} tool'], r[f'{METRIC} ours']], [yp, yp], color='#c9c8c3', linewidth=1.5, zorder=1)
        ax.scatter(r[f'{METRIC} tool'], yp, color=TOOL_COLOUR, s=22, zorder=3)
        ax.scatter(r[f'{METRIC} ours'], yp, color=OUR_COLOUR, s=22, zorder=3)
        ax.annotate(f'{int(r["positive"]):,}', (1, yp), xycoords=('axes fraction', 'data'), xytext=(3, 0),
                    textcoords='offset points', va='center', fontsize=8, color=TOOL_COLOUR)
    ax.set_title(f'({chr(65 + k)}) {DISPLAY_NAMES.get(tool, tool)}', loc='left')
    ax.set_xlim(lo, hi)
    ax.set_yticks(ypos, [STUDY_NAMES[u] for u in UNITS])
    ax.tick_params(axis='y', length=0)
for ax in list(axes.flat)[len(TOOLS):]:
    ax.axis('off')
for k, ax in enumerate(list(axes.flat)[:len(TOOLS)]):
    if k + n_cols >= len(TOOLS):   # lowest panel of its column: x-axis numbers and label
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_xlabel(f'{METRIC} (0.5 = no skill)' if METRIC == 'AUROC' else METRIC)
handles = [plt.Line2D([], [], marker='o', linestyle='', color=c, label=l)
           for c, l in [(TOOL_COLOUR, 'Other tool'), (OUR_COLOUR, 'O-GlcNAcFinder')]]
fig.legend(handles=handles, loc='outside lower center', ncols=2, frameon=False,
           title='Proteins with less than 30% identity to the training proteins of both models. Numbers: positive sites.')

fig.savefig(f'{FIG_DIR}/figure_external_pairwise_homologs_removed_{METRIC}.pdf')
buffer = io.BytesIO()
fig.savefig(buffer, format='png', dpi=300)
Image.open(buffer).convert('RGB').save(f'{FIG_DIR}/figure_external_pairwise_homologs_removed_{METRIC}.tif', dpi=(300, 300))
plt.show()

# %%
import os
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))

import io
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

# ---------------------------------------------------------------------------
# External validation per species (final model)
#   Species: the organism of each external protein as reported by its study (NB01). Labels: the
#   pooled unit of cell 0 restricted to the species' proteins (a site is positive if any study of
#   that species localized it; masked sites excluded). Two subsets, as in cell 1: all proteins, and
#   homologs removed (proteins with the same accession or sequence as, or identity >= 30% to, a
#   training or validation protein). Per species: AUROC, AUPRC, and MCC, precision and recall at the
#   validation cut-off, with the number of training proteins and positive sites of the same species.
#   Outputs: metrics_by_species.csv and a figure.
# ---------------------------------------------------------------------------
OUT_DIR = 'data_processed/external_validation'
FIG_DIR = 'results/figures/external_validation'
os.makedirs(FIG_DIR, exist_ok=True)

labels = pd.read_parquet(f'{OUT_DIR}/labels_and_scores.parquet')
pooled = labels[(labels['unit'] == 'All studies pooled') & (labels['label'] != 'masked')].copy()
pooled['y'] = (pooled['label'] == 'positive').astype(int)
CUTOFF = torch.load('data_processed/final_model/final_model.pt', map_location='cpu')['cutoff']['probability']
assert (pooled['called'] == (pooled['probability'] >= CUTOFF)).all()

ext_sites = pd.read_parquet('data_processed/external_sites_mapped.parquet')
species_of = ext_sites.groupby('seq_id')['organism'].agg(lambda x: sorted(set(x)))
several = species_of[species_of.str.len() > 1]
print(f'External proteins reported under several species: {len(several)}'
      + (f' ({", ".join(f"{s}: {v}" for s, v in several.items())}; first in alphabetical order used)' if len(several) else ''))
pooled['organism'] = pooled['seq_id'].map(species_of.str[0])
assert pooled['organism'].notna().all()

overlap = pd.read_parquet(f'{OUT_DIR}/protein_overlap.parquet').set_index('seq_id')
pooled['homolog'] = (pooled['seq_id'].map(overlap['same_protein']).to_numpy(bool)
                     | pooled['seq_id'].map(overlap['homolog_30']).to_numpy(bool))

proteins = pd.read_parquet('data_processed/atlas5_proteins.parquet')
train = proteins[proteins['split'] == 'train']
train_counts = (train.assign(n_pos=train['positive'].str.len()).groupby('organism')
                .agg(**{'training proteins': ('protein_id', 'size'), 'training positive sites': ('n_pos', 'sum')}))

SUBSETS = {'all proteins': lambda d: np.ones(len(d), bool),
           'homologs (≥ 30% identity) removed': lambda d: ~d['homolog'].to_numpy()}
rows = []
for organism, d_org in pooled.groupby('organism'):
    for subset, keep_of in SUBSETS.items():
        d = d_org[keep_of(d_org)]
        y, p, called = d['y'].to_numpy(), d['probability'].to_numpy(), d['called'].to_numpy(bool)
        tp = int((called & (y == 1)).sum())
        both = 0 < y.sum() < len(y)
        rows.append({'organism': organism, 'subset': subset, 'proteins': d['seq_id'].nunique(), 'sites': len(d),
                     'positive': int(y.sum()), 'positive rate': y.mean() if len(y) else np.nan,
                     'AUROC': roc_auc_score(y, p) if both else np.nan,
                     'AUPRC': average_precision_score(y, p) if both else np.nan,
                     'MCC': matthews_corrcoef(y, called) if both else np.nan,
                     'precision': tp / max(called.sum(), 1), 'recall': tp / max(y.sum(), 1)})
by_species = (pd.DataFrame(rows).join(train_counts, on='organism')
              .fillna({'training proteins': 0, 'training positive sites': 0})
              .astype({'training proteins': int, 'training positive sites': int}))
by_species.to_csv(f'{OUT_DIR}/metrics_by_species.csv', index=False)
print(by_species.to_string(index=False, float_format='{:.3f}'.format))

# Figure: AUROC per species, both subsets (one blue hue, light to dark, and two marker shapes),
# species with most positive sites at the top
# (Frontiers: 180 mm wide, text >= 8 pt, no red or green; PDF and RGB TIFF at 300 dpi)
STYLE = {'all proteins': ('#8fb8ea', 'o'), 'homologs (≥ 30% identity) removed': ('#1f5fa8', 's')}
order = (by_species[by_species['subset'] == 'all proteins'].sort_values('positive')['organism']).tolist()
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False})
height = len(order) * 10 + 32
fig, ax = plt.subplots(figsize=(180 / 25.4, height / 25.4))
fig.subplots_adjust(left=0.22, right=0.60, top=1 - 10 / height, bottom=22 / height)
ypos = {o: k for k, o in enumerate(order)}
for subset, (colour, marker) in STYLE.items():
    d = by_species[by_species['subset'] == subset]
    offset = 0.15 if subset == 'all proteins' else -0.15
    y = [ypos[o] + offset for o in d['organism']]
    ax.scatter(d['AUROC'], y, color=colour, marker=marker, s=24, zorder=3, label=subset[0].upper() + subset[1:])
    for yy, val in zip(y, d['AUROC']):
        ax.annotate(f'{val:.3f}', (val, yy), xytext=(5, 0), textcoords='offset points', va='center', fontsize=8)
    for yy, (_, r) in zip(y, d.iterrows()):
        ax.annotate(f'{r["positive"]:,} ({r["proteins"]:,})', (1.32, yy), xycoords=('axes fraction', 'data'),
                    ha='right', va='center', fontsize=8)
for o in order:
    r = by_species[(by_species['organism'] == o) & (by_species['subset'] == 'all proteins')].iloc[0]
    ax.annotate(f'{r["training positive sites"]:,} ({r["training proteins"]:,})', (1.88, ypos[o]),
                xycoords=('axes fraction', 'data'), ha='right', va='center', fontsize=8)
for x, text in [(1.32, 'External positive sites\n(proteins)'), (1.88, 'Training positive sites\n(proteins)')]:
    ax.annotate(text, (x, 1), xycoords='axes fraction', xytext=(0, 4), textcoords='offset points',
                ha='right', va='bottom', fontsize=8, fontweight='bold')
ax.set_yticks(list(ypos.values()), [f'$\\it{{{o.replace(" ", "~")}}}$' for o in order])   # species names in italics
ax.set_ylim(-0.6, len(order) - 0.4)
ax.tick_params(axis='y', length=0)
ax.set_xlim(0.5, 1.0)
ax.set_xlabel('AUROC (0.5 = no skill)')
fig.legend(loc='lower center', bbox_to_anchor=(0.41, 0), ncols=2, frameon=False, fontsize=8)

fig.savefig(f'{FIG_DIR}/figure_external_by_species_AUROC.pdf')
buffer = io.BytesIO()
fig.savefig(buffer, format='png', dpi=300)
Image.open(buffer).convert('RGB').save(f'{FIG_DIR}/figure_external_by_species_AUROC.tif', dpi=(300, 300))
plt.show()

# %%
from scipy.stats import hypergeom

# ---------------------------------------------------------------------------
# Top-k hit rate of the final model within proteins (test set and external sets)
#   Setting: a protein known to carry O-GlcNAc whose site is unknown. Within each protein with
#   >= 1 positive site, its S/T are ranked by calibrated probability (masked S/T excluded); a
#   protein is a hit at k if >= 1 positive site is among its top k S/T. Ties at the k-th place are
#   broken at random: the expected hit rate over every order of the tied S/T is computed exactly,
#   as for the other predictors (next cell).
#   Random expectation per protein (k S/T drawn without replacement from its n S/T, m of them
#   positive): P(>= 1 positive) = 1 - hypergeometric P(0); enrichment = observed / expected hit rate.
#   Test set: the held-out test split (NB05 scores); its proteins have < 30% identity to training and
#   validation by construction (GraphPart). External sets: all proteins, and proteins with a >= 30%
#   homolog in training/validation removed (cell 1).
#   Unlabelled S/T are not proven negatives, so hit rates are lower bounds.
#   K_VALUES: k values computed (all saved in the CSV files); K_MAIN: k reported in the text and figures;
#   printed: k = 1, 3, 5 and K_MAIN.
#   Uses `scored` and UNITS of the comparison cell. Outputs: topk_hit_rate.csv, topk_hit_rate_by_size.csv.
# ---------------------------------------------------------------------------
K_VALUES = range(1, 21)
K_MAIN = 10
SIZE_BINS = [0, 50, 150, 300, np.inf]   # S/T per protein, for the stratified table
SIZE_LABELS = ['< 50', '50-149', '150-299', '>= 300']

overlap = pd.read_parquet(f'{OUT_DIR}/protein_overlap.parquet').set_index('seq_id')
homolog = (overlap['same_protein'] | overlap['homolog_30']).astype(bool)

def p_none(n, m, r):
    """Probability that r S/T drawn without replacement from n (m of them positive) contain no positive."""
    p = 1.0
    for i in range(r):
        p *= (n - m - i) / (n - i)
        if p <= 0:
            return 0.0
    return p

def hit_probability(score, y, k):
    """Expected P(>= 1 positive among the top k), with ties at the k-th score broken at random."""
    k = min(k, len(score))
    threshold = np.sort(score)[::-1][k - 1]
    above, tied = score > threshold, score == threshold
    if y[above].any():
        return 1.0
    return 1.0 - p_none(int(tied.sum()), int(y[tied].sum()), k - int(above.sum()))

def per_protein(d):
    """One row per protein with >= 1 positive: n S/T, m positives and the expected hit at each k."""
    rows = {}
    for sid, g in d.groupby('seq_id', sort=False):
        y, s = g['y'].to_numpy(), g['probability'].to_numpy()
        if y.sum() == 0:
            continue
        rows[sid] = {'n': len(y), 'm': int(y.sum()), **{f'hit@{k}': hit_probability(s, y, k) for k in K_VALUES}}
    return pd.DataFrame.from_dict(rows, orient='index')

def summarize(p):
    row = {'proteins': len(p), 'median S/T': p['n'].median(), 'median positives': p['m'].median()}
    for k in K_VALUES:
        observed = p[f'hit@{k}'].mean()
        expected = (1 - hypergeom.pmf(0, p['n'], p['m'], np.minimum(k, p['n']))).mean()
        row.update({f'hit@{k}': observed, f'random@{k}': expected, f'enrichment@{k}': observed / expected})
    return row

test = pd.read_parquet('data_processed/final_model/test_scores.parquet').rename(columns={'protein_id': 'seq_id'})
p = per_protein(test)
rows = [{'unit': 'Test set', 'subset': 'held-out split (< 30% identity)', **summarize(p)}]
size_sets = {'Test set': p}
for unit in UNITS:
    d = scored[scored['unit'] == unit]
    for subset, keep in [('all proteins', np.ones(len(d), bool)),
                         ('homologs (>= 30% identity) removed', ~d['seq_id'].map(homolog).to_numpy(bool))]:
        p = per_protein(d[keep])
        rows.append({'unit': unit, 'subset': subset, **summarize(p)})
        if unit == 'All studies pooled':
            size_sets[f'External pooled, {subset}'] = p
topk = pd.DataFrame(rows)
if os.path.exists(f'{OUT_DIR}/topk_hit_rate.csv'):
    previous = pd.read_csv(f'{OUT_DIR}/topk_hit_rate.csv')
    diff = (previous[[f'hit@{k}' for k in K_VALUES]] - topk[[f'hit@{k}' for k in K_VALUES]]).abs().max().max()
    print(f'Previous version (ties broken by position) vs this one: hit rate max |diff| {diff:.4f}\n')
topk.to_csv(f'{OUT_DIR}/topk_hit_rate.csv', index=False)

cols = ['unit', 'subset', 'proteins'] + [f'{c}@{k}' for k in (1, 3, 5, K_MAIN) for c in ('hit', 'random', 'enrichment')]
with pd.option_context('display.width', 250):
    print(topk[cols].to_string(index=False, float_format='{:.3f}'.format))

# Test set and pooled external set, by protein size (long proteins have more S/T to choose from)
print('\nBy number of S/T per protein:')
size_rows = []
for name, p in size_sets.items():
    for label, part in p.groupby(pd.cut(p['n'], SIZE_BINS, labels=SIZE_LABELS, right=False), observed=True):
        size_rows.append({'set': name, 'S/T per protein': label, **summarize(part)})
size_table = pd.DataFrame(size_rows)
size_table.to_csv(f'{OUT_DIR}/topk_hit_rate_by_size.csv', index=False)
with pd.option_context('display.width', 250):
    print(size_table[['set', 'S/T per protein', 'proteins'] + [f'{c}@{k}' for k in (1, 3, 5, K_MAIN) for c in ('hit', 'random', 'enrichment')]]
          .to_string(index=False, float_format='{:.3f}'.format))

# %%
# ---------------------------------------------------------------------------
# Top-k hit rate within proteins: each existing predictor vs our model (test set and external sets)
#   Same setting and measure as the previous cell: within each protein with >= 1 positive site, the
#   S/T are ranked by each model's score; hit at k = >= 1 positive among the top k. Ties are broken
#   at random: the expected hit rate over random tie-breaking is computed exactly (some tools give
#   many equal scores). Random choice: k S/T drawn without replacement (hypergeometric).
#   Pairwise, as for AUROC: for each predictor, only proteins with < 30% identity to the training
#   data of both models are kept, and both models are ranked on the same sites.
#     Test set: our test split is already < 30% identical to our training and validation data; each
#       predictor's test scores and homolog flags come from its benchmark notebook (NB06-NB11;
#       LM-OGlcNAc-Site: repository-snapshot filter; O-GlcNAcPRED-DL: server-data filter of the model
#       applied to each protein; BSB 2025: proteins of <= 1,022 residues only).
#     External sets: scores (`model_cols`) of the comparison cell; homolog flags from
#       tool_overlap.parquet, as in the pairwise AUROC cell (homolog_columns).
#   Unlabelled S/T are not proven negatives, so hit rates are lower bounds.
#   Run after the pairwise comparison cell and the top-k cell of our model (K_VALUES, K_MAIN).
#   Output: topk_hit_rate_tools.csv.
# ---------------------------------------------------------------------------
def p_none(n, m, r):
    """Probability that r S/T drawn without replacement from n (m of them positive) contain no positive."""
    p = 1.0
    for i in range(r):
        p *= (n - m - i) / (n - i)
        if p <= 0:
            return 0.0
    return p

def hit_probability(score, y, k):
    """Expected P(>= 1 positive among the top k), with ties at the k-th score broken at random."""
    k = min(k, len(score))
    threshold = np.sort(score)[::-1][k - 1]
    above, tied = score > threshold, score == threshold
    if y[above].any():
        return 1.0
    return 1.0 - p_none(int(tied.sum()), int(y[tied].sum()), k - int(above.sum()))

def topk_pair(d, tool_col, ours_col):
    """Mean hit rates over the proteins of d (columns seq_id, y and the two score columns)."""
    hits = {'tool': [], 'ours': [], 'random': []}
    for _, g in d.groupby('seq_id', sort=False):
        y = g['y'].to_numpy()
        n, m = len(y), int(y.sum())
        if m == 0:
            continue
        hits['random'].append([1.0 - p_none(n, m, min(k, n)) for k in K_VALUES])
        for who, col in [('tool', tool_col), ('ours', ours_col)]:
            s = g[col].to_numpy()
            hits[who].append([hit_probability(s, y, k) for k in K_VALUES])
    row = {'proteins': len(hits['random']), 'positives': int(d.loc[d['seq_id'].isin(d.loc[d['y'] == 1, 'seq_id']), 'y'].sum())}
    for who in ('tool', 'ours', 'random'):
        mean = np.mean(hits[who], axis=0) if hits[who] else [np.nan] * len(K_VALUES)
        row.update({f'{who}@{k}': v for k, v in zip(K_VALUES, mean)})
    return row

# Test set: each predictor's saved test scores, our test scores, and its homolog flags
TD = 'data_processed/other_tools'
ours_test = pd.read_parquet('data_processed/final_model/test_scores.parquet')[['protein_id', 'position', 'probability']]
def test_set(path, score_col, flags_path, flag_col='homolog_30'):
    t = pd.read_parquet(path)[['protein_id', 'position', 'label', score_col]]
    t = t.merge(ours_test, on=['protein_id', 'position'], how='left', validate='one_to_one')
    assert t['probability'].notna().all()
    flags = pd.read_csv(flags_path).set_index('protein_id')[flag_col].astype(bool)
    t = t[~t['protein_id'].map(flags).to_numpy(bool)]
    return t.rename(columns={'protein_id': 'seq_id', 'label': 'y', score_col: 'tool'})

pdl = pd.read_parquet(f'{TD}/O-GlcNAcPRED-DL/test_scores.parquet')
pdl_flags = pd.read_csv(f'{TD}/O-GlcNAcPRED-DL/training_data_server/test_protein_overlap_server.csv').set_index('protein_id')
pdl_species = np.where(pdl['matched_model'] == 'mouse', 'mouse', 'human')
pdl_homolog = np.array([bool(pdl_flags.at[p, f'homolog_30_{s}']) for p, s in zip(pdl['protein_id'], pdl_species)])
pdl_test = (pdl.loc[~pdl_homolog, ['protein_id', 'position', 'label', 'score_matched']]
            .merge(ours_test, on=['protein_id', 'position'], how='left', validate='one_to_one')
            .rename(columns={'protein_id': 'seq_id', 'label': 'y', 'score_matched': 'tool'}))
assert pdl_test['probability'].notna().all()

TEST_SETS = {
    'LM-OGlcNAc-Site': test_set(f'{TD}/LM-OGlcNAc-Site/test_scores.parquet', 'score_layer33',
                                f'{TD}/LM-OGlcNAc-Site/training_data_git/test_protein_overlap_snapshot.csv'),
    'O-GlcNAcPRED-DL': pdl_test,
    'Stack-OglyPred-PLM': test_set(f'{TD}/Stack-OglyPred-PLM/test_scores.parquet', 'score',
                                   f'{TD}/Stack-OglyPred-PLM/test_protein_overlap.csv'),
    'RNN-[600,75] (Seber)': test_set(f'{TD}/Seber/test_scores.parquet', 'score_rnn600_75', f'{TD}/Seber/test_protein_overlap.csv'),
    'RNN-225 (Seber and Braatz)': test_set(f'{TD}/Seber/test_scores.parquet', 'score_rnn225', f'{TD}/Seber/test_protein_overlap.csv'),
    'BSB 2025 (Arcanjo et al.)': test_set(f'{TD}/BSB2025/test_scores.parquet', 'score', f'{TD}/BSB2025/test_protein_overlap.csv'),
}
rows = []
for tool, t in TEST_SETS.items():
    t = t.assign(y=t['y'].astype(int))
    rows.append({'set': 'Test set', 'tool': tool, **topk_pair(t, 'tool', 'probability')})

# External sets: `scored` and `model_cols` of the comparison cell; homolog flags from tool_overlap.parquet
tool_flags = pd.read_parquet(f'{OUT_DIR}/tool_overlap.parquet').set_index('seq_id')
ours_homolog = scored['seq_id'].map(tool_flags['homolog_ours']).to_numpy(bool)
ext = scored[['unit', 'seq_id', 'y']].copy()
ext['ours'] = model_cols['Our model'][0]
for tool in TOOLS:
    ext['tool'] = model_cols[tool][0]
    for unit in UNITS:
        homolog_tool = scored['seq_id'].map(tool_flags[homolog_columns(tool, unit)].any(axis=1)).to_numpy(bool)
        keep = (scored['unit'] == unit).to_numpy() & ~ours_homolog & ~homolog_tool
        rows.append({'set': unit, 'tool': tool, **topk_pair(ext[keep], 'tool', 'ours')})
topk_tools = pd.DataFrame(rows)
topk_tools.to_csv(f'{OUT_DIR}/topk_hit_rate_tools.csv', index=False)

cols = ['tool', 'proteins'] + [f'{w}@{k}' for k in (1, 3, 5, K_MAIN) for w in ('tool', 'ours', 'random')]
for s in ('Test set', 'All studies pooled'):
    print(f'\n{s}: proteins with < 30% identity to both training sets (pairwise)')
    with pd.option_context('display.width', 250):
        print(topk_tools.loc[topk_tools['set'] == s, cols].to_string(index=False, float_format='{:.3f}'.format))
for k in K_VALUES:
    d = topk_tools[topk_tools['proteins'] >= 20]
    print(f'top-{k}: ours higher in {(d[f"ours@{k}"] > d[f"tool@{k}"]).sum()} of {len(d)} comparisons with >= 20 proteins')

# %%
import io
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------
# Figures: top-k hit rate within proteins, each existing predictor vs our model
#   (1) Hit rate against k (K_VALUES) for the predictor (grey), our model (blue) and random choice
#       (dotted), one panel per predictor; test set and pooled external set, proteins with < 30%
#       identity to the training data of both models (topk_tools of the previous cell). The dashed
#       vertical line marks K_MAIN.
#   (2) Both sets in one figure at K_MAIN: per predictor, its hit rate (grey), that of our model on the
#       same proteins (blue) and random choice (tick). In brackets: proteins compared.
#   (3) Every external unit at K_MAIN, in two versions: every protein of each external set, and
#       proteins with < 30% identity to the training data of both models. Units with fewer than
#       MIN_PROTEINS proteins are not drawn; numbers on the right: proteins compared.
#       Uses topk_pair() and topk_tools of the previous cell.
#   Frontiers: 180 mm wide, text >= 8 pt, no red or green; PDF and RGB TIFF at 300 dpi.
#   K_VALUES and K_MAIN: top-k cell of our model.
#   Outputs: topk_hit_rate_ours_vs_tools_external.csv and five figures in FIG_DIR.
# ---------------------------------------------------------------------------
MIN_PROTEINS = 20
SETS = {'Test set': 'Test set', 'All studies pooled': 'External studies, pooled'}
RANDOM_COLOUR = '#4a4945'
LINK_COLOUR = '#c9c8c3'
GRID_COLOUR = '#e6e5e0'
assert K_MAIN in K_VALUES
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False})

def save_figure(fig, name):
    fig.savefig(f'{FIG_DIR}/{name}.pdf')
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=300)
    Image.open(buffer).convert('RGB').save(f'{FIG_DIR}/{name}.tif', dpi=(300, 300))

def axis_max(d):
    """Upper x limit: next multiple of 0.1 above the highest drawn hit rate at K_MAIN (at most 1)."""
    d = d[d['proteins'] >= MIN_PROTEINS]
    return min(1.0, np.floor(d[[f'tool@{K_MAIN}', f'ours@{K_MAIN}']].max().max() * 10) / 10 + 0.1)

def dumbbell(ax, d, ypos):
    """Predictor, our model and random choice at K_MAIN, one row per entry of d."""
    for yp, (_, r) in zip(ypos, d.iterrows()):
        if r['proteins'] < MIN_PROTEINS:
            ax.text(0.01, yp, f'< {MIN_PROTEINS} proteins', va='center', fontsize=8, color=TOOL_COLOUR)
            continue
        ax.plot([r[f'tool@{K_MAIN}'], r[f'ours@{K_MAIN}']], [yp, yp], color=LINK_COLOUR, linewidth=1.5, zorder=1)
        ax.scatter(r[f'random@{K_MAIN}'], yp, marker='|', s=60, color=RANDOM_COLOUR, zorder=2)
        ax.scatter(r[f'tool@{K_MAIN}'], yp, color=TOOL_COLOUR, s=22, zorder=3)
        ax.scatter(r[f'ours@{K_MAIN}'], yp, color=OUR_COLOUR, s=22, zorder=3)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color=GRID_COLOUR, linewidth=0.6)
    ax.set_axisbelow(True)

DUMBBELL_HANDLES = [plt.Line2D([], [], marker='o', linestyle='', color=TOOL_COLOUR, label='Predictor'),
                    plt.Line2D([], [], marker='o', linestyle='', color=OUR_COLOUR, label='Our model'),
                    plt.Line2D([], [], marker='|', linestyle='', markersize=9, color=RANDOM_COLOUR, label='Random choice')]

# (1) Hit rate against k, one panel per predictor
k_ticks = [k for k in K_VALUES if k == 1 or k % 5 == 0]
n_cols = 3
n_rows = -(-len(TOOLS) // n_cols)
for set_name, file_name in [('Test set', 'figure_topk_pairwise_test'),
                            ('All studies pooled', 'figure_topk_pairwise_external_pooled')]:
    d = topk_tools[topk_tools['set'] == set_name].set_index('tool').loc[TOOLS]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(180 / 25.4, 55 * n_rows / 25.4), sharex=True, sharey=True,
                             layout='constrained', squeeze=False)
    for ax, (tool, r) in zip(axes.flat, d.iterrows()):
        ax.axvline(K_MAIN, color=LINK_COLOUR, linewidth=1, linestyle='--', zorder=0)
        for who, colour, style, label in [('tool', TOOL_COLOUR, '-', 'Predictor'), ('ours', OUR_COLOUR, '-', 'Our model'),
                                          ('random', RANDOM_COLOUR, ':', 'Random choice')]:
            ax.plot(list(K_VALUES), [r[f'{who}@{k}'] for k in K_VALUES], linestyle=style, color=colour, linewidth=1.5,
                    marker='o' if who != 'random' else None, markersize=2.5, label=label)
        ax.set_title(f'{tool}\n{int(r["proteins"]):,} proteins', loc='left')
        ax.set_xticks(k_ticks)
        ax.set_ylim(0, 1)
    for ax in list(axes.flat)[len(TOOLS):]:
        ax.axis('off')
    for k, ax in enumerate(list(axes.flat)[:len(TOOLS)]):
        if k % n_cols == 0:
            ax.set_ylabel('Proteins with a site in the top k')
        if k + n_cols >= len(TOOLS):   # lowest panel of its column: x-axis numbers and label
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.set_xlabel('k (top-ranked S/T per protein)')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncols=3, frameon=False,
               title=f"{SETS[set_name]}: proteins with < 30% identity to both models' training data; dashed line: k = {K_MAIN}")
    save_figure(fig, file_name)
    plt.show()

# (2) Test set and pooled external set at K_MAIN
ypos = np.arange(len(TOOLS))[::-1]
shown = topk_tools[topk_tools['set'].isin(SETS)]
fig, axes = plt.subplots(len(SETS), 1, figsize=(180 / 25.4, 125 / 25.4), sharex=True, layout='constrained')
for ax, (set_name, set_label) in zip(axes, SETS.items()):
    d = shown[shown['set'] == set_name].set_index('tool').loc[TOOLS]
    dumbbell(ax, d, ypos)
    ax.set_yticks(ypos, [f'{t} ({int(n):,})' for t, n in zip(TOOLS, d['proteins'])])
    ax.set_title(f'Top {K_MAIN}', loc='left')
    ax.set_ylabel(set_label)
    ax.set_xlim(0, axis_max(shown))
    print(f'{set_label}, top {K_MAIN}: ours higher than {(d[f"ours@{K_MAIN}"] > d[f"tool@{K_MAIN}"]).sum()} '
          f'of {len(d)} predictors')
axes[-1].set_xlabel(f'Proteins with a site among the top {K_MAIN} S/T')
fig.legend(handles=DUMBBELL_HANDLES, loc='outside lower center', ncols=3, frameon=False,
           title="Proteins with < 30% identity to both models' training data; in brackets: proteins compared")
save_figure(fig, f'figure_topk{K_MAIN}_ours_vs_tools_test_external_pooled')
plt.show()

# (3) Every external unit at K_MAIN: all proteins, and proteins with < 30% identity to the training data
#     of both models (the latter from the previous cell)
ext_all = scored[['unit', 'seq_id', 'y']].copy()
ext_all['ours'] = model_cols['Our model'][0]
rows = []
for tool in TOOLS:
    ext_all['tool'] = model_cols[tool][0]
    for unit in UNITS:
        in_unit = (scored['unit'] == unit).to_numpy()
        rows.append({'subset': 'all proteins', 'tool': tool, 'unit': unit, **topk_pair(ext_all[in_unit], 'tool', 'ours')})
ours_vs = pd.concat([pd.DataFrame(rows),
                     topk_tools[topk_tools['set'].isin(UNITS)].rename(columns={'set': 'unit'})
                     .assign(subset='homologs (>= 30% identity) removed')], ignore_index=True)
ours_vs.to_csv(f'{OUT_DIR}/topk_hit_rate_ours_vs_tools_external.csv', index=False)

SUBSETS = {'all proteins': ('every protein of each external set', f'figure_topk{K_MAIN}_ours_vs_tools_external_all'),
           'homologs (>= 30% identity) removed': ("proteins with < 30% identity to both models' training data",
                                                  f'figure_topk{K_MAIN}_ours_vs_tools_external_homologs_removed')}
ypos = np.arange(len(UNITS))[::-1]
for subset, (description, file_name) in SUBSETS.items():
    d_sub = ours_vs[ours_vs['subset'] == subset]
    print(f'\nTop-{K_MAIN} hit rate, {description} (tool / ours / random):')
    for tool in TOOLS:
        d = d_sub[d_sub['tool'] == tool].set_index('unit').loc[UNITS]
        print(f'  {tool}: ' + '; '.join(f'{u} {r[f"tool@{K_MAIN}"]:.3f} / {r[f"ours@{K_MAIN}"]:.3f} / {r[f"random@{K_MAIN}"]:.3f} '
                                        f'({int(r["proteins"])} proteins)' for u, r in d.iterrows()))
    drawn = d_sub[d_sub['proteins'] >= MIN_PROTEINS]
    print(f'Ours higher in {(drawn[f"ours@{K_MAIN}"] > drawn[f"tool@{K_MAIN}"]).sum()} of {len(drawn)} comparisons '
          f'with >= {MIN_PROTEINS} proteins')

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(180 / 25.4, 52 * n_rows / 25.4), sharex=True, sharey=True,
                             layout='constrained', squeeze=False)
    for ax, tool in zip(axes.flat, TOOLS):
        d = d_sub[d_sub['tool'] == tool].set_index('unit').loc[UNITS]
        dumbbell(ax, d, ypos)
        for yp, (_, r) in zip(ypos, d.iterrows()):
            if r['proteins'] >= MIN_PROTEINS:
                ax.annotate(f'{int(r["proteins"]):,}', (1, yp), xycoords=('axes fraction', 'data'), xytext=(3, 0),
                            textcoords='offset points', va='center', fontsize=8, color=TOOL_COLOUR)
        ax.set_title(tool, loc='left')
        ax.set_xlim(0, axis_max(d_sub))
        ax.set_yticks(ypos, UNITS)
    for ax in list(axes.flat)[len(TOOLS):]:
        ax.axis('off')
    for k, ax in enumerate(list(axes.flat)[:len(TOOLS)]):
        if k + n_cols >= len(TOOLS):   # lowest panel of its column: x-axis numbers and label
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.set_xlabel(f'Proteins with a site in the top {K_MAIN}')
    fig.legend(handles=DUMBBELL_HANDLES, loc='outside lower center', ncols=3, frameon=False,
               title=f'External sets, {description}; numbers: proteins compared')
    save_figure(fig, file_name)
    plt.show()

# %%
import io
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------
# Figures: top-k hit rate within proteins, each existing predictor vs our model, every external protein
#   Same as the previous figure cell, without the homology filter, on the external sets only (every
#   protein of each external set; `scored`, `model_cols` of the comparison cell). The test set is not
#   used: it is homology-reduced for our model only. Proteins similar to a model's training data are
#   included, so these figures show what a user would get, not a fair comparison.
#   (1) Hit rate against k (K_VALUES) for the predictor (grey), our model (blue) and random choice
#       (dotted), one panel per predictor; pooled external set. Dashed line: K_MAIN.
#   (2) Pooled external set at K_MAIN: per predictor, its hit rate (grey), that of our model (blue) and
#       random choice (tick).
#   Every external unit at K_MAIN: figure_topk{K_MAIN}_ours_vs_tools_external_all of the previous cell.
#   Uses K_VALUES, K_MAIN (top-k cell of our model) and topk_pair (top-k tools cell).
#   Frontiers: 180 mm wide, text >= 8 pt, no red or green; PDF and RGB TIFF at 300 dpi.
#   Outputs: topk_hit_rate_tools_all_proteins.csv and two figures in FIG_DIR.
# ---------------------------------------------------------------------------
POOLED = 'All studies pooled'
RANDOM_COLOUR = '#4a4945'
LINK_COLOUR = '#c9c8c3'
GRID_COLOUR = '#e6e5e0'
assert K_MAIN in K_VALUES
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False})

def save_figure(fig, name):
    fig.savefig(f'{FIG_DIR}/{name}.pdf')
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=300)
    Image.open(buffer).convert('RGB').save(f'{FIG_DIR}/{name}.tif', dpi=(300, 300))

# Every protein of each external unit
ext_every = scored[['unit', 'seq_id', 'y']].copy()
ext_every['ours'] = model_cols['Our model'][0]
rows = []
for tool in TOOLS:
    ext_every['tool'] = model_cols[tool][0]
    for unit in UNITS:
        rows.append({'set': unit, 'tool': tool, **topk_pair(ext_every[(scored['unit'] == unit).to_numpy()], 'tool', 'ours')})
topk_tools_all = pd.DataFrame(rows)
topk_tools_all.to_csv(f'{OUT_DIR}/topk_hit_rate_tools_all_proteins.csv', index=False)

cols = ['tool', 'proteins'] + [f'{w}@{k}' for k in (1, 3, 5, K_MAIN) for w in ('tool', 'ours', 'random')]
print(f'\n{POOLED}: every protein (no homology filter)')
with pd.option_context('display.width', 250):
    print(topk_tools_all.loc[topk_tools_all['set'] == POOLED, cols].to_string(index=False, float_format='{:.3f}'.format))
for k in (1, 3, 5, K_MAIN):
    d = topk_tools_all[topk_tools_all['proteins'] >= 20]
    print(f'top-{k}: ours higher in {(d[f"ours@{k}"] > d[f"tool@{k}"]).sum()} of {len(d)} comparisons '
          f'(external units, >= 20 proteins)')

d = topk_tools_all[topk_tools_all['set'] == POOLED].set_index('tool').loc[TOOLS]

# (1) Hit rate against k, one panel per predictor
k_ticks = [k for k in K_VALUES if k == 1 or k % 5 == 0]
n_cols = 3
n_rows = -(-len(TOOLS) // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(180 / 25.4, 55 * n_rows / 25.4), sharex=True, sharey=True,
                         layout='constrained', squeeze=False)
for ax, (tool, r) in zip(axes.flat, d.iterrows()):
    ax.axvline(K_MAIN, color=LINK_COLOUR, linewidth=1, linestyle='--', zorder=0)
    for who, colour, style, label in [('tool', TOOL_COLOUR, '-', 'Predictor'), ('ours', OUR_COLOUR, '-', 'Our model'),
                                      ('random', RANDOM_COLOUR, ':', 'Random choice')]:
        ax.plot(list(K_VALUES), [r[f'{who}@{k}'] for k in K_VALUES], linestyle=style, color=colour, linewidth=1.5,
                marker='o' if who != 'random' else None, markersize=2.5, label=label)
    ax.set_title(f'{tool}\n{int(r["proteins"]):,} proteins', loc='left')
    ax.set_xticks(k_ticks)
    ax.set_ylim(0, 1)
for ax in list(axes.flat)[len(TOOLS):]:
    ax.axis('off')
for k, ax in enumerate(list(axes.flat)[:len(TOOLS)]):
    if k % n_cols == 0:
        ax.set_ylabel('Proteins with a site in the top k')
    if k + n_cols >= len(TOOLS):   # lowest panel of its column: x-axis numbers and label
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_xlabel('k (top-ranked S/T per protein)')
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside lower center', ncols=3, frameon=False,
           title=f'External studies, pooled: every protein (no homology filter); dashed line: k = {K_MAIN}')
save_figure(fig, 'figure_topk_pairwise_external_pooled_all_proteins')
plt.show()

# (2) Pooled external set at K_MAIN
ypos = np.arange(len(TOOLS))[::-1]
x_max = min(1.0, np.ceil(d[[f'tool@{K_MAIN}', f'ours@{K_MAIN}']].max().max() * 10 + 0.25) / 10)
fig, ax = plt.subplots(figsize=(180 / 25.4, 70 / 25.4), layout='constrained')
for yp, (_, r) in zip(ypos, d.iterrows()):
    ax.plot([r[f'tool@{K_MAIN}'], r[f'ours@{K_MAIN}']], [yp, yp], color=LINK_COLOUR, linewidth=1.5, zorder=1)
    ax.scatter(r[f'random@{K_MAIN}'], yp, marker='|', s=60, color=RANDOM_COLOUR, zorder=2)
    ax.scatter(r[f'tool@{K_MAIN}'], yp, color=TOOL_COLOUR, s=22, zorder=3)
    ax.scatter(r[f'ours@{K_MAIN}'], yp, color=OUR_COLOUR, s=22, zorder=3)
ax.set_yticks(ypos, [f'{t} ({int(n):,})' for t, n in zip(TOOLS, d['proteins'])])
ax.tick_params(axis='y', length=0)
ax.grid(axis='x', color=GRID_COLOUR, linewidth=0.6)
ax.set_axisbelow(True)
ax.set_title(f'Top {K_MAIN}', loc='left')
ax.set_ylabel('External studies, pooled')
ax.set_xlim(0, x_max)
ax.set_xlabel(f'Proteins with a site among the top {K_MAIN} S/T')
print(f'External studies, pooled, top {K_MAIN}, every protein: ours higher than '
      f'{(d[f"ours@{K_MAIN}"] > d[f"tool@{K_MAIN}"]).sum()} of {len(d)} predictors')
handles = [plt.Line2D([], [], marker='o', linestyle='', color=TOOL_COLOUR, label='Predictor'),
           plt.Line2D([], [], marker='o', linestyle='', color=OUR_COLOUR, label='Our model'),
           plt.Line2D([], [], marker='|', linestyle='', markersize=9, color=RANDOM_COLOUR, label='Random choice')]
fig.legend(handles=handles, loc='outside lower center', ncols=3, frameon=False,
           title='Every protein (no homology filter); in brackets: proteins compared')
save_figure(fig, f'figure_topk{K_MAIN}_ours_vs_tools_external_pooled_all_proteins')
plt.show()

# %%
import io
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------
# Top-k hit rate within proteins: all models on the same proteins (our model and the six predictors)
#   Ranking that includes our model: every model is scored on the same proteins and the same S/T.
#     Test set: proteins kept by the homolog filter of every predictor (TEST_SETS of the top-k tools
#       cell; our test split is already < 30% identical to our training data); S/T scored by every
#       predictor (inner join on protein and position).
#     External sets: per unit, proteins with < 30% identity to the training data of every model (flags
#       of tool_overlap.parquet; O-GlcNAcPRED-DL: the model applied to the unit).
#   Hit rate and random choice as in the top-k tools cell (hit_probability, p_none; ties broken at random).
#   Sets with fewer than MIN_PROTEINS proteins with a site are listed but not drawn.
#   Uses K_VALUES and K_MAIN (top-k cell of our model) and TEST_SETS, hit_probability, p_none,
#   tool_flags, ours_homolog (top-k tools cell). Unlabelled S/T are not proven negatives.
#   Frontiers: 180 mm wide, text >= 8 pt, no red or green; PDF and RGB TIFF at 300 dpi.
#   Outputs: topk_hit_rate_all_models.csv, figure_topk{K_MAIN}_all_models_same_proteins in FIG_DIR.
# ---------------------------------------------------------------------------
MIN_PROTEINS = 20
RANDOM_COLOUR = '#4a4945'
GRID_COLOUR = '#e6e5e0'
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False})

def save_figure(fig, name):
    fig.savefig(f'{FIG_DIR}/{name}.pdf')
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=300)
    Image.open(buffer).convert('RGB').save(f'{FIG_DIR}/{name}.tif', dpi=(300, 300))

def all_model_rows(d, set_name):
    """Mean hit rate at every k for each model and for random choice, over the proteins of d with >= 1 positive."""
    rates = {m: [] for m in MODELS + ['Random choice']}
    for _, g in d.groupby('seq_id', sort=False):
        y = g['y'].to_numpy()
        n, m = len(y), int(y.sum())
        if m == 0:
            continue
        rates['Random choice'].append([1.0 - p_none(n, m, min(k, n)) for k in K_VALUES])
        for model in MODELS:
            rates[model].append([hit_probability(g[model].to_numpy(), y, k) for k in K_VALUES])
    rows = []
    for model, r in rates.items():
        mean = np.mean(r, axis=0) if r else [np.nan] * len(K_VALUES)
        rows.append({'set': set_name, 'model': model, 'proteins': len(r), **{f'hit@{k}': v for k, v in zip(K_VALUES, mean)}})
    return rows

# Test set: S/T of the proteins kept by every predictor, with every model's score
test_all = None
for tool, t in TEST_SETS.items():
    t = t[['seq_id', 'position', 'y', 'tool', 'probability']].rename(columns={'tool': tool, 'y': f'y {tool}'})
    if test_all is None:
        test_all = t.rename(columns={'probability': 'Our model'})
    else:
        test_all = test_all.merge(t.drop(columns='probability'), on=['seq_id', 'position'], how='inner', validate='one_to_one')
label_cols = [f'y {tool}' for tool in TEST_SETS]
assert (test_all[label_cols].astype(int).nunique(axis=1) == 1).all()
test_all['y'] = test_all[label_cols[0]].astype(int)
rows = all_model_rows(test_all, 'Test set')

# External sets: proteins with < 30% identity to the training data of every model
allm = scored[['unit', 'seq_id', 'y']].copy()
for model in MODELS:
    allm[model] = model_cols[model][0]
for unit in UNITS:
    homolog_any = ours_homolog.copy()
    for tool in TOOLS:
        homolog_any |= scored['seq_id'].map(tool_flags[homolog_columns(tool, unit)].any(axis=1)).to_numpy(bool)
    rows += all_model_rows(allm[(allm['unit'] == unit).to_numpy() & ~homolog_any], unit)
topk_all = pd.DataFrame(rows)
topk_all.to_csv(f'{OUT_DIR}/topk_hit_rate_all_models.csv', index=False)

SET_ORDER = ['Test set'] + UNITS
for k in (1, 3, 5, K_MAIN):
    print(f"\nHit rate at top {k}, all models on the same proteins:")
    with pd.option_context('display.width', 250):
        print(topk_all.pivot(index='model', columns='set', values=f'hit@{k}').loc[MODELS + ['Random choice'], SET_ORDER]
              .to_string(float_format='{:.3f}'.format))
n_of = topk_all[topk_all['model'] == 'Our model'].set_index('set')['proteins']
print('\nProteins with a site per set: ' + ', '.join(f'{s} {int(n_of[s])}' for s in SET_ORDER))
print(f'\nRanking at top {K_MAIN} (sets with >= {MIN_PROTEINS} proteins with a site):')
for s in SET_ORDER:
    if n_of[s] >= MIN_PROTEINS:
        v = topk_all[topk_all['set'] == s].set_index('model')[f'hit@{K_MAIN}'].sort_values(ascending=False)
        print(f'  {s} ({int(n_of[s])} proteins): ' + ', '.join(f'{m} {x:.3f}' for m, x in v.items()))

# Figure: one panel per set; hit rate at K_MAIN per model, our model in blue; dotted line: random choice
n_cols = 4
n_rows = -(-len(SET_ORDER) // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(180 / 25.4, 62 * n_rows / 25.4), sharex=True, sharey=True,
                         layout='constrained', squeeze=False)
ypos = np.arange(len(MODELS))[::-1]
values = topk_all.set_index(['set', 'model'])
drawn = [s for s in SET_ORDER if n_of[s] >= MIN_PROTEINS]
hi = min(1.0, values.loc[drawn, f'hit@{K_MAIN}'].max() + 0.15)
colours = [OUR_COLOUR if m == 'Our model' else TOOL_COLOUR for m in MODELS]
for ax, s in zip(axes.flat, SET_ORDER):
    v = values.loc[s]
    ax.set_yticks(ypos, MODELS)
    ax.tick_params(axis='y', length=0)
    ax.set_title(f'{s}\n{int(n_of[s]):,} proteins with a site', loc='left')
    if n_of[s] < MIN_PROTEINS:
        ax.text(0.02, 0.5, f'< {MIN_PROTEINS} proteins:\nnot drawn', transform=ax.transAxes, color=TOOL_COLOUR)
        continue
    ax.hlines(ypos, 0, v.loc[MODELS, f'hit@{K_MAIN}'], color=colours, linewidth=1.5)
    ax.scatter(v.loc[MODELS, f'hit@{K_MAIN}'], ypos, color=colours, s=22, zorder=3)
    for yp, val in zip(ypos, v.loc[MODELS, f'hit@{K_MAIN}']):
        ax.annotate(f'{val:.2f}', (val, yp), xytext=(4, 0), textcoords='offset points', va='center', fontsize=8)
    ax.axvline(v.loc['Random choice', f'hit@{K_MAIN}'], color=RANDOM_COLOUR, linewidth=1, linestyle=':')
    ax.grid(axis='x', color=GRID_COLOUR, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(0, hi)
for ax in list(axes.flat)[len(SET_ORDER):]:
    ax.axis('off')
for i, ax in enumerate(list(axes.flat)[:len(SET_ORDER)]):
    if i + n_cols >= len(SET_ORDER):   # lowest panel of its column: x-axis numbers
        ax.xaxis.set_tick_params(labelbottom=True)
fig.suptitle("All models on the same proteins: proteins with < 30% identity to the training data of every model",
             fontsize=8, x=0.01, ha='left')
fig.supxlabel(f'Proteins with a site among the top {K_MAIN} S/T (dotted line: random choice)', fontsize=8)
save_figure(fig, f'figure_topk{K_MAIN}_all_models_same_proteins')
plt.show()

# %%
# ---------------------------------------------------------------------------
# LM-OGlcNAc-Site vs our model at every k, pooled external set
#   All proteins: every protein of the external sets (topk_hit_rate_ours_vs_tools_external.csv).
#   Homologs removed, pairwise: proteins with < 30% identity to both models' training data
#     (topk_hit_rate_tools.csv).
#   Homologs removed, all models: proteins with < 30% identity to the training data of every model
#     (topk_hit_rate_all_models.csv).
# ---------------------------------------------------------------------------
TOOL = 'LM-OGlcNAc-Site'
POOLED = 'All studies pooled'

every = pd.read_csv(f'{OUT_DIR}/topk_hit_rate_ours_vs_tools_external.csv')
every = every[(every['subset'] == 'all proteins') & (every['unit'] == POOLED) & (every['tool'] == TOOL)].iloc[0]
pairwise = pd.read_csv(f'{OUT_DIR}/topk_hit_rate_tools.csv')
pairwise = pairwise[(pairwise['set'] == POOLED) & (pairwise['tool'] == TOOL)].iloc[0]
all_models = pd.read_csv(f'{OUT_DIR}/topk_hit_rate_all_models.csv')
all_models = all_models[all_models['set'] == POOLED].set_index('model')

ks = sorted(int(c.split('@')[1]) for c in pairwise.index if c.startswith('ours@'))
rows = []
for k in ks:
    values = {'all proteins': (every[f'tool@{k}'], every[f'ours@{k}']),
              'homologs removed, pairwise': (pairwise[f'tool@{k}'], pairwise[f'ours@{k}']),
              'homologs removed, all models': (all_models.at[TOOL, f'hit@{k}'], all_models.at['Our model', f'hit@{k}'])}
    row = {'k': k}
    for name, (tool, ours) in values.items():
        row.update({f'{name}: LM': tool, f'{name}: ours': ours,
                    f'{name}: winner': 'ours' if ours > tool else ('LM' if tool > ours else 'tie')})
    rows.append(row)
print(f'{TOOL} vs our model, pooled external set; proteins: all {int(every["proteins"])}, '
      f'pairwise {int(pairwise["proteins"])}, all models {int(all_models.at["Our model", "proteins"])}')
with pd.option_context('display.width', 300, 'display.max_columns', None):
    print(pd.DataFrame(rows).to_string(index=False, float_format='{:.3f}'.format))

# %%
# ---------------------------------------------------------------------------
# Top-k check at every k: our model vs each predictor other than LM-OGlcNAc-Site
#   For every comparison with >= MIN_PROTEINS proteins with a site, lists the k at which our model is
#   not higher than the predictor (with predictor / ours hit rates).
#     Homologs removed, pairwise: proteins with < 30% identity to both models' training data
#       (topk_hit_rate_tools.csv; test set and external units).
#     All proteins: every protein of each external unit (topk_hit_rate_ours_vs_tools_external.csv).
#     Homologs removed, all models: proteins with < 30% identity to the training data of every model
#       (topk_hit_rate_all_models.csv; test set and external units).
# ---------------------------------------------------------------------------
EXCLUDED = 'LM-OGlcNAc-Site'
MIN_PROTEINS = 20

pairwise = pd.read_csv(f'{OUT_DIR}/topk_hit_rate_tools.csv')
every = pd.read_csv(f'{OUT_DIR}/topk_hit_rate_ours_vs_tools_external.csv')
every = every[every['subset'] == 'all proteins'].rename(columns={'unit': 'set'})
all_models = pd.read_csv(f'{OUT_DIR}/topk_hit_rate_all_models.csv')
ks = sorted(int(c.split('@')[1]) for c in pairwise.columns if c.startswith('ours@'))

# All models on the same proteins: one row per (set, predictor), with our model's hit rates alongside
rows = []
for s, g in all_models.groupby('set', sort=False):
    g = g.set_index('model')
    for tool in g.index.difference(['Our model', 'Random choice']):
        rows.append({'set': s, 'tool': tool, 'proteins': g.at['Our model', 'proteins'],
                     **{f'tool@{k}': g.at[tool, f'hit@{k}'] for k in ks},
                     **{f'ours@{k}': g.at['Our model', f'hit@{k}'] for k in ks}})
same_proteins = pd.DataFrame(rows)

for name, d in [('Homologs removed, pairwise', pairwise), ('All proteins (external sets)', every),
                ('Homologs removed, all models on the same proteins', same_proteins)]:
    d = d[(d['tool'] != EXCLUDED) & (d['proteins'] >= MIN_PROTEINS)]
    n_lower = 0
    print(f'\n{name} (k = {ks[0]}-{ks[-1]}; sets with >= {MIN_PROTEINS} proteins with a site):')
    for _, r in d.iterrows():
        not_higher = [k for k in ks if not r[f'ours@{k}'] > r[f'tool@{k}']]
        n_lower += len(not_higher)
        if not_higher:
            print(f'  {r["set"]} | {r["tool"]} ({int(r["proteins"])} proteins): ours not higher at k = '
                  + ', '.join(f'{k} ({r[f"tool@{k}"]:.3f} vs {r[f"ours@{k}"]:.3f})' for k in not_higher))
    print(f'  ours higher in {len(d) * len(ks) - n_lower} of {len(d) * len(ks)} (comparison, k) pairs; '
          f'all other comparisons: ours higher at every k')
