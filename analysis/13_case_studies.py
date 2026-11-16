# %%
import os
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))

import json, urllib.parse, urllib.request
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, EsmModel
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Case studies: proteins with O-GlcNAc sites reported in functional studies, scored with the
# frozen final model
#   The scoring code is the one used for the external validation (NB05, last cell), which
#   repeats the feature code of NB02 unchanged; final model, Platt parameters and cut-off are
#   read from final_model.pt. A CPU runtime is enough for single proteins.
# ---------------------------------------------------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR = 'data_processed/final_model'
CASE_DIR = 'data_processed/case_studies'
os.makedirs(CASE_DIR, exist_ok=True)
LOCAL_FILE_DIR = '/content'

MAX_RES, OVERLAP, K = 1022, 256, 20
ESM_PATH, MAX_TOKENS = 'facebook/esm2_t30_150M_UR50D', 65536   # as in NB02 for ESM2-150M
LOCAL_FILE = f'{LOCAL_FILE_DIR}/case_residues.npy'
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

final = torch.load(f'{OUT_DIR}/final_model.pt', map_location='cpu')
assert final['plm'] == 'ESM2-150M' and final['window'] == 'pm20'
clf = nn.Sequential(nn.Linear(len(final['mean']), final['config']['hidden']), nn.ReLU(),
                    nn.Dropout(final['config']['dropout']), nn.Linear(final['config']['hidden'], 1))
clf.load_state_dict(final['state'])
clf = clf.to(DEVICE).eval()
f_mean, f_std = final['mean'].to(DEVICE), final['std'].to(DEVICE)
PLATT_A, PLATT_B, EXT_CUTOFF = final['platt']['a'], final['platt']['b'], final['cutoff']['probability']

def window_starts(length):
    """0-based start of each window: the fewest windows of MAX_RES residues overlapping by
    at least OVERLAP, spaced evenly from the protein's first to its last residue."""
    if length <= MAX_RES:
        return [0]
    n = int(np.ceil((length - OVERLAP) / (MAX_RES - OVERLAP)))
    return [round(i * (length - MAX_RES) / (n - 1)) for i in range(n)]

def embed_windows(model, tokenizer, seqs, out, rows, max_tokens):
    """Write the last-layer embedding of every residue of seqs[k] to out[rows[k]:rows[k] + len(seqs[k])]."""
    order = sorted(range(len(seqs)), key=lambda k: -len(seqs[k]))
    batches, current = [], []
    for k in order:
        # The first window of a batch is its longest, so it sets the padded length
        if current and (len(current) + 1) * (len(seqs[current[0]]) + 2) > max_tokens:
            batches.append(current)
            current = []
        current.append(k)
    batches.append(current)
    for batch in tqdm(batches, desc='batches'):
        toks = tokenizer([seqs[k] for k in batch], return_tensors='pt', padding=True).to(DEVICE)
        with torch.no_grad():
            hidden = model(**toks).last_hidden_state.float().cpu().numpy()
        for j, k in enumerate(batch):
            out[rows[k]:rows[k] + len(seqs[k])] = hidden[j, 1:len(seqs[k]) + 1]

tokenizer = AutoTokenizer.from_pretrained(ESM_PATH)
esm = EsmModel.from_pretrained(ESM_PATH, add_pooling_layer=False).to(DEVICE).eval()

def score_proteins(ids, seqs, keep_features=False):
    """Every S/T of the given proteins: logit, calibrated probability and call (features if asked)."""
    # Windows and the most central window of each S/T (NB02 cell 0)
    windows = pd.DataFrame([{'protein_id': pid, 'window': w, 'first_position': s + 1, 'length': min(len(seq), MAX_RES)}
                            for pid, seq in zip(ids, seqs) for w, s in enumerate(window_starts(len(seq)))])
    windows['row'] = windows['length'].cumsum() - windows['length']
    windows_of = {pid: g.to_dict('records') for pid, g in windows.groupby('protein_id', sort=False)}
    sites = []
    for pid, seq in zip(ids, seqs):
        ws = windows_of[pid]
        for p, aa in enumerate(seq, start=1):
            if aa not in 'ST':
                continue
            inside = [w for w in ws if w['first_position'] <= p < w['first_position'] + w['length']]
            w = max(inside, key=lambda w: min(p - w['first_position'], w['first_position'] + w['length'] - 1 - p))
            sites.append({'protein_id': pid, 'position': p, 'residue': aa, 'window_row': w['row'],
                          'offset': p - w['first_position']})
    sites = pd.DataFrame(sites)

    # ESM2-150M embeddings of every window (NB02 cell 1), on the Colab local disk
    seq_of = dict(zip(ids, seqs))
    window_seqs = [seq_of[p][f - 1:f - 1 + n]
                   for p, f, n in zip(windows['protein_id'], windows['first_position'], windows['length'])]
    M = np.lib.format.open_memmap(LOCAL_FILE, mode='w+', dtype=np.float32,
                                  shape=(int(windows['length'].sum()), esm.config.hidden_size))
    embed_windows(esm, tokenizer, window_seqs, M, windows['row'].to_numpy(), MAX_TOKENS)

    # Features (NB02 cell 4) and the final model, one window at a time
    length_of_window = dict(zip(windows['row'], windows['length']))
    logit = np.empty(len(sites), dtype=np.float32)
    feats = np.empty((len(sites), 2 * esm.config.hidden_size), dtype=np.float32) if keep_features else None
    for row0, idx in tqdm(sites.groupby('window_row').indices.items(), desc='features and scores'):
        block = np.asarray(M[row0:row0 + length_of_window[row0]])
        L = len(block)
        csum = np.zeros((L + 1, block.shape[1]))
        np.cumsum(block, axis=0, dtype=np.float64, out=csum[1:])
        i = sites['offset'].to_numpy()[idx]
        lo, hi = np.maximum(i - K, 0), np.minimum(i + K + 1, L)
        window_mean = ((csum[hi] - csum[lo]) / (hi - lo)[:, None]).astype(np.float32)
        X = np.concatenate([block[i], window_mean], axis=1)
        if keep_features:
            feats[idx] = X
        with torch.no_grad():
            logit[idx] = clf((torch.from_numpy(X).to(DEVICE) - f_mean) / f_std).squeeze(1).cpu().numpy()
    del M
    os.remove(LOCAL_FILE)

    z = PLATT_A * logit.astype(np.float64) + PLATT_B   # float64, as in NB05
    sites['logit'] = logit
    sites['probability'] = 1 / (1 + np.exp(-z))
    sites['called'] = sites['probability'] >= EXT_CUTOFF
    return sites.drop(columns=['window_row', 'offset']), feats

# %%
# ---------------------------------------------------------------------------
# Case study: human LATS1 (Meng et al. 2026, J Biol Chem 302:110990, PMID 41338457)
#   Sites mapped by ETD mass spectrometry in 293T cells (searched against human Swiss-Prot):
#   S479, S482, T484, T485; a quadruple alanine mutant (4A) reduced the O-GlcNAc signal.
#   Steps: sequence from UniProt (reviewed human LATS1 and LATS2 entries, looked up by gene
#   name); check that the reported residues match; check whether LATS1 (or its paralog LATS2)
#   is in our dataset and in which split, and whether these sites are labelled there; score
#   every S/T of LATS1 with the final model.
# ---------------------------------------------------------------------------
REPORTED = {479: 'S', 482: 'S', 484: 'T', 485: 'T'}

def uniprot_reviewed(gene, taxon=9606):
    """Accession and sequence of the reviewed UniProtKB entry for a gene name and organism."""
    query = f'gene_exact:{gene} AND organism_id:{taxon} AND reviewed:true'
    url = 'https://rest.uniprot.org/uniprotkb/search?format=fasta&query=' + urllib.parse.quote(query)
    text = urllib.request.urlopen(url, timeout=60).read().decode()
    entries = [e for e in text.split('>')[1:]]
    assert len(entries) == 1, f'{gene}: {len(entries)} reviewed entries'
    header, *lines = entries[0].strip().split('\n')
    return header.split('|')[1], ''.join(lines), header

acc, seq, header = uniprot_reviewed('LATS1')
print(f'LATS1: {header}\nlength {len(seq)}')
for p, r in REPORTED.items():
    print(f'  position {p}: reported {r}, sequence {seq[p - 1]}')
assert all(seq[p - 1] == r for p, r in REPORTED.items())

# In our dataset?
atlas = pd.read_parquet('data_processed/atlas5_proteins.parquet')
for gene in ['LATS1', 'LATS2']:
    a, s, _ = uniprot_reviewed(gene)
    hit = atlas[(atlas['protein_id'].str.split('-').str[0] == a) | (atlas['sequence'] == s)]
    if hit.empty:
        print(f'{gene} ({a}): not in our dataset')
    for r in hit.itertuples():
        print(f'{gene} ({a}): in our dataset as {r.protein_id}, split {r.split}, '
              f'labelled positives {sorted(r.positive)}, masked {sorted(r.masked)}')
        if gene == 'LATS1':
            print('  reported sites labelled positive there:', [p for p in REPORTED if p in set(r.positive)])

# Scores for every S/T of LATS1
lats1, _ = score_proteins([acc], [seq])
lats1['rank_in_protein'] = lats1['probability'].rank(ascending=False, method='min').astype(int)
lats1['reported'] = lats1['position'].isin(REPORTED)
lats1.to_csv(f'{CASE_DIR}/LATS1_scores.csv', index=False)

print(f'\nS/T in LATS1: {len(lats1)}   called positive at the validation cut-off ({EXT_CUTOFF:.4f}): '
      f'{lats1["called"].sum()}')
print('\nReported sites and their neighbourhood (S/T from 470 to 495):')
region = lats1[lats1['position'].between(470, 495)]
print(region[['position', 'residue', 'probability', 'called', 'rank_in_protein', 'reported']]
      .to_string(index=False, float_format='{:.4f}'.format))
print(f'\nPeptide 470-495: {seq[469:495]}')
print('\nTop 10 S/T of LATS1 by predicted probability:')
print(lats1.nsmallest(10, 'rank_in_protein')[['position', 'residue', 'probability', 'called', 'reported']]
      .to_string(index=False, float_format='{:.4f}'.format))

# %%
# ---------------------------------------------------------------------------
# Case-study candidates from the external validation: proteins with many sites
#   Proteins with at least MIN_SITES sites localized by the study, ranked by number of sites,
#   with the identity to the closest training/validation protein (MMseqs2, NB12; identical
#   residues / shorter length) and whether it is the same protein. Model scores are not shown,
#   so the choice does not depend on how the model performed.
# ---------------------------------------------------------------------------
MIN_SITES = 8
pr = pd.read_csv('data_processed/external_validation/protein_results.csv')
candidates = pr[pr['positive'] >= MIN_SITES].copy()
candidates['identity to closest train/val (%)'] = 100 * candidates['max_identity']
candidates = (candidates[['study', 'accessions', 'length', 'positive', 'st_scored',
                          'identity to closest train/val (%)', 'closest_seen_protein', 'same_protein']]
              .sort_values('positive', ascending=False))
print(f'Proteins with >= {MIN_SITES} localized sites: {len(candidates)}')
print(candidates.to_string(index=False, float_format='{:.0f}'.format))

# %%
# ---------------------------------------------------------------------------
# Case-study candidates from the external validation: proteins with many sites
#   Proteins with at least MIN_SITES sites localized by the study, ranked by site density
#   (sites per 100 residues), with the identity to the closest training/validation protein
#   (MMseqs2, NB12; identical residues / shorter length) and whether it is the same protein.
#   Model scores are not shown, so the choice does not depend on how the model performed.
# ---------------------------------------------------------------------------
MIN_SITES = 8
pr = pd.read_csv('data_processed/external_validation/protein_results.csv')
candidates = pr[pr['positive'] >= MIN_SITES].copy()
candidates['sites per 100 aa'] = 100 * candidates['positive'] / candidates['length']
candidates['identity to closest train/val (%)'] = 100 * candidates['max_identity']
candidates = (candidates[['study', 'accessions', 'length', 'positive', 'sites per 100 aa', 'st_scored',
                          'identity to closest train/val (%)', 'closest_seen_protein', 'same_protein']]
              .sort_values('sites per 100 aa', ascending=False))
print(f'Proteins with >= {MIN_SITES} localized sites: {len(candidates)}')
print(candidates.to_string(index=False, float_format='{:.1f}'.format))

# %%
# ---------------------------------------------------------------------------
# Case-study candidates from the external validation: unseen proteins with many sites
#   Proteins with at least MIN_SITES sites localized by the study and identity < 30% to every
#   training/validation protein (MMseqs2, NB12; identical residues / shorter length; the same
#   criterion as our split), ranked by site density (sites per 100 residues). Also shown: the
#   identity to the closest training/validation protein and how many of the experimental
#   sites the final model calls positive (validation cut-off).
# ---------------------------------------------------------------------------
MIN_SITES, MAX_IDENTITY = 5, 0.30
pr = pd.read_csv('data_processed/external_validation/protein_results.csv')
candidates = pr[(pr['positive'] >= MIN_SITES) & (pr['max_identity'] < MAX_IDENTITY)].copy()
candidates['sites per 100 aa'] = 100 * candidates['positive'] / candidates['length']
candidates['identity to closest train/val (%)'] = 100 * candidates['max_identity']
candidates['found'] = candidates['true_positive']
candidates['found (%)'] = 100 * candidates['true_positive'] / candidates['positive']
candidates = (candidates[['study', 'accessions', 'length', 'positive', 'found', 'found (%)', 'sites per 100 aa',
                          'st_scored', 'identity to closest train/val (%)', 'closest_seen_protein']]
              .sort_values('sites per 100 aa', ascending=False))
print(f'Unseen proteins (identity < {100 * MAX_IDENTITY:.0f}%) with >= {MIN_SITES} localized sites: {len(candidates)}')
print(candidates.to_string(index=False, float_format='{:.1f}'.format))

# %%
# ---------------------------------------------------------------------------
# Case-study candidates from the external validation: unseen human proteins with many sites
#   Human studies only (HCC liver 2026, NOTISE 2026, Fu 2026). Proteins with at least MIN_SITES
#   sites localized by the study and identity < 30% to every training/validation protein
#   (MMseqs2, NB12; identical residues / shorter length; the same criterion as our split),
#   ranked by site density (sites per 100 residues). Also shown: the identity to the closest
#   training/validation protein and how many of the experimental sites the final model calls
#   positive (validation cut-off).
# ---------------------------------------------------------------------------
HUMAN_STUDIES = ['HCC liver 2026', 'NOTISE 2026', 'Fu 2026']
MIN_SITES, MAX_IDENTITY = 5, 0.30
pr = pd.read_csv('data_processed/external_validation/protein_results.csv')
candidates = pr[pr['study'].isin(HUMAN_STUDIES) & (pr['positive'] >= MIN_SITES)
                & (pr['max_identity'] < MAX_IDENTITY)].copy()
candidates['sites per 100 aa'] = 100 * candidates['positive'] / candidates['length']
candidates['identity to closest train/val (%)'] = 100 * candidates['max_identity']
candidates['found'] = candidates['true_positive']
candidates['found (%)'] = 100 * candidates['true_positive'] / candidates['positive']
candidates = (candidates[['study', 'accessions', 'length', 'positive', 'found', 'found (%)', 'sites per 100 aa',
                          'st_scored', 'identity to closest train/val (%)', 'closest_seen_protein']]
              .sort_values('sites per 100 aa', ascending=False))
print(f'Unseen human proteins (identity < {100 * MAX_IDENTITY:.0f}%) with >= {MIN_SITES} localized sites: '
      f'{len(candidates)}')
print(candidates.to_string(index=False, float_format='{:.1f}'.format))

# %%
import subprocess
subprocess.run('pip install -q biopython', shell=True, check=True)
from Bio import Align
from Bio.Align import substitution_matrices

# ---------------------------------------------------------------------------
# Case study 1: ZFR (zinc finger RNA-binding protein), human Q96KR1
#   Experimental sites (localized to one residue) from the external validation:
#     human ZFR: NOTISE 2026 (HEK293T) and Fu 2026 (HEK293T, HepG2, Jurkat);
#     mouse Zfr (O88532): mouse-brain study 2025, transferred to human positions through a
#     global alignment (BLOSUM62, gap open -10, extend -0.5) when the aligned human residue is
#     the same S or T.
#   Scores: the final model's calibrated probability for every S/T of human ZFR, as computed
#   for the external validation (NB05), with the call at the validation cut-off and the rank
#   within the protein.
# ---------------------------------------------------------------------------
HUMAN, MOUSE = 'Q96KR1', 'O88532'
ext = pd.read_parquet('data_processed/external_sites_mapped.parquet')
proteins = pd.read_parquet('data_processed/external_proteins.parquet').set_index('seq_id')
scores = pd.read_parquet('data_processed/final_model/external_scores.parquet')

human_rows = ext[(ext['resolved_accession'] == HUMAN) & ext['localized']]
mouse_rows = ext[(ext['resolved_accession'] == MOUSE) & ext['localized']]
human_ids, mouse_ids = human_rows['seq_id'].unique(), mouse_rows['seq_id'].unique()
assert len(human_ids) == 1 and len(mouse_ids) == 1, (human_ids, mouse_ids)   # one sequence each
h_seq, m_seq = proteins.at[human_ids[0], 'sequence'], proteins.at[mouse_ids[0], 'sequence']
print(f'Human ZFR {HUMAN}: {len(h_seq)} aa   mouse Zfr {MOUSE}: {len(m_seq)} aa')

# Our dataset
atlas = pd.read_parquet('data_processed/atlas5_proteins.parquet')
for name, acc, s in [('human', HUMAN, h_seq), ('mouse', MOUSE, m_seq)]:
    hit = atlas[(atlas['protein_id'].str.split('-').str[0] == acc) | (atlas['sequence'] == s)]
    print(f'{name} ZFR in our dataset: ' + ('; '.join(f'{r.protein_id} ({r.split}), labelled {sorted(r.positive)}'
                                                     for r in hit.itertuples()) or 'absent'))

# Mouse positions -> human positions
aligner = Align.PairwiseAligner(mode='global', substitution_matrix=substitution_matrices.load('BLOSUM62'),
                                open_gap_score=-10, extend_gap_score=-0.5)
aln = aligner.align(h_seq, m_seq)[0]
m_to_h = {}
for (h0, h1), (m0, m1) in zip(*aln.aligned):
    for k in range(h1 - h0):
        m_to_h[m0 + k + 1] = h0 + k + 1
identity = sum(h_seq[h - 1] == m_seq[m - 1] for m, h in m_to_h.items()) / len(h_seq)
print(f'Human-mouse identity: {100 * identity:.1f}% of human positions')

sources = {}
for study, g in human_rows.groupby('study'):
    sources[study] = {int(p) for p in g['position']}
mouse_pos = [int(p) for p in mouse_rows['position']]
sources['Mouse brain 2025 (mouse)'] = {m_to_h[p] for p in mouse_pos if p in m_to_h and h_seq[m_to_h[p] - 1] == m_seq[p - 1]}
print(f'Mouse sites: {len(mouse_pos)}; transferred to the same human S/T: {len(sources["Mouse brain 2025 (mouse)"])}')

# Scores of every S/T of human ZFR
zfr = scores[scores['seq_id'] == human_ids[0]].copy()
zfr['rank'] = zfr['probability'].rank(ascending=False, method='min').astype(int)
for src, pos in sources.items():
    zfr[src] = zfr['position'].isin(pos)
zfr['sources'] = zfr[list(sources)].sum(axis=1)
zfr.to_csv(f'{CASE_DIR}/ZFR_human_scores.csv', index=False)

print(f'\nS/T in human ZFR: {len(zfr)}   called positive at the validation cut-off: {zfr["called"].sum()}')
print('\nExperimental sites (any source), by position:')
cols = ['position', 'residue', 'probability', 'called', 'rank'] + list(sources) + ['sources']
print(zfr[zfr['sources'] > 0][cols].to_string(index=False, float_format='{:.3f}'.format))

summary = []
for label, mask in [*[(src, zfr[src]) for src in sources], ('in >= 2 sources', zfr['sources'] >= 2),
                    ('not reported', zfr['sources'] == 0)]:
    d = zfr[mask]
    summary.append({'sites': label, 'n': len(d), 'called': int(d['called'].sum()),
                    'median rank': d['rank'].median(), 'median probability': d['probability'].median()})
print('\nSummary:')
print(pd.DataFrame(summary).to_string(index=False, float_format='{:.3f}'.format))

# %%
import io, json
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------
# Figure: ZFR (human, Q96KR1) - experimental O-GlcNAc sites and model scores along the sequence
#   (top) UniProt annotation: DZF domain and disordered regions (features retrieved from UniProt).
#   (middle) Calibrated probability of every S/T (stems); sites reported by at least one
#   2025-26 study in blue, the others in grey; orange line: validation cut-off. Labels: sites
#   reported by >= 3 sources.
#   (bottom) Source of each reported site: the three human studies, the mouse-brain study
#   (mouse sites transferred through the alignment), and, for reference, the positives already
#   in Atlas 5.0 (earlier literature; ZFR is a test protein, never used for training).
#   Frontiers figure rules: 180 mm wide, text >= 8 pt, solid lines only, no red or green; PDF and
#   RGB TIFF at 300 dpi.
# ---------------------------------------------------------------------------
FIG_DIR = 'results/figures/case_studies'
os.makedirs(FIG_DIR, exist_ok=True)
BLUE, GREY, LIGHT, ORANGE, INK = '#2a78d6', '#a9a8a2', '#cfcec8', '#d9730d', '#4a4945'

features = json.load(urllib.request.urlopen(f'https://rest.uniprot.org/uniprotkb/{HUMAN}.json', timeout=60))['features']
domains = [(f['description'], f['location']['start']['value'], f['location']['end']['value'])
           for f in features if f['type'] in ('Domain', 'Zinc finger')]
disordered = [(f['location']['start']['value'], f['location']['end']['value'])
              for f in features if f['type'] == 'Region' and f.get('description') == 'Disordered']
atlas_pos = set()
for r in atlas[atlas['protein_id'].str.split('-').str[0] == HUMAN].itertuples():
    atlas_pos |= set(int(p) for p in r.positive)
ROWS = [('NOTISE 2026', 'NOTISE (HEK293T)'), ('Fu 2026', 'Fu (3 cell lines)'), ('HCC liver 2026', 'HCC liver'),
        ('Mouse brain 2025 (mouse)', 'Mouse brain (mouse)')]
L = len(h_seq)

def save_figure(fig, name, width_mm):
    """PDF (vector, for the LaTeX manuscript) and RGB TIFF at 300 dpi (production format)."""
    assert abs(fig.get_size_inches()[0] * 25.4 - width_mm) < 1e-6
    fig.savefig(f'{FIG_DIR}/{name}.pdf')
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=300)
    Image.open(buffer).convert('RGB').save(f'{FIG_DIR}/{name}.tif', dpi=(300, 300))

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False})
fig, (ax_dom, ax_p, ax_src) = plt.subplots(3, 1, figsize=(180 / 25.4, 95 / 25.4), sharex=True, layout='constrained',
                                           gridspec_kw={'height_ratios': [0.5, 3, 1.6]})

# Annotation track
ax_dom.plot([1, L], [0, 0], color=GREY, linewidth=1.5)
for s, e in disordered:
    ax_dom.fill_between([s, e], -0.25, 0.25, color=LIGHT, linewidth=0)
for name, s, e in domains:
    ax_dom.fill_between([s, e], -0.4, 0.4, color='#8a8984', linewidth=0)
    ax_dom.text((s + e) / 2, 0, name, ha='center', va='center', color='white')
ax_dom.set_ylim(-0.6, 0.6)
ax_dom.axis('off')
ax_dom.text(1, 0.55, 'disordered regions (light) and domains (dark), UniProt', color=INK, va='bottom')

# Probability stems
reported = zfr['sources'] > 0
for mask, colour, z in [(~reported, GREY, 1), (reported, BLUE, 3)]:
    d = zfr[mask]
    ax_p.vlines(d['position'], 0, d['probability'], color=colour, linewidth=1, zorder=z)
    ax_p.scatter(d['position'], d['probability'], s=6 + 8 * d['sources'], color=colour, zorder=z + 1,
                 edgecolors='white', linewidths=0.5)
ax_p.axhline(EXT_CUTOFF, color=ORANGE, linewidth=1)
ax_p.text(L, EXT_CUTOFF, ' cut-off', color=ORANGE, va='center', ha='left')
for r in zfr[zfr['sources'] >= 3].itertuples():
    ax_p.annotate(f'{r.residue}{r.position}', (r.position, r.probability), xytext=(0, 4), textcoords='offset points',
                  ha='center', va='bottom', color=INK)
ax_p.set_ylabel('Predicted probability')
ax_p.set_ylim(0, 1.15 * zfr['probability'].max())

# Sources
for k, (src, label) in enumerate(ROWS + [('atlas', 'Atlas 5.0 (earlier)')]):
    y = len(ROWS) - k
    pos = sorted(atlas_pos) if src == 'atlas' else zfr.loc[zfr[src], 'position']
    ax_src.scatter(pos, [y] * len(pos), s=10, color=GREY if src == 'atlas' else BLUE, edgecolors='none')
    ax_src.text(-0.01 * L, y, label, ha='right', va='center')
ax_src.set_yticks([])
ax_src.spines['left'].set_visible(False)
ax_src.set_ylim(-0.5, len(ROWS) + 0.5)
ax_src.set_xlim(-0.01 * L, 1.01 * L)
ax_src.set_xlabel('Position in human ZFR (Q96KR1)')

fig.suptitle(f'ZFR: {int(reported.sum())} sites reported in 2025-26 studies (blue) among {len(zfr)} S/T', x=0.01,
             ha='left', fontsize=8)
save_figure(fig, 'figure_case_ZFR', 180)
plt.show()

# %%
# ---------------------------------------------------------------------------
# Case study 2: YEATS2, human Q9ULM3
#   Experimental sites (localized to one residue): the human external-validation studies, and
#   T604 from the functional study of Zhao et al. 2025, J Biol Chem (YEATS2 O-GlcNAcylation and
#   the ATAC complex in lung cancer; ETD mass spectrometry and the T604A mutant; PMID 40541806).
#   Scores: the final model's calibrated probability for every S/T of YEATS2, as computed for
#   the external validation (NB05), with the call at the validation cut-off and the rank within
#   the protein. Sites already labelled in our dataset (YEATS2 is a test protein) are listed.
# ---------------------------------------------------------------------------
Y_ACC = 'Q9ULM3'
Y_LITERATURE = {604: 'Zhao et al. 2025 (PMID 40541806)'}

y_rows = ext[(ext['resolved_accession'] == Y_ACC) & ext['localized']]
y_ids = y_rows['seq_id'].unique()
assert len(y_ids) == 1, y_ids
y_seq = proteins.at[y_ids[0], 'sequence']
print(f'YEATS2 {Y_ACC}: {len(y_seq)} aa; studies reporting it: {sorted(y_rows["study"].unique())}')
assert all(y_seq[p - 1] in 'ST' for p in Y_LITERATURE), 'literature residue is not S/T in this sequence'

hit = atlas[(atlas['protein_id'].str.split('-').str[0] == Y_ACC) | (atlas['sequence'] == y_seq)]
y_labelled, y_split = set(), 'absent'
for r in hit.itertuples():
    y_split, y_labelled = r.split, set(int(p) for p in r.positive)
    print(f'In our dataset as {r.protein_id} ({r.split}); labelled positives {sorted(y_labelled)}')

y_sources = {study: {int(p) for p in g['position']} for study, g in y_rows.groupby('study')}
y_sources['Zhao 2025 (functional)'] = set(Y_LITERATURE)

yeats2 = scores[scores['seq_id'] == y_ids[0]].copy()
yeats2['rank'] = yeats2['probability'].rank(ascending=False, method='min').astype(int)
for src, pos in y_sources.items():
    yeats2[src] = yeats2['position'].isin(pos)
yeats2['sources'] = yeats2[list(y_sources)].sum(axis=1)
yeats2['labelled in our dataset'] = np.where(yeats2['position'].isin(y_labelled), y_split, 'no')
yeats2.to_csv(f'{CASE_DIR}/YEATS2_scores.csv', index=False)

print(f'\nS/T in YEATS2: {len(yeats2)}   called positive at the validation cut-off: {yeats2["called"].sum()}')
print('\nExperimental sites (any source), by position:')
cols = ['position', 'residue', 'probability', 'called', 'rank'] + list(y_sources) + ['sources', 'labelled in our dataset']
print(yeats2[yeats2['sources'] > 0][cols].to_string(index=False, float_format='{:.3f}'.format))

summary = []
for label, mask in [*[(src, yeats2[src]) for src in y_sources], ('in >= 2 sources', yeats2['sources'] >= 2),
                    ('not reported', yeats2['sources'] == 0)]:
    d = yeats2[mask]
    summary.append({'sites': label, 'n': len(d), 'called': int(d['called'].sum()),
                    'median rank': d['rank'].median(), 'median probability': d['probability'].median()})
print('\nSummary:')
print(pd.DataFrame(summary).to_string(index=False, float_format='{:.3f}'.format))
print('\nTop 15 S/T of YEATS2 by predicted probability:')
print(yeats2.nsmallest(15, 'rank')[['position', 'residue', 'probability', 'called', 'sources',
                                    'labelled in our dataset']].to_string(index=False, float_format='{:.3f}'.format))

# %%
import io, json
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

# ---------------------------------------------------------------------------
# Figure: YEATS2 (human, Q9ULM3) - experimental O-GlcNAc sites and model scores along the sequence
#   (A) UniProt annotation: domains and disordered regions (features retrieved from UniProt).
#   (B) Calibrated probability of every S/T (stems); sites reported by at least one 2025-26 study
#   in blue (black outline: not in O-GlcNAcAtlas 5.0, i.e. new), the others in grey; orange line:
#   validation cut-off. Labels: sites reported by >= 2 sources; FOCUS (the site of Zhao et al.
#   2025) also with its rank among the S/T of the protein.
#   (C) Source of each reported site: the three human studies, the functional study of
#   Zhao et al. 2025 (T604), and, for reference, the positives already in O-GlcNAcAtlas 5.0
#   (earlier literature; YEATS2 is a test protein, never used for training), as dark diamonds so
#   they are not read as the grey 'other S/T' of (B).
#   Frontiers figure rules: 180 mm wide, text >= 8 pt, panels labelled (A)-(C), solid lines only,
#   no red or green; PDF and RGB TIFF at 300 dpi.
# ---------------------------------------------------------------------------
FIG_DIR = 'results/figures/case_studies'
os.makedirs(FIG_DIR, exist_ok=True)
BLUE, GREY, LIGHT, ORANGE, INK = '#2a78d6', '#a9a8a2', '#cfcec8', '#d9730d', '#4a4945'
FOCUS = 604   # T604, the site studied by Zhao et al. 2025

features = json.load(urllib.request.urlopen(f'https://rest.uniprot.org/uniprotkb/{Y_ACC}.json', timeout=60))['features']
domains = [(f['description'], f['location']['start']['value'], f['location']['end']['value'])
           for f in features if f['type'] in ('Domain', 'Zinc finger')]
disordered = [(f['location']['start']['value'], f['location']['end']['value'])
              for f in features if f['type'] == 'Region' and f.get('description') == 'Disordered']
ROWS = [('NOTISE 2026', 'Griffin et al. (2026)'), ('Fu 2026', 'Fu et al. (2026)'), ('HCC liver 2026', 'Hou et al. (2026)'),
        ('Zhao 2025 (functional)', 'Zhao et al. (2025)')]
L = len(y_seq)

def save_figure(fig, name, width_mm):
    """PDF (vector, for the LaTeX manuscript) and RGB TIFF at 300 dpi (production format)."""
    assert abs(fig.get_size_inches()[0] * 25.4 - width_mm) < 1e-6
    fig.savefig(f'{FIG_DIR}/{name}.pdf')
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=300)
    Image.open(buffer).convert('RGB').save(f'{FIG_DIR}/{name}.tif', dpi=(300, 300))

def ordinal(n):
    return f'{n}{"th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")}'

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False})
fig, (ax_dom, ax_p, ax_src) = plt.subplots(3, 1, figsize=(180 / 25.4, 100 / 25.4), sharex=True, layout='constrained',
                                           gridspec_kw={'height_ratios': [0.5, 3, 1.6]})

# (A) Annotation track
ax_dom.plot([1, L], [0, 0], color=GREY, linewidth=1.5)
for s, e in disordered:
    ax_dom.fill_between([s, e], -0.25, 0.25, color=LIGHT, linewidth=0)
for name, s, e in domains:
    ax_dom.fill_between([s, e], -0.4, 0.4, color='#8a8984', linewidth=0)
    ax_dom.text((s + e) / 2, 0, name, ha='center', va='center', color='white')
ax_dom.set_ylim(-0.6, 0.6)
ax_dom.axis('off')
ax_dom.set_title('(A) Domains (dark) and disordered regions (light), UniProt', loc='left')

# (B) Probability stems
reported = yeats2['sources'] > 0
new = reported & (yeats2['labelled in our dataset'] == 'no')
for mask, colour, z in [(~reported, GREY, 1), (reported, BLUE, 3)]:
    d = yeats2[mask]
    ax_p.vlines(d['position'], 0, d['probability'], color=colour, linewidth=1, zorder=z)
    ax_p.scatter(d['position'], d['probability'], s=6 + 8 * d['sources'], color=colour, zorder=z + 1,
                 edgecolors=np.where(new[mask], 'black', 'white'), linewidths=np.where(new[mask], 1.0, 0.5))
ax_p.axhline(EXT_CUTOFF, color=ORANGE, linewidth=1)
ax_p.annotate(f'Cut-off ({EXT_CUTOFF:.3f})', (1, EXT_CUTOFF), xycoords=('axes fraction', 'data'), xytext=(5, 0),
              textcoords='offset points', color=ORANGE, va='center', ha='left', annotation_clip=False)
focus_p = yeats2.loc[yeats2['position'] == FOCUS, 'probability'].iloc[0]
focus_rank = int((yeats2['probability'] > focus_p).sum()) + 1
for r in yeats2[yeats2['sources'] >= 2].itertuples():
    text = f'{r.residue}{r.position}' + (f' ({ordinal(focus_rank)} of {len(yeats2)})' if r.position == FOCUS else '')
    ax_p.annotate(text, (r.position, r.probability), xytext=(0, 4), textcoords='offset points',
                  ha='center', va='bottom', color=INK)
ax_p.set_ylabel('Predicted probability')
ax_p.set_ylim(0, 1.15 * yeats2['probability'].max())
ax_p.set_title(f'(B) Probability of each of the {len(yeats2)} serines and threonines', loc='left')
handles = [Line2D([], [], linestyle='', marker='o', markersize=5, color=BLUE, markeredgecolor='white',
                  label='Reported in 2025-2026'),
           Line2D([], [], linestyle='', marker='o', markersize=5, color=BLUE, markeredgecolor='black',
                  label='Reported in 2025-2026, not in O-GlcNAcAtlas 5.0'),
           Line2D([], [], linestyle='', marker='o', markersize=4, color=GREY, markeredgecolor='white',
                  label='Other serines and threonines'),
           Line2D([], [], linestyle='', marker='D', markersize=4, color=INK, markeredgecolor='none',
                  label='In O-GlcNAcAtlas 5.0 (earlier studies)')]
fig.legend(handles=handles, loc='outside lower center', ncols=2, frameon=False)

# (C) Sources
atlas_pos = sorted(y_labelled)
for k, (src, label) in enumerate(ROWS + [('atlas', 'O-GlcNAcAtlas 5.0')]):
    y = len(ROWS) - k
    pos = atlas_pos if src == 'atlas' else yeats2.loc[yeats2[src], 'position']
    ax_src.scatter(pos, [y] * len(pos), s=12 if src == 'atlas' else 10, marker='D' if src == 'atlas' else 'o',
                   color=INK if src == 'atlas' else BLUE, edgecolors='none')
    ax_src.text(-0.01 * L, y, label, ha='right', va='center')
ax_src.set_yticks([])
ax_src.spines['left'].set_visible(False)
ax_src.set_ylim(-0.5, len(ROWS) + 0.5)
ax_src.set_xlim(-0.01 * L, 1.01 * L)
ax_src.set_xlabel('Position in human YEATS2 (Q9ULM3)')
ax_src.set_title('(C) Sites reported by each study', loc='left')

save_figure(fig, 'figure_case_YEATS2', 180)
plt.show()

# %%
import subprocess

# ---------------------------------------------------------------------------
# Screening of the case-study panel: experimentally localized O-GlcNAc sites from 2025-26 papers
#   For each protein: the reviewed UniProtKB entries for its gene in the candidate organisms
#   (looked up by gene name), kept when every reported residue matches the sequence; then,
#   against our dataset: same protein (accession or identical sequence) and its split, the
#   reported sites already labelled there, and the highest identity to any training or
#   validation protein (exhaustive MMseqs2 alignment, identity = identical residues / shorter
#   length, as for the splits).
# ---------------------------------------------------------------------------
PANEL = [  # protein, UniProt query, organisms (taxon IDs), reported sites, PMID, site evidence
    ('LATS1', 'gene_exact:LATS1', [9606], {479: 'S', 482: 'S', 484: 'T', 485: 'T'}, 41338457, 'ETD MS'),
    ('Vaccinia A4', 'gene:A4L AND organism_name:"Vaccinia virus"', [None], {96: 'S', 155: 'T', 172: 'T', 248: 'T'},
     40407344, 'MS (HCD fragments listed)'),
    ('SEC24C', 'gene_exact:SEC24C', [9606, 10090], {205: 'S'}, 40617351, 'MS, "unambiguously identified"'),
    ('G3BP1', 'gene_exact:G3BP1', [9606, 10090, 10116], {268: 'T'}, 42622529, 'EThcD MS'),
    ('YEATS2', 'gene_exact:YEATS2', [9606], {604: 'T'}, 40541806, 'ETD MS + T604A'),
    ('UBAP2L', 'gene_exact:UBAP2L', [9606], {305: 'S'}, 41029457, 'ETD MS'),
    ('METTL3', 'gene_exact:METTL3', [9606, 10090], {186: 'T', 192: 'S', 193: 'S'}, 40651967,
     'one MS spectrum (S193) + mutants'),
]

def uniprot_entries(query, taxon):
    """Reviewed UniProtKB entries matching a query (optionally restricted to an organism)."""
    q = f'({query}) AND reviewed:true' + (f' AND organism_id:{taxon}' if taxon else '')
    url = 'https://rest.uniprot.org/uniprotkb/search?format=fasta&query=' + urllib.parse.quote(q)
    text = urllib.request.urlopen(url, timeout=60).read().decode()
    out = []
    for e in text.split('>')[1:]:
        header, *lines = e.strip().split('\n')
        out.append((header.split('|')[1], header, ''.join(lines)))
    return out

atlas = pd.read_parquet('data_processed/atlas5_proteins.parquet')
seen = atlas[atlas['split'].isin(['train', 'val'])]
rows, panel_seqs = [], {}
for name, query, taxa, sites, pmid, evidence in PANEL:
    for taxon in taxa:
        for acc, header, seq in uniprot_entries(query, taxon):
            match = all(p <= len(seq) and seq[p - 1] == r for p, r in sites.items())
            hit = atlas[(atlas['protein_id'].str.split('-').str[0] == acc) | (atlas['sequence'] == seq)]
            split = ', '.join(hit['split']) if len(hit) else 'absent'
            labelled = sorted({p for r in hit.itertuples() for p in sites if p in set(r.positive)})
            rows.append({'protein': name, 'entry': acc, 'organism': header.split('OS=')[1].split(' OX=')[0],
                         'length': len(seq), 'residues match': match, 'our dataset': split,
                         'sites already labelled': labelled, 'PMID': pmid, 'site evidence': evidence})
            if match:
                panel_seqs[f'{name}|{acc}'] = seq
screen = pd.DataFrame(rows)

# Highest identity to a training or validation protein (exhaustive MMseqs2 alignment)
MMSEQS = '/content/mmseqs/bin/mmseqs'
if not os.path.exists(MMSEQS):
    subprocess.run('wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -O /content/mmseqs.tar.gz '
                   '&& tar xzf /content/mmseqs.tar.gz -C /content', shell=True, check=True)
D = '/content/panel_overlap'
os.makedirs(D, exist_ok=True)
with open(f'{D}/q.fasta', 'w') as f:
    f.writelines(f'>{k}\n{s}\n' for k, s in panel_seqs.items())
with open(f'{D}/t.fasta', 'w') as f:
    f.writelines(f'>{k}\n{s}\n' for k, s in zip(seen['protein_id'], seen['sequence']))
cmds = [f'{MMSEQS} createdb {D}/q.fasta {D}/qdb', f'{MMSEQS} createdb {D}/t.fasta {D}/tdb',
        f'ln -sf {D}/tdb.index {D}/pref',
        f"""awk -v size=$(wc -c < {D}/tdb.index) '{{ print $1"\\t0\\t"size; }}' {D}/qdb.index > {D}/pref.index""",
        f"""awk 'BEGIN {{ printf("%c%c%c%c",7,0,0,0); exit; }}' > {D}/pref.dbtype""",
        f'{MMSEQS} align {D}/qdb {D}/tdb {D}/pref {D}/aln --alignment-mode 3 -e 0.001 --min-seq-id 0 -c 0 '
        f'--max-accept 100000 --threads {os.cpu_count()}',
        f'{MMSEQS} convertalis {D}/qdb {D}/tdb {D}/aln {D}/aln.tsv --format-output query,target,fident,alnlen,qlen,tlen']
for c in cmds:
    subprocess.run(c, shell=True, check=True, capture_output=True)
aln = pd.read_csv(f'{D}/aln.tsv', sep='\t', names=['query', 'target', 'fident', 'alnlen', 'qlen', 'tlen'])
aln['identity'] = (aln['fident'] * aln['alnlen']).round() / aln[['qlen', 'tlen']].min(axis=1)
best = aln.sort_values('identity', ascending=False).drop_duplicates('query').set_index('query')
key = screen['protein'] + '|' + screen['entry']
screen['max identity to train/val'] = key.map(best['identity'])
screen['closest train/val protein'] = key.map(best['target'])
screen.to_csv(f'{CASE_DIR}/panel_screen.csv', index=False)
pd.set_option('display.width', 250, 'display.max_colwidth', 40)
print(screen.to_string(index=False, float_format='{:.2f}'.format))

# %%
# ---------------------------------------------------------------------------
# Scores of the case-study panel (final model, validation cut-off)
#   One entry per protein (the organism of the study; Vaccinia A4: strain Western Reserve).
#   Group of each reported site, from the screen above:
#     unseen          protein absent from our dataset, no homolog in training/validation, or a
#                     test protein (never used for training or model selection);
#     seen, site new  training protein, this site not labelled positive (negative or masked in
#                     training; the label is printed at the end);
#     seen, site known training protein, this site already labelled positive in training.
#   For every reported site: calibrated probability, call, and rank among the protein's S/T
#   (1 = highest probability; percentile = share of the protein's S/T scored below it).
# ---------------------------------------------------------------------------
SELECTED = {'LATS1': 'O95835', 'Vaccinia A4': 'P29191', 'SEC24C': 'P53992', 'G3BP1': 'Q13283',
            'YEATS2': 'Q9ULM3', 'UBAP2L': 'Q14157', 'METTL3': 'Q86U44'}
print('Vaccinia A4 strains WR and Copenhagen identical:',
      panel_seqs['Vaccinia A4|P29191'] == panel_seqs['Vaccinia A4|P20983'])

rows, all_scores = [], []
for name, query, taxa, sites, pmid, evidence in PANEL:
    acc = SELECTED[name]
    seq = panel_seqs[f'{name}|{acc}']
    entry = screen[(screen['protein'] == name) & (screen['entry'] == acc)].iloc[0]
    s, _ = score_proteins([acc], [seq])
    s['rank_in_protein'] = s['probability'].rank(ascending=False, method='min').astype(int)
    s['percentile'] = 100 * (s['probability'].rank(method='min') - 1) / len(s)
    s['protein'] = name
    all_scores.append(s)
    for p in sites:
        r = s[s['position'] == p].iloc[0]
        # group per reported site: a site of a training protein is "known" only if it was itself labelled positive
        if entry['our dataset'] in ('absent', 'test'):
            group = 'unseen'
        elif p in entry['sites already labelled']:
            group = 'seen, site known'
        else:
            group = 'seen, site new'
        rows.append({'protein': name, 'entry': acc, 'group': group, 'site': f'{r.residue}{p}',
                     'probability': r.probability, 'called': r.called, 'rank': f'{r.rank_in_protein}/{len(s)}',
                     'percentile': r.percentile,
                     'labelled in our dataset': entry['our dataset'] if p in entry['sites already labelled'] else 'no',
                     'PMID': pmid})
panel_results = pd.DataFrame(rows)
pd.concat(all_scores).to_csv(f'{CASE_DIR}/panel_all_scores.csv', index=False)
panel_results.to_csv(f'{CASE_DIR}/panel_site_results.csv', index=False)

print(f'\nReported sites, final model (cut-off {EXT_CUTOFF:.4f}):')
print(panel_results.to_string(index=False, float_format='{:.3f}'.format))
print('\nSummary by group:')
print(panel_results.groupby('group').agg(sites=('site', 'size'), called=('called', 'sum'),
                                         median_percentile=('percentile', 'median')).to_string())

# Label in our dataset of the "seen, site new" sites (negative or masked), from sites.parquet
site_labels = pd.read_parquet('data_processed/embeddings/sites.parquet', columns=['protein_id', 'position', 'label', 'split'])
print('\nTraining label of the "seen, site new" sites:')
for r in panel_results[panel_results['group'] == 'seen, site new'].itertuples():
    pos = int(r.site[1:])
    hit = site_labels[(site_labels['protein_id'].str.split('-').str[0] == r.entry) & (site_labels['position'] == pos)]
    print(f'  {r.protein} {r.entry} {r.site}: '
          + ('; '.join(f'{h.protein_id} {h.split} {h.label}' for h in hit.itertuples())
             or 'no row with this accession (protein matched by sequence)'))

# %%
# ---------------------------------------------------------------------------
# Label in our dataset of the reported sites of training proteins that were not labelled positive
# (negative or masked), from sites.parquet
# ---------------------------------------------------------------------------
site_labels = pd.read_parquet('data_processed/embeddings/sites.parquet', columns=['protein_id', 'position', 'label', 'split'])
new_sites = panel_results[panel_results['group'] == 'seen, site new']
for r in new_sites.itertuples():
    pos = int(r.site[1:])
    hit = site_labels[(site_labels['protein_id'].str.split('-').str[0] == r.entry) & (site_labels['position'] == pos)]
    print(f'{r.protein} {r.entry} {r.site}: ' + ('; '.join(f'{h.protein_id} {h.split} {h.label}' for h in hit.itertuples())
                                                  or 'no row with this accession (matched by sequence?)'))
