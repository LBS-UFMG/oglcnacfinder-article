# %%
import os
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))

import subprocess, sys
import tensorflow as tf
subprocess.run(f'pip install -q "tf-keras~={tf.__version__.rsplit(".", 1)[0]}.0" gensim biopython nltk',
               shell=True, check=True)

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score

# ---------------------------------------------------------------------------
# O-GlcNAcPRED-DL (Hu et al. 2024, J Proteome Res 23:95-106), run with the authors'
# trained models
#   Code and models: github.com/hufengzhu/O-GlcNAcPRED-DL-master, pinned to its last commit.
#   Two ensembles (human, mouse) of five CNN-BiLSTM models on 29-residue windows centred on
#   each S/T; prediction scripts Hwmean_voting.py (human) and Mwmean_voting.py (mouse).
#   The scripts are run as released, with these changes (each tagged in the saved copies):
#     - Keras 2 is imported as tf_keras (current TensorFlow ships Keras 3);
#     - the feature modules are imported from a package named O_gly, as in the scripts
#       (the repository stores them under feature/);
#     - the hard-coded AAindex path (D:\python code\...) points to the repository copy;
#     - the input FASTA file is a parameter;
#     - the final print of every score is removed.
#   The scripts standardize the AAindex features with a StandardScaler fitted on the input
#   file itself, so each file is scored in one run.
# ---------------------------------------------------------------------------
REPO_URL = 'https://github.com/hufengzhu/O-GlcNAcPRED-DL-master'
REPO_COMMIT = '2e953b33672abbe002ea163d7a52e4b2f42bf3a8'   # 2023-09-28, "Update README.md"
REPO = '/content/O-GlcNAcPRED-DL'
PKG = '/content/ogp_pkg'   # holds the O_gly package
OUT_DIR = 'data_processed/other_tools/O-GlcNAcPRED-DL'
os.makedirs(f'{OUT_DIR}/code', exist_ok=True)

if not os.path.exists(REPO):
    subprocess.run(f'git clone -q {REPO_URL} {REPO}', shell=True, check=True)
subprocess.run(f'git -C {REPO} checkout -q {REPO_COMMIT}', shell=True, check=True)

def patch(source, changes):
    """Apply line changes {old: new} to the source; new=None removes the line. Each changed line
    is tagged with its original text, and each change must match at least once."""
    lines = source.split('\n')
    for old, new in changes.items():
        hits = [i for i, line in enumerate(lines) if old in line]
        assert hits, f'not found: {old}'
        for i in hits:
            original = lines[i].strip()
            if new is None:
                lines[i] = f'# [removed] {original}'
            else:
                lines[i] = f'{lines[i].replace(old, new)}  # [modified] original: {original}'
    return '\n'.join(lines)

def save(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

# Feature modules, importable as O_gly.AAindex.AAindex_sl and O_gly.word2vec.w2v_fea
AAINDEX_FILE = f'{REPO}/feature/AAindex/AAindex_normalized.txt'
aaindex_src = patch(open(f'{REPO}/feature/AAindex/AAindex_sl.py', encoding='utf-8').read(),
                    {r"r'D:\python code\O-GlcNAcPRED-DL-master\feature\AAindex\AAindex_normalized.txt'": repr(AAINDEX_FILE)})
w2v_src = open(f'{REPO}/feature/word2vec/w2v_fea.py', encoding='utf-8').read()
for folder, name, text in [('AAindex', 'AAindex_sl', aaindex_src), ('word2vec', 'w2v_fea', w2v_src)]:
    save(f'{PKG}/O_gly/{folder}/{name}.py', text)
    save(f'{PKG}/O_gly/{folder}/__init__.py', '')
    save(f'{OUT_DIR}/code/O_gly/{folder}/{name}.py', text)
save(f'{PKG}/O_gly/__init__.py', '')
sys.path.insert(0, PKG)

# Prediction scripts
SCRIPT_CHANGES = {
    'from keras.layers import *': 'from tf_keras.layers import *',
    'from keras.models import *': 'from tf_keras.models import *',
    'from keras.callbacks import': 'from tf_keras.callbacks import',
    'print(b)': None,
}
INPUT_LINE = {'human': "file = './test_data/independent test sets/Ind_H.fasta'",
              'mouse': "file = './test_data/independent test sets/Ind_M.fasta'"}
SCRIPTS = {}
for species, script in [('human', 'Hwmean_voting.py'), ('mouse', 'Mwmean_voting.py')]:
    SCRIPTS[species] = patch(open(f'{REPO}/{script}', encoding='utf-8').read(),
                             {**SCRIPT_CHANGES, INPUT_LINE[species]: 'file = FASTA_FILE'})
    save(f'{OUT_DIR}/code/{script}', SCRIPTS[species])

def predict(species, fasta_file):
    """Ensemble score of every window in the FASTA file (one run, one scaler fit on the file)."""
    namespace = {'FASTA_FILE': os.path.abspath(fasta_file)}
    cwd = os.getcwd()
    os.chdir(REPO)   # the scripts read the models and word2vec files from relative paths
    try:
        exec(SCRIPTS[species], namespace)
    finally:
        os.chdir(cwd)
    return np.asarray(namespace['b']).ravel()

def metrics(y, score):
    """Sn, Sp, Acc, MCC (%, MCC as a fraction) at the published cut-off (score >= 0.5), and AUC (%)."""
    pred = score >= 0.5
    return {'Sn': 100 * pred[y == 1].mean(), 'Sp': 100 * (~pred[y == 0]).mean(), 'Acc': 100 * (pred == y).mean(),
            'MCC': matthews_corrcoef(y, pred), 'AUC': 100 * roc_auc_score(y, score)}

# ---------------------------------------------------------------------------
# Installation check: their independent test sets, Ind_H and Ind_M
#   The FASTA files carry no labels (headers Seq1, Seq2, ...). Positives: 299 (human) and
#   143 (mouse), one tenth of the 2,990 and 1,434 positives (9:1 split, Hu et al. 2024, p98);
#   the remaining windows (1,495 and 715) are negatives. These counts give Table 4's
#   percentages exactly (human Sn 75.25% = 225/299, Sp 71.51% = 1,069/1,495).
#   Assumed order: positives first. Installation and order are confirmed if the scores
#   reproduce Table 4 (Hu et al. 2024, p102).
# ---------------------------------------------------------------------------
CHECK = {'human': ('Ind_H.fasta', 299, {'Sn': 75.25, 'Sp': 71.51, 'Acc': 72.13, 'MCC': 0.3624, 'AUC': 82.39}),
         'mouse': ('Ind_M.fasta', 143, {'Sn': 78.32, 'Sp': 72.31, 'Acc': 73.31, 'MCC': 0.3928, 'AUC': 82.83})}
rows = []
for species, (fasta, n_pos, published) in CHECK.items():
    score = predict(species, f'{REPO}/test_data/independent test sets/{fasta}')
    y = (np.arange(len(score)) < n_pos).astype(int)
    assert len(score) == 6 * n_pos
    ours = metrics(y, score)
    rows += [{'set': fasta, 'source': 'Table 4', **published}, {'set': fasta, 'source': 'this run', **ours}]

check = pd.DataFrame(rows)
check.to_csv(f'{OUT_DIR}/installation_check.csv', index=False)
print(check.to_string(index=False, formatters={'Sn': '{:.2f}'.format, 'Sp': '{:.2f}'.format, 'Acc': '{:.2f}'.format,
                                                'MCC': '{:.4f}'.format, 'AUC': '{:.2f}'.format}))

# %%
import json, subprocess, time
from datetime import date

# ---------------------------------------------------------------------------
# Compute records (compute statement): the wall-clock time of each timed step, with the hardware,
# saved as results/compute/{notebook}_{step}.json; collected by the summary notebook (NB15).
# ---------------------------------------------------------------------------
def gpu_name():
    """Name of the first GPU (nvidia-smi), or None on a CPU runtime."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                             capture_output=True, text=True, check=True).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return out.splitlines()[0] if out else None

def save_compute(notebook, step, seconds, **counts):
    """Wall-clock time of one step, with the hardware, saved as results/compute/{notebook}_{step}.json."""
    record = {'notebook': notebook, 'step': step, 'seconds': round(seconds, 1), **counts,
              'gpu': gpu_name(), 'cpu_cores': os.cpu_count(), 'date': date.today().isoformat()}
    os.makedirs('results/compute', exist_ok=True)
    with open(f'results/compute/{notebook}_{step}.json', 'w') as f:
        json.dump(record, f, indent=1)
    print(f'{notebook} {step}: {seconds / 60:.1f} min on {record["gpu"] or "CPU"}   ' +
          '   '.join(f'{k} {v:,}' for k, v in counts.items()))

# %%
import ast, time
from Bio import SeqIO
from sklearn.metrics import average_precision_score

# ---------------------------------------------------------------------------
# Our test set: 29-residue windows from their windowing code (seq_cut/cut.py)
#   na_se() and strcat() are taken from the repository and executed with one change: the
#   output file is a parameter (the script writes NUP62_cut.fasta). na_se() writes a window
#   for every S/T, with the 0-based position in the header; windows near the ends are
#   padded with X.
#   Scored windows: every S/T of the test proteins except masked residues (the same sites
#   as the embeddings notebook), written to one FASTA file in protein and position order.
# ---------------------------------------------------------------------------
WORK = '/content/ogp_work'
os.makedirs(WORK, exist_ok=True)
PROTEINS_FASTA, CUT_FASTA, SITES_FASTA = (f'{WORK}/test_proteins.fasta', f'{WORK}/test_proteins_cut.fasta',
                                          f'{WORK}/test_sites.fasta')

cut_src = open(f'{REPO}/seq_cut/cut.py', encoding='utf-8').read()
functions = [ast.get_source_segment(cut_src, n) for n in ast.parse(cut_src).body
             if isinstance(n, ast.FunctionDef) and n.name in ('strcat', 'na_se')]
cut_functions = patch('\n\n'.join(functions), {'"NUP62_cut.fasta"': 'CUT_FASTA'})
save(f'{OUT_DIR}/code/cut_functions.py', cut_functions)
exec('from Bio import SeqIO\nfrom Bio.Seq import Seq\nfrom Bio.SeqRecord import SeqRecord\n' + cut_functions)

proteins = pd.read_parquet('data_processed/atlas5_proteins.parquet')
test = proteins[proteins['split'] == 'test'].reset_index(drop=True)
with open(PROTEINS_FASTA, 'w') as f:
    for pid, seq in zip(test['protein_id'], test['sequence']):
        f.write(f'>{pid}\n{seq}\n')
print('Proteins shorter than 29 residues (printed by na_se):')
na_se(PROTEINS_FASTA)

windows = {}
for record in SeqIO.parse(CUT_FASTA, 'fasta'):
    pid, position = record.id.rsplit('|position=', 1)
    windows[(pid, int(position) + 1)] = str(record.seq)

rows, n_masked = [], 0
for pid, seq, positive, masked, organism in zip(test['protein_id'], test['sequence'], test['positive'],
                                                 test['masked'], test['organism']):
    positive, masked = set(positive), set(masked)
    for i, aa in enumerate(seq):
        if aa not in 'ST':
            continue
        if i + 1 in masked:
            n_masked += 1
            continue
        rows.append({'protein_id': pid, 'organism': organism, 'position': i + 1, 'residue': aa,
                     'label': int(i + 1 in positive), 'window': windows[(pid, i + 1)]})
sites = pd.DataFrame(rows)
assert len(windows) == len(sites) + n_masked and (sites['window'].str.len() == 29).all()

# Same test sites as the embeddings notebook
reference = pd.read_parquet('data_processed/embeddings/sites.parquet', columns=['split', 'label'])
reference = reference[(reference['split'] == 'test') & (reference['label'] != 'masked')]
assert len(sites) == len(reference) and sites['label'].sum() == (reference['label'] == 'positive').sum()

# Target at the window centre (index 14); na_se() leaves it off centre only in proteins
# shorter than 29 residues (their first-residues branch pads a window that is cut short)
off_centre = (sites['window'].str[14] != sites['residue']).to_numpy()
assert set(sites.loc[off_centre, 'protein_id']) <= set(test.loc[test['length'] < 29, 'protein_id'])
print(f'\nSites with the target off the window centre: {off_centre.sum():,} '
      f'(in {sites.loc[off_centre, "protein_id"].nunique()} proteins shorter than 29 residues)')

# Letters outside the 20 standard amino acids: the AAindex encoding maps X to zeros (both
# models) and U to zeros (human model only), and fails on any other letter
extra = set(''.join(sites['window'])) - set('ACDEFGHIKLMNPQRSTVWY')
print(f'Letters outside the 20 standard amino acids: {sorted(extra)}')
assert extra <= {'X'}, f'letters the mouse encoding cannot handle: {sorted(extra - {"X"})}'

with open(SITES_FASTA, 'w') as f:
    for k, w in enumerate(sites['window']):
        f.write(f'>Seq{k + 1}\n{w}\n')

n_pos = sites['label'].sum()
print(f'\nTest proteins: {len(test):,}   S/T sites scored: {len(sites):,} '
      f'({n_pos:,} positive, {len(sites) - n_pos:,} negative)   masked S/T excluded: {n_masked:,}')
print('\nSites per organism (model used for the main result):')
sites['matched_model'] = np.where(sites['organism'] == 'Mus musculus', 'mouse', 'human')
print(sites.groupby(['matched_model', 'organism']).agg(proteins=('protein_id', 'nunique'), sites=('label', 'size'),
                                                       positive=('label', 'sum'))
      .sort_values('sites', ascending=False).to_string())
both = test['organisms'].str.contains('Homo sapiens') & test['organisms'].str.contains('Mus musculus')
print(f'Test proteins whose identical sequence is reported in both human and mouse: {both.sum()} '
      f'(model follows the kept ID\'s organism)')

# ---------------------------------------------------------------------------
# Scores: each model scores all test windows in one run (one scaler fit on all of them).
#   Main result: the mouse model for mouse proteins, the human model for all others.
#   Memory: their word2vec encoding holds 28 copies of the (windows x 28 x 30) float64 array
#   at once, about 17.5 GB for 93k windows, so this needs a high-RAM runtime.
#   Timing: wall-clock time of each model's run (feature encoding and the five networks),
#   saved as results/compute/NB07_test_{model}.json (compute-records cell).
#   If scores from an earlier run exist, the new scores are compared with them before they are replaced.
# ---------------------------------------------------------------------------
for species in ('human', 'mouse'):
    t0 = time.perf_counter()
    print(f'\nScoring with the {species} model (encoding ~93k windows takes a few minutes)...')
    sites[f'score_{species}'] = predict(species, SITES_FASTA)
    save_compute('NB07', f'test_{species}', time.perf_counter() - t0, proteins=len(test), st_sites=len(sites))
sites['score_matched'] = np.where(sites['matched_model'] == 'mouse', sites['score_mouse'], sites['score_human'])
assert sites[['score_human', 'score_mouse']].notna().all().all()

if os.path.exists(f'{OUT_DIR}/test_scores.parquet'):
    previous = pd.read_parquet(f'{OUT_DIR}/test_scores.parquet')
    assert previous[['protein_id', 'position']].equals(sites[['protein_id', 'position']])
    print('Previous run vs this run, max |diff|: ' + ', '.join(
        f'{c} {(previous[c] - sites[c]).abs().max():.1e}' for c in ('score_human', 'score_mouse', 'score_matched')))
sites.drop(columns='window').to_parquet(f'{OUT_DIR}/test_scores.parquet', index=False)

summary = pd.DataFrame([{'score': s, 'AUROC': roc_auc_score(sites['label'], sites[f'score_{s}']),
                         'AUPRC': average_precision_score(sites['label'], sites[f'score_{s}'])}
                        for s in ('matched', 'human', 'mouse')])
print(f'\nAll test sites (positive fraction {n_pos / len(sites):.4f}):')
print(summary.to_string(index=False, float_format='{:.4f}'.format))

# %%
import torch

# ---------------------------------------------------------------------------
# Per-organism comparison: each O-GlcNAcPRED-DL model on its own species only
#   Human test proteins: their human model vs our final model.
#   Mouse test proteins: their mouse model vs our final model.
#   Same sites for both models in each row. Yes/no calls: their published cut-off (score >= 0.5);
#   ours, calibrated probability >= its validation MCC cut-off (NB05).
# ---------------------------------------------------------------------------
ours = pd.read_parquet('data_processed/final_model/test_scores.parquet')[['protein_id', 'position', 'probability']]
OUR_CUTOFF = torch.load('data_processed/final_model/final_model.pt', map_location='cpu')['cutoff']['probability']
scores = sites.merge(ours, on=['protein_id', 'position'], how='left', validate='one_to_one')
assert scores['probability'].notna().all()

def evaluate(y, score, cutoff):
    called = score >= cutoff
    tp = int((called & (y == 1)).sum())
    return {'AUROC': roc_auc_score(y, score), 'AUPRC': average_precision_score(y, score),
            'MCC': matthews_corrcoef(y, called), 'precision': tp / max(called.sum(), 1), 'recall': tp / y.sum()}

rows = []
for organism, their_col, their_name in [('Homo sapiens', 'score_human', 'O-GlcNAcPRED-DL human model'),
                                        ('Mus musculus', 'score_mouse', 'O-GlcNAcPRED-DL mouse model')]:
    sub = scores[scores['organism'] == organism]
    y_sub = sub['label'].to_numpy()
    info = {'organism': organism, 'proteins': sub['protein_id'].nunique(), 'sites': len(sub),
            'positive': int(y_sub.sum())}
    rows.append({**info, 'model': their_name, 'cut-off': 0.5, **evaluate(y_sub, sub[their_col].to_numpy(), 0.5)})
    rows.append({**info, 'model': 'Our model', 'cut-off': OUR_CUTOFF,
                 **evaluate(y_sub, sub['probability'].to_numpy(), OUR_CUTOFF)})

by_organism = pd.DataFrame(rows)
by_organism.to_csv(f'{OUT_DIR}/test_metrics_by_organism.csv', index=False)
print(by_organism.to_string(index=False, float_format='{:.4f}'.format))

# %%
import torch
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

# ---------------------------------------------------------------------------
# Overlap of our test proteins with O-GlcNAcPRED-DL's training data (decision 5b)
#   Their training data (O-GlcNAcAtlas 2.0) are not released, and Atlas 2.0 is not offered for
#   download. As for LM-OGlcNAc-Site (NB06, also trained on Atlas 2.0), the archived Atlas 4.0
#   release is used as a superset of their training proteins (assumption: Atlas 2.0 proteins are
#   contained in Atlas 4.0); the overlap of each test protein with Atlas 4.0 was computed in
#   NB06 (data_processed/other_tools/Atlas4.0/test_protein_overlap.csv):
#     (a) homology: identity >= 30% to any Atlas 4.0 protein (exhaustive MMseqs2 alignment);
#     (b) same protein: accession (isoform suffix ignored) or exact sequence in Atlas 4.0.
#   Each removes whole test proteins. Comparisons: all organisms (their organism-matched model)
#   and each of their models on its own species (human model on human proteins, mouse model on
#   mouse proteins); their call at >= 0.5, ours at the validation MCC cut-off.
# ---------------------------------------------------------------------------
scores = pd.read_parquet(f'{OUT_DIR}/test_scores.parquet')
ours = pd.read_parquet('data_processed/final_model/test_scores.parquet')[['protein_id', 'position', 'probability']]
OUR_CUTOFF = torch.load('data_processed/final_model/final_model.pt', map_location='cpu')['cutoff']['probability']
scores = scores.merge(ours, on=['protein_id', 'position'], how='left', validate='one_to_one')
assert scores['probability'].notna().all()

flags = pd.read_csv('data_processed/other_tools/Atlas4.0/test_protein_overlap.csv').set_index('protein_id')
assert set(scores['protein_id']) <= set(flags.index)
same_protein = scores['protein_id'].map(flags['same_protein']).to_numpy(bool)
homolog_30 = scores['protein_id'].map(flags['homolog_30']).to_numpy(bool)
SUBSETS = {'all test proteins': np.ones(len(scores), bool),
           '(b) same protein removed': ~same_protein,
           '(a) homologs >= 30% removed': ~homolog_30}
GROUPS = {'all organisms': (np.ones(len(scores), bool), 'score_matched', 'O-GlcNAcPRED-DL (matched)'),
          'Homo sapiens': ((scores['organism'] == 'Homo sapiens').to_numpy(), 'score_human', 'O-GlcNAcPRED-DL human'),
          'Mus musculus': ((scores['organism'] == 'Mus musculus').to_numpy(), 'score_mouse', 'O-GlcNAcPRED-DL mouse')}

rows = []
for group, (in_group, their_col, their_name) in GROUPS.items():
    for subset, in_subset in SUBSETS.items():
        keep = in_group & in_subset
        y_sub = scores.loc[keep, 'label'].to_numpy()
        for name, col, cut in [(their_name, their_col, 0.5), ('Our model', 'probability', OUR_CUTOFF)]:
            score = scores.loc[keep, col].to_numpy()
            called = score >= cut
            tp = int((called & (y_sub == 1)).sum())
            both_classes = 0 < y_sub.sum() < len(y_sub)
            rows.append({'organisms': group, 'subset': subset, 'proteins': scores.loc[keep, 'protein_id'].nunique(),
                         'sites': int(keep.sum()), 'positive': int(y_sub.sum()), 'model': name,
                         'AUROC': roc_auc_score(y_sub, score) if both_classes else np.nan,
                         'AUPRC': average_precision_score(y_sub, score) if both_classes else np.nan,
                         'MCC': matthews_corrcoef(y_sub, called) if both_classes else np.nan,
                         'precision': tp / max(called.sum(), 1), 'recall': tp / max(y_sub.sum(), 1)})
filtered = pd.DataFrame(rows)
filtered.to_csv(f'{OUT_DIR}/test_metrics_filtered.csv', index=False)

for group, (in_group, _, _) in GROUPS.items():
    ids = scores.loc[in_group, 'protein_id'].unique()
    print(f'{group}: {len(ids)} test proteins   same protein as an Atlas 4.0 protein: {int(flags.loc[ids, "same_protein"].sum())}   '
          f'homolog >= 30%: {int(flags.loc[ids, "homolog_30"].sum())}')
print()
print(filtered.to_string(index=False, float_format='{:.4f}'.format))

# %%
import os
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))

import subprocess
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

# ---------------------------------------------------------------------------
# Overlap of our test proteins with O-GlcNAcPRED-DL's own development data
#   Replaces the Atlas 4.0 superset of cell 4 (whose files are kept unchanged). The tool's
#   download page (oglcnac.org/pred_dl/download, "Positive dataset and negative dataset used for
#   model development") offers one file per species and class: 29-residue windows centred on an
#   S/T, padded with X at the protein ends (as by their seq_cut/cut.py), without protein IDs.
#   Positives are listed 5 times each; unique positives (human 2,990, mouse 1,434) equal the
#   paper's counts. The files are identical to the Wayback Machine captures of 2023-10-03/04.
#   Proteins behind the windows: every protein of three sequence sets (our Atlas 5.0 proteins,
#   all splits; the Atlas 4.0 sequences of NB06; the LM-OGlcNAc-Site snapshot sequences of NB06,
#   which include entries deleted from UniProt) that contains a window, positive or negative, at
#   an S/T (exact match after padding the sequence with 14 X at each end). Each model's seen
#   proteins come from its own species' files.
#   Overlap, each removing whole test proteins (same criteria as cell 4), per model:
#     (a) homology: identity >= 30% to a seen protein, identity = identical residues / length of
#         the shorter protein, exhaustive MMseqs2 alignment (fake prefilter, --alignment-mode 3,
#         E <= 0.001); (a) also removes every (b) protein;
#     (b) same protein: exact sequence of a seen protein, or a shared accession (isoform suffix
#         ignored).
#   Organism-matched rows use the flags of the model applied to each protein (mouse model for
#   mouse proteins, human model for the others). New files only (names ending in _server).
# ---------------------------------------------------------------------------
OUT_DIR = 'data_processed/other_tools/O-GlcNAcPRED-DL'
SERVER = f'{OUT_DIR}/training_data_server'
os.makedirs(SERVER, exist_ok=True)
FILES = {(species, label): f'{species}_{label}.txt' for species in ('human', 'mouse') for label in ('positive', 'negative')}
for name in FILES.values():   # curl: the server refuses Python's default client (as in NB06)
    if not os.path.exists(f'{SERVER}/{name}'):
        subprocess.run(['curl', '-sSfL', '-o', f'{SERVER}/{name}', f'https://oglcnac.org/static/fasta/{name}'], check=True)
windows = {key: {l.strip() for l in open(f'{SERVER}/{name}') if l.strip()} for key, name in FILES.items()}
for (species, label), w in windows.items():
    assert {len(x) for x in w} == {29} and {x[14] for x in w} <= {'S', 'T'}
    print(f'{species} {label}: {len(w):,} unique windows')
assert len(windows['human', 'positive']) == 2990 and len(windows['mouse', 'positive']) == 1434

# Candidate proteins: our Atlas 5.0 proteins, Atlas 4.0 and the LM-OGlcNAc-Site snapshot (NB06)
def read_fasta(path, key):
    seqs, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith('>'):
            name = key(line[1:])
            seqs[name] = ''
        elif name:
            seqs[name] += line
    return seqs

proteins = pd.read_parquet('data_processed/atlas5_proteins.parquet')
test = proteins[proteins['split'] == 'test'].reset_index(drop=True)
base = lambda acc: acc.split('-')[0]
ids_of_seq = {}   # sequence -> accessions (isoform suffix removed) under which it is known
def add(seq, ids):
    ids_of_seq.setdefault(seq, set()).update(base(i) for i in ids)
for seq, merged in zip(proteins['sequence'], proteins['merged_ids']):
    add(seq, merged.split('; '))
for header, seq in read_fasta('data_processed/other_tools/Atlas4.0/atlas4_proteins.fasta', key=lambda h: h).items():
    add(seq, header.split('|')[1:3])   # >source|accession|retrieved entry
for acc, seq in read_fasta('data_processed/other_tools/LM-OGlcNAc-Site/training_data_git/snapshot_proteins.fasta',
                           key=lambda h: h.split()[0]).items():
    add(seq, [acc])
print(f'\nCandidate sequences: {len(ids_of_seq):,}')

# Windows of every candidate sequence at each S/T, compared with the training windows
all_windows = set().union(*windows.values())
found = {}   # training window -> candidate sequences containing it
for seq in tqdm(ids_of_seq, desc='Matching windows'):
    padded = 'X' * 14 + seq + 'X' * 14
    for i, aa in enumerate(seq):
        if aa in 'ST' and padded[i:i + 29] in all_windows:
            found.setdefault(padded[i:i + 29], set()).add(seq)
seen = {}
for species in ('human', 'mouse'):
    seen[species] = set()
    for label in ('positive', 'negative'):
        w = windows[species, label]
        matched = [x for x in w if x in found]
        seen[species] |= set().union(*(found[x] for x in matched))
        print(f'{species} {label}: windows found in a candidate sequence {len(matched):,} of {len(w):,} '
              f'({len(matched) / len(w):.1%})')
    print(f'{species} model: seen sequences {len(seen[species]):,}')

# MMseqs2: every test protein against every seen sequence (as in cell 4 / NB06)
seen_list = sorted(seen['human'] | seen['mouse'])
seen_id = {s: f'seen{k}' for k, s in enumerate(seen_list)}
with open(f'{SERVER}/seen_proteins_server.fasta', 'w') as f:   # header: id|species|accessions
    for s in seen_list:
        species = '+'.join(sp for sp in ('human', 'mouse') if s in seen[sp])
        f.write(f'>{seen_id[s]}|{species}|{";".join(sorted(ids_of_seq[s]))}\n{s}\n')

MMSEQS = '/content/mmseqs/bin/mmseqs'
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

ALN_DIR = '/content/server_overlap'
os.makedirs(ALN_DIR, exist_ok=True)
ALN_FILE = os.path.abspath(f'{SERVER}/test_vs_seen_alignments_server.tsv')   # MMseqs2 runs in ALN_DIR
if not os.path.exists(ALN_FILE):
    if not os.path.exists(MMSEQS):
        subprocess.run('wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -O /content/mmseqs.tar.gz '
                       '&& tar xzf /content/mmseqs.tar.gz -C /content', shell=True, check=True)
    print('MMseqs2 version:', subprocess.run([MMSEQS, 'version'], capture_output=True, text=True).stdout.strip())
    write_fasta(f'{ALN_DIR}/test.fasta', test['protein_id'], test['sequence'])
    write_fasta(f'{ALN_DIR}/seen.fasta', [seen_id[s] for s in seen_list], seen_list)
    run_streamed([MMSEQS, 'createdb', f'{ALN_DIR}/test.fasta', f'{ALN_DIR}/qdb'], cwd=ALN_DIR)
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

# Per test protein and model: highest identity to a seen protein, and both overlap flags
species_of_id = {seen_id[s]: s for s in seen_list}
overlap = test[['protein_id', 'organism', 'merged_ids']].copy()
test_ids = [{base(a) for a in ids.split('; ')} for ids in test['merged_ids']]
for species in ('human', 'mouse'):
    sub = aln[aln['target'].map(lambda t: species_of_id[t] in seen[species])]
    overlap[f'max_identity_{species}'] = overlap['protein_id'].map(sub.groupby('query')['id_shortest'].max()).fillna(0.0)
    seen_accessions = set().union(*(ids_of_seq[s] for s in seen[species]))
    overlap[f'same_protein_{species}'] = (test['sequence'].isin(seen[species]).to_numpy()
                                          | np.array([bool(ids & seen_accessions) for ids in test_ids]))
    overlap[f'homolog_30_{species}'] = (overlap[f'max_identity_{species}'] >= 0.30) | overlap[f'same_protein_{species}']
    assert (overlap.loc[test['sequence'].isin(seen[species]).to_numpy(), f'max_identity_{species}'] >= 0.99).all()
overlap.to_csv(f'{SERVER}/test_protein_overlap_server.csv', index=False)

# Metrics: the same comparisons as cell 4
scores = pd.read_parquet(f'{OUT_DIR}/test_scores.parquet')
OUR_CUTOFF = torch.load('data_processed/final_model/final_model.pt', map_location='cpu')['cutoff']['probability']
ours = pd.read_parquet('data_processed/final_model/test_scores.parquet')[['protein_id', 'position', 'probability']]
scores = scores.merge(ours, on=['protein_id', 'position'], how='left', validate='one_to_one')
assert scores['probability'].notna().all()
flags = overlap.set_index('protein_id')
def flag(kind, species):
    return scores['protein_id'].map(flags[f'{kind}_{species}']).to_numpy(bool)
model_species = np.where(scores['matched_model'] == 'mouse', 'mouse', 'human')
matched_flag = lambda kind: np.where(model_species == 'mouse', flag(kind, 'mouse'), flag(kind, 'human'))
GROUPS = {'all organisms': (np.ones(len(scores), bool), 'score_matched', 'O-GlcNAcPRED-DL (matched)', matched_flag),
          'Homo sapiens': ((scores['organism'] == 'Homo sapiens').to_numpy(), 'score_human', 'O-GlcNAcPRED-DL human',
                           lambda kind: flag(kind, 'human')),
          'Mus musculus': ((scores['organism'] == 'Mus musculus').to_numpy(), 'score_mouse', 'O-GlcNAcPRED-DL mouse',
                           lambda kind: flag(kind, 'mouse'))}

rows = []
for group, (in_group, their_col, their_name, flags_of) in GROUPS.items():
    subsets = {'all test proteins': np.ones(len(scores), bool),
               '(b) same protein removed': ~flags_of('same_protein'),
               '(a) homologs >= 30% removed': ~flags_of('homolog_30')}
    for subset, in_subset in subsets.items():
        keep = in_group & in_subset
        y_sub = scores.loc[keep, 'label'].to_numpy()
        both_classes = 0 < y_sub.sum() < len(y_sub)
        for name, col, cut in [(their_name, their_col, 0.5), ('Our model', 'probability', OUR_CUTOFF)]:
            score = scores.loc[keep, col].to_numpy()
            called = score >= cut
            tp = int((called & (y_sub == 1)).sum())
            rows.append({'organisms': group, 'subset': subset, 'proteins': scores.loc[keep, 'protein_id'].nunique(),
                         'sites': int(keep.sum()), 'positive': int(y_sub.sum()), 'model': name,
                         'AUROC': roc_auc_score(y_sub, score) if both_classes else np.nan,
                         'AUPRC': average_precision_score(y_sub, score) if both_classes else np.nan,
                         'MCC': matthews_corrcoef(y_sub, called) if both_classes else np.nan,
                         'precision': tp / max(called.sum(), 1), 'recall': tp / max(y_sub.sum(), 1)})
filtered = pd.DataFrame(rows)
filtered.to_csv(f'{OUT_DIR}/test_metrics_filtered_server.csv', index=False)

atlas4 = pd.read_csv('data_processed/other_tools/Atlas4.0/test_protein_overlap.csv').set_index('protein_id')
print()
for group, (in_group, _, _, flags_of) in GROUPS.items():
    ids = scores.loc[in_group, 'protein_id'].unique()
    proteins_flagged = lambda kind: scores.loc[in_group & flags_of(kind), 'protein_id'].nunique()
    print(f'{group}: {len(ids)} test proteins   same protein: {proteins_flagged("same_protein")} '
          f'(Atlas 4.0: {int(atlas4.loc[ids, "same_protein"].sum())})   homolog >= 30%: '
          f'{proteins_flagged("homolog_30")} (Atlas 4.0: {int(atlas4.loc[ids, "homolog_30"].sum())})')
print()
print(filtered.to_string(index=False, float_format='{:.4f}'.format))

# %%
import ast, time
from Bio import SeqIO

# ---------------------------------------------------------------------------
# External validation sets: every S/T of every external protein (NB01), one run per study
#   Each study's proteins are submitted as one file, as a lab would submit its own proteins. The
#   scripts fit their feature scaler on the input file, so a protein reported by several studies
#   gets one score per study (NB12 averages them for the pooled set).
#   Model: as for the main test-set result (organism-matched), the mouse model for the mouse study
#   and the human model for the other studies (human and Drosophila).
#   Windows: their windowing code (seq_cut/cut.py), as for the test set (cell 2). Their feature
#   encoding accepts the 20 standard amino acids and X (both models) and U (human model only);
#   windows with any other letter cannot be scored and are counted (score left empty).
#   Each study is saved when scored and skipped on a rerun. Memory: ~17.5 GB per 93,000 windows
#   (test set), so the largest study needs a high-RAM runtime.
#   Timing: wall-clock time of each study's run, saved as results/compute/NB07_external_{study}.json
#   (compute-records cell) together with the study's scores; a skipped study keeps its record.
#   If scores from an earlier run exist, the new scores are compared with them before they are replaced.
# ---------------------------------------------------------------------------
cut_src = open(f'{REPO}/seq_cut/cut.py', encoding='utf-8').read()
functions = [ast.get_source_segment(cut_src, n) for n in ast.parse(cut_src).body
             if isinstance(n, ast.FunctionDef) and n.name in ('strcat', 'na_se')]
cut_functions = patch('\n\n'.join(functions), {'"NUP62_cut.fasta"': 'CUT_FASTA'})
exec('from Bio import SeqIO\nfrom Bio.Seq import Seq\nfrom Bio.SeqRecord import SeqRecord\n' + cut_functions)

EXT_WORK = '/content/ogp_external'
EXT_RUNS = f'{OUT_DIR}/external_runs'
os.makedirs(EXT_WORK, exist_ok=True)
os.makedirs(EXT_RUNS, exist_ok=True)
MOUSE_STUDIES = {'Mouse brain 2025'}
ACCEPTED = {'human': set('ACDEFGHIKLMNPQRSTVWYXU'), 'mouse': set('ACDEFGHIKLMNPQRSTVWYX')}

external = pd.read_parquet('data_processed/external_proteins.parquet')
studies = sorted({s for listed in external['studies'] for s in listed})
assert MOUSE_STUDIES <= set(studies), studies
print(f'External proteins: {len(external):,}   studies: {", ".join(studies)}')

for study in studies:
    run_file = f'{EXT_RUNS}/{study.replace(" ", "_")}.parquet'
    if os.path.exists(run_file):
        print(f'{study}: already scored ({run_file})')
        continue
    species = 'mouse' if study in MOUSE_STUDIES else 'human'
    proteins_study = external[[study in listed for listed in external['studies']]]
    stem = f'{EXT_WORK}/{study.replace(" ", "_")}'
    proteins_fasta, CUT_FASTA, sites_fasta = f'{stem}_proteins.fasta', f'{stem}_cut.fasta', f'{stem}_sites.fasta'
    with open(proteins_fasta, 'w') as f:
        for sid, seq in zip(proteins_study['seq_id'], proteins_study['sequence']):
            f.write(f'>{sid}\n{seq}\n')
    print(f'\n{study}: proteins shorter than 29 residues (printed by na_se):')
    na_se(proteins_fasta)

    windows = {}
    for record in SeqIO.parse(CUT_FASTA, 'fasta'):
        sid, position = record.id.rsplit('|position=', 1)
        windows[(sid, int(position) + 1)] = str(record.seq)
    sites = pd.DataFrame([(sid, i + 1, aa, windows[(sid, i + 1)])
                          for sid, seq in zip(proteins_study['seq_id'], proteins_study['sequence'])
                          for i, aa in enumerate(seq) if aa in 'ST'],
                         columns=['seq_id', 'position', 'residue', 'window'])
    assert len(sites) == len(windows) and (sites['window'].str.len() == 29).all()
    scorable = sites['window'].map(lambda w: set(w) <= ACCEPTED[species]).to_numpy()
    with open(sites_fasta, 'w') as f:
        for k, w in enumerate(sites.loc[scorable, 'window']):
            f.write(f'>Seq{k + 1}\n{w}\n')

    t0 = time.perf_counter()
    print(f'{study}: {len(proteins_study):,} proteins, {len(sites):,} S/T windows, {species} model '
          f'(not scorable: {(~scorable).sum():,})', flush=True)
    score = predict(species, sites_fasta)
    seconds = time.perf_counter() - t0
    assert len(score) == scorable.sum()
    sites['score'] = np.nan
    sites.loc[scorable, 'score'] = score
    sites.insert(0, 'study', study)
    sites['model'] = species
    save_compute('NB07', f'external_{study.replace(" ", "_")}', seconds,
                 proteins=len(proteins_study), st_sites=int(scorable.sum()))
    sites.drop(columns='window').to_parquet(run_file, index=False)

ext_scores = pd.concat([pd.read_parquet(f'{EXT_RUNS}/{s.replace(" ", "_")}.parquet') for s in studies], ignore_index=True)
if os.path.exists(f'{OUT_DIR}/external_scores.parquet'):
    previous = pd.read_parquet(f'{OUT_DIR}/external_scores.parquet')
    assert previous[['study', 'seq_id', 'position']].equals(ext_scores[['study', 'seq_id', 'position']])
    print(f'Previous run vs this run: score max |diff| {(previous["score"] - ext_scores["score"]).abs().max():.1e}')
ext_scores.to_parquet(f'{OUT_DIR}/external_scores.parquet', index=False)
summary = ext_scores.groupby(['study', 'model']).agg(proteins=('seq_id', 'nunique'), sites=('score', 'size'),
                                                       not_scored=('score', lambda s: int(s.isna().sum())),
                                                       called=('score', lambda s: int((s >= 0.5).sum())))
print('\n' + summary.to_string())
print(f'\nSaved {len(ext_scores):,} rows (study, seq_id, position): {OUT_DIR}/external_scores.parquet')
