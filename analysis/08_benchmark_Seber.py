# %%
import os
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))

import hashlib, json, subprocess, sys
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Seber RNN models, run with the authors' code and trained weights
#   Repository: github.com/PedroSeber/O-GlcNAcylation_Prediction (MIT). Two models, each run
#   with the repository version released with it; no changes to their code.
#     - RNN-[600,75]: Seber 2024, arXiv:2402.17131. Commit 8d97344 (2025-09-17, last commit).
#       Its Predict.py loads RNN-[600,75]_20-window_dict.pt when the file is present.
#     - RNN-225: Seber & Braatz 2024, Comput Chem Eng 189:108818. Commit b5629d5
#       (2023-11-27, the repository as it stood until 2025). Its Predict.py loads
#       RNN-225_20-window_dict.pt. The 2025 Predict.py cannot load this file: its model class
#       names the LSTM weights lstm.0.* and the file stores them as lstm.*.
#   Both versions encode the same +/-20 windows (get_nearby_AA and OHE_for_LSTM are identical).
#   The 2025 model class places the LSTM layers on the GPU (.cuda()), so a GPU runtime is needed.
# ---------------------------------------------------------------------------
assert torch.cuda.is_available(), 'the 2025 model class places its LSTM layers on the GPU'

REPO_URL = 'https://github.com/PedroSeber/O-GlcNAcylation_Prediction'
MODELS = {
    'RNN-[600,75]': {'commit': '8d97344c70ab08be008c6bfd9aa507c42d6c279d', 'dir': '/content/seber_2025',
                     'file': 'RNN-[600,75]_20-window_dict.pt',
                     'sha256': '811a962372499e50553d5ec3c3d95e61b38baf7387701dcd6b0cb225f73b8eab',
                     'reference': 'Seber 2024, arXiv:2402.17131'},
    'RNN-225': {'commit': 'b5629d55f51ba9bbd4728fcec944baf61c837162', 'dir': '/content/seber_2023',
                'file': 'RNN-225_20-window_dict.pt',
                'sha256': 'f9eaadb43c6e48c9d1579cd5f0f62fd6f7b6af754d6bdf376695e66556f3fb44',
                'reference': 'Seber & Braatz 2024, Comput Chem Eng 189:108818'},
}
OUT_DIR = 'data_processed/other_tools/Seber'
os.makedirs(OUT_DIR, exist_ok=True)

def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

for name, m in MODELS.items():
    if not os.path.exists(m['dir']):
        subprocess.run(f'git clone -q {REPO_URL} {m["dir"]}', shell=True, check=True)
    subprocess.run(f'git -C {m["dir"]} checkout -q {m["commit"]}', shell=True, check=True)
    head = subprocess.run(f'git -C {m["dir"]} rev-parse HEAD', shell=True, check=True,
                          capture_output=True, text=True).stdout.strip()
    assert head == m['commit']
    assert sha256(f'{m["dir"]}/{m["file"]}') == m['sha256'], f'{name}: model file differs'

# The 2023 version has only RNN-225; the 2025 version has both files and loads RNN-[600,75]
assert not os.path.exists(f'{MODELS["RNN-225"]["dir"]}/{MODELS["RNN-[600,75]"]["file"]}')

def predict_one(name, window):
    """Run the model's Predict.py on a single window and return its printed line."""
    result = subprocess.run([sys.executable, 'Predict.py', window], cwd=MODELS[name]['dir'],
                            check=True, capture_output=True, text=True)
    return result.stdout.strip().splitlines()[-1]

# Loading check: one window from their training data (first protein, first site, +/-20)
train = pd.read_csv(f'{MODELS["RNN-[600,75]"]["dir"]}/OVSlab_allSpecies_O-GlcNAcome_PS.csv', nrows=1)
seq, site = train.at[0, 'sequence'], train.at[0, 'oglcnac sites'].split(';')[0]
i = int(site[1:]) - 1
assert seq[i] == site[0]
window = seq[i - 20:i + 21]
print(f'Check window: {train.at[0, "UniprotKB ID"]} {site}  {window}')
for name in MODELS:
    print(f'{name:>13}: {predict_one(name, window)}')

with open(f'{OUT_DIR}/provenance.json', 'w') as f:
    json.dump({'repository': REPO_URL, 'models': MODELS, 'torch': torch.__version__,
               'gpu': torch.cuda.get_device_name()}, f, indent=1)

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
import importlib.util, shutil, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

# ---------------------------------------------------------------------------
# Installation check: their own test set
#   Both papers report results on the same split of their data (Seber & Braatz 2024,
#   Section 2.1; Seber 2024, Methods): the "Wulff-Fuentes et al. (2021) - Modified" dataset,
#   558,168 unique S/T sites, 13,637 positive (2.44%), with 20% held out for testing by
#   train_test_split(test_size=0.2, random_state=123) in ANN_train.py.
#     1. Their one_hot_encode_csv.py (commit 8d97344, run unchanged with window 5, as in their
#        pipeline) rebuilds the data from OVSlab_allSpecies_O-GlcNAcome_PS.csv: one-hot
#        windows of +/-20 (OH_LSTM_data_v5_20-window.npy) and labels (OH_data_v5_5-window.csv).
#     2. The split depends only on the number of rows, so it is taken on the row indices.
#     3. Predict.py reads window strings, so each one-hot test window is written back as a
#        41-letter string, with all-zero rows as X; their encoder maps X to an all-zero row,
#        so Predict.py receives exactly the same input (checked below).
#     4. Each model scores the windows with its own Predict.py; metrics at the threshold of
#        each paper's Table 3 (0.60 for both) are compared with the published values.
# ---------------------------------------------------------------------------
WORK = '/content/seber_data'
os.makedirs(WORK, exist_ok=True)
REPO_2025 = MODELS['RNN-[600,75]']['dir']
PUBLISHED = {   # Table 3 of each paper: threshold, recall, precision, F1, MCC (%)
    'RNN-225': {'threshold': 0.60, 'Recall': 35.47, 'Precision': 36.90, 'F1': 36.17, 'MCC': 34.57},
    'RNN-[600,75]': {'threshold': 0.60, 'Recall': 33.11, 'Precision': 47.10, 'F1': 38.88, 'MCC': 38.20},
}

def run_streamed(cmd, cwd):
    """Run a command and stream its output (including carriage-return progress lines)."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for ch in iter(lambda: proc.stdout.read(1), ''):
        sys.stdout.write(ch)
        if ch in '\r\n':
            sys.stdout.flush()
    assert proc.wait() == 0, f'{cmd} failed'

# 1. Their data
DATA_CSV = 'OVSlab_allSpecies_O-GlcNAcome_PS.csv'
X_FILE, Y_FILE = f'{WORK}/OH_LSTM_data_v5_20-window.npy', f'{WORK}/OH_data_v5_5-window.csv'
if not (os.path.exists(X_FILE) and os.path.exists(Y_FILE)):
    shutil.copy(f'{REPO_2025}/{DATA_CSV}', WORK)
    t0 = time.time()
    run_streamed([sys.executable, '-u', f'{REPO_2025}/one_hot_encode_csv.py', DATA_CSV, '-ws', '5'], cwd=WORK)
    print(f'\nData rebuilt in {(time.time() - t0) / 60:.1f} min')
X = np.load(X_FILE, mmap_mode='r')
y = pd.read_csv(Y_FILE).squeeze().to_numpy()
assert X.shape == (558168, 41, 20) and len(y) == len(X) and y.sum() == 13637

# 2. Their test split
_, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=123)
X_test, y_test = np.asarray(X[test_idx]), y[test_idx]
print(f'Test set: {len(y_test):,} sites, {y_test.sum():,} positive ({100 * y_test.mean():.2f}%)')

# 3. One-hot windows back to strings; LabelBinarizer orders its columns alphabetically
spec = importlib.util.spec_from_file_location('seber_predict_2025', f'{REPO_2025}/Predict.py')
predict_2025 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predict_2025)
ALPHABET = np.array(list('ACDEFGHIKLMNPQRSTVWY'))
assert (X_test.sum(axis=2) <= 1).all()
letters = np.where(X_test.sum(axis=2) == 1, ALPHABET[X_test.argmax(axis=2)], 'X')
windows = [''.join(row) for row in letters]
assert np.array_equal(predict_2025.OHE_for_LSTM(np.array(windows), 20), X_test)
print('\nDecoded windows re-encode to the original one-hot test windows')

# 4. Scores and metrics
rows = []
for name, m in MODELS.items():
    key = 'rnn225' if name == 'RNN-225' else 'rnn600_75'
    in_csv = f'{WORK}/check_{key}.csv'
    pd.DataFrame({'Sequences': windows}).to_csv(in_csv, index=False)
    t0 = time.time()
    run_streamed([sys.executable, '-u', 'Predict.py', in_csv], cwd=m['dir'])
    out = pd.read_csv(f'{WORK}/check_{key}_predictions.csv')
    assert out['Sequence'].tolist() == windows
    score = out['Predicted O-GlcNAcylation Chance'].to_numpy()
    pd.DataFrame({'site_index': test_idx, 'label': y_test, 'score': score}).to_parquet(
        f'{OUT_DIR}/installation_check_scores_{key}.parquet', index=False)
    called = score >= PUBLISHED[name]['threshold']
    ours = {'Recall': 100 * recall_score(y_test, called), 'Precision': 100 * precision_score(y_test, called),
            'F1': 100 * f1_score(y_test, called), 'MCC': 100 * matthews_corrcoef(y_test, called)}
    rows += [{'model': name, 'source': 'published (Table 3)', **PUBLISHED[name]},
             {'model': name, 'source': 'this run', 'threshold': PUBLISHED[name]['threshold'], **ours}]
    print(f'\n{name}: scored in {(time.time() - t0) / 60:.1f} min')

check = pd.DataFrame(rows)
check.to_csv(f'{OUT_DIR}/installation_check.csv', index=False)
print()
print(check.to_string(index=False, float_format='{:.2f}'.format))

# %%
import ast
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------------------
# Our test set: +/-20 windows from their windowing function
#   get_nearby_AA() is taken unchanged from their one_hot_encode_csv.py (identical in both
#   commits) and called as in their pipeline, with the 0-based index of the S/T and window 20.
#   Sites within 20 residues of the N-terminus get a window that starts at residue 1 (target
#   not centred); sites near the C-terminus get a shorter window (target at index 20); both
#   are zero-padded on the right by their encoder, as in their training data.
#   Scored sites: every S/T of the test proteins except masked residues (the same sites as
#   the embeddings notebook), in protein and position order.
#   Cut-off for yes/no calls: 0.60, the best-F1 threshold reported for both models on their
#   own test set (Table 3 of each paper).
#   Timing: wall-clock time of each model's Predict.py run (a new Python process: model loading,
#   one-hot encoding and scoring), saved as results/compute/NB08_{set}_{model}.json (compute-records
#   cell). If scores from an earlier run exist, the new scores are compared with them before they
#   are replaced.
# ---------------------------------------------------------------------------
CUTOFF = 0.60

ohe_src = open(f'{REPO_2025}/one_hot_encode_csv.py', encoding='utf-8').read()
node = next(n for n in ast.parse(ohe_src).body if isinstance(n, ast.FunctionDef) and n.name == 'get_nearby_AA')
exec(ast.get_source_segment(ohe_src, node))

proteins = pd.read_parquet('data_processed/atlas5_proteins.parquet')
test = proteins[proteins['split'] == 'test'].reset_index(drop=True)

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
                     'label': int(i + 1 in positive), 'window': get_nearby_AA(seq, i, 20)})
sites = pd.DataFrame(rows)

# Same test sites as the embeddings notebook
reference = pd.read_parquet('data_processed/embeddings/sites.parquet', columns=['split', 'label'])
reference = reference[(reference['split'] == 'test') & (reference['label'] != 'masked')]
assert len(sites) == len(reference) and sites['label'].sum() == (reference['label'] == 'positive').sum()

# Window checks: at most 41 residues; target at index 20 except near the N-terminus
assert (sites['window'].str.len() <= 41).all()
off_centre = (sites['position'] <= 20).to_numpy()
assert (sites['window'].str[20][~off_centre] == sites['residue'][~off_centre]).all()
assert all(w[p - 1] == r for w, p, r in zip(sites['window'][off_centre], sites['position'][off_centre],
                                           sites['residue'][off_centre]))
short = (sites['window'].str.len() < 41).to_numpy()
print(f'Test proteins: {len(test):,}   S/T sites scored: {len(sites):,} ({sites["label"].sum():,} positive)   '
      f'masked S/T excluded: {n_masked:,}')
print(f'Windows with the target off centre (position <= 20): {off_centre.sum():,}   '
      f'windows shorter than 41 residues: {short.sum():,}')
extra = set(''.join(sites['window'])) - set('ACDEFGHIKLMNPQRSTVWY')
print(f'Letters outside the 20 standard amino acids (encoded as zero rows): {sorted(extra)}')

# Scores: each model scores all test windows with its own Predict.py
for name, m in MODELS.items():
    key = 'rnn225' if name == 'RNN-225' else 'rnn600_75'
    in_csv = f'{WORK}/test_{key}.csv'
    pd.DataFrame({'Sequences': sites['window']}).to_csv(in_csv, index=False)
    t0 = time.perf_counter()
    run_streamed([sys.executable, '-u', 'Predict.py', in_csv], cwd=m['dir'])
    seconds = time.perf_counter() - t0
    out = pd.read_csv(f'{WORK}/test_{key}_predictions.csv')
    assert out['Sequence'].tolist() == sites['window'].tolist()
    sites[f'score_{key}'] = out['Predicted O-GlcNAcylation Chance'].to_numpy()
    print()
    save_compute('NB08', f'test_{key}', seconds, proteins=len(test), st_sites=len(sites))

if os.path.exists(f'{OUT_DIR}/test_scores.parquet'):
    previous = pd.read_parquet(f'{OUT_DIR}/test_scores.parquet')
    assert previous[['protein_id', 'position']].equals(sites[['protein_id', 'position']])
    print('Previous run vs this run, max |diff|: ' + ', '.join(
        f'{c} {(previous[c] - sites[c]).abs().max():.1e}' for c in ('score_rnn600_75', 'score_rnn225')))
sites.drop(columns='window').to_parquet(f'{OUT_DIR}/test_scores.parquet', index=False)

y_true = sites['label'].to_numpy()
summary = []
for name in MODELS:
    key = 'rnn225' if name == 'RNN-225' else 'rnn600_75'
    score = sites[f'score_{key}'].to_numpy()
    called = score >= CUTOFF
    tp = int((called & (y_true == 1)).sum())
    summary.append({'model': name, 'AUROC': roc_auc_score(y_true, score),
                    'AUPRC': average_precision_score(y_true, score),
                    'MCC': matthews_corrcoef(y_true, called), 'precision': tp / max(called.sum(), 1),
                    'recall': tp / y_true.sum(), 'called positive': int(called.sum())})
summary = pd.DataFrame(summary)
summary.to_csv(f'{OUT_DIR}/test_metrics.csv', index=False)
print(f'\nAll test sites (positive fraction {y_true.mean():.4f}); yes/no calls at score >= {CUTOFF}:')
print(summary.to_string(index=False, float_format='{:.4f}'.format))

# %%
# ---------------------------------------------------------------------------
# Overlap of our test proteins with their training data (decision 5b)
#   Their training set: every protein of OVSlab_allSpecies_O-GlcNAcome_PS.csv (4,541 entries).
#   Their test split is by window, not by protein, so every protein contributed training
#   windows. Two definitions of overlap, each removing whole test proteins:
#     (a) homology, as in our own split: identity >= 30% to any of their proteins, with
#         identity = identical residues / length of the shorter protein, from an exhaustive
#         MMseqs2 alignment of every test protein against every one of their proteins
#         (fake prefilter, --alignment-mode 3, E <= 0.001; E-values relative to their 4,541
#         proteins);
#     (b) same protein: the UniProt accession (isoform suffix ignored) or the exact sequence
#         is in their file.
#   Metrics on the remaining sites for both RNNs (cut-off 0.60) and our final model
#   (calibrated probability >= its validation MCC cut-off).
# ---------------------------------------------------------------------------
MMSEQS = '/content/mmseqs/bin/mmseqs'
if not os.path.exists(MMSEQS):
    subprocess.run('wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -O /content/mmseqs.tar.gz '
                   '&& tar xzf /content/mmseqs.tar.gz -C /content', shell=True, check=True)
mmseqs_version = subprocess.run([MMSEQS, 'version'], capture_output=True, text=True).stdout.strip()
print('MMseqs2 version:', mmseqs_version)

theirs = pd.read_csv(f'{REPO_2025}/{DATA_CSV}')
assert theirs['UniprotKB ID'].is_unique
ALN_DIR = '/content/seber_overlap'
os.makedirs(ALN_DIR, exist_ok=True)
ALN_FILE = os.path.abspath(f'{OUT_DIR}/test_vs_training_alignments.tsv')   # MMseqs2 runs in ALN_DIR

def write_fasta(path, ids, seqs):
    with open(path, 'w') as f:
        for i, s in zip(ids, seqs):
            f.write(f'>{i}\n{s}\n')

if not os.path.exists(ALN_FILE):
    write_fasta(f'{ALN_DIR}/test.fasta', test['protein_id'], test['sequence'])
    write_fasta(f'{ALN_DIR}/theirs.fasta', theirs['UniprotKB ID'], theirs['sequence'])
    run_streamed([MMSEQS, 'createdb', f'{ALN_DIR}/test.fasta', f'{ALN_DIR}/qdb'], cwd=ALN_DIR)
    run_streamed([MMSEQS, 'createdb', f'{ALN_DIR}/theirs.fasta', f'{ALN_DIR}/tdb'], cwd=ALN_DIR)
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

# Per test protein: closest of their proteins, and both overlap flags
best = aln.sort_values('id_shortest', ascending=False).drop_duplicates('query').set_index('query')
overlap = test[['protein_id', 'organism', 'sequence']].copy()
overlap['max_identity'] = overlap['protein_id'].map(best['id_shortest']).fillna(0.0)
overlap['closest_training_protein'] = overlap['protein_id'].map(best['target'])
base = lambda ids: pd.Series(ids).astype(str).str.split('-').str[0]
overlap['same_protein'] = (base(overlap['protein_id']).isin(set(base(theirs['UniprotKB ID']))).to_numpy()
                           | overlap['sequence'].isin(set(theirs['sequence'])).to_numpy())
overlap['homolog_30'] = overlap['max_identity'] >= 0.30
assert (overlap.loc[overlap['sequence'].isin(set(theirs['sequence'])), 'max_identity'] >= 0.99).all()
overlap.drop(columns='sequence').to_csv(f'{OUT_DIR}/test_protein_overlap.csv', index=False)

# Our final model's test scores and cut-off (NB05)
ours = pd.read_parquet('data_processed/final_model/test_scores.parquet')[['protein_id', 'position', 'probability']]
OUR_CUTOFF = torch.load('data_processed/final_model/final_model.pt', map_location='cpu')['cutoff']['probability']
sites = sites.drop(columns='probability', errors='ignore').merge(ours, on=['protein_id', 'position'], how='left',
                                                                 validate='one_to_one')
assert sites['probability'].notna().all()

SCORES = {'RNN-[600,75]': ('score_rnn600_75', CUTOFF), 'RNN-225': ('score_rnn225', CUTOFF),
          'Our model': ('probability', OUR_CUTOFF)}
SUBSETS = {'all test proteins': np.ones(len(sites), bool),
           '(b) same protein removed': ~sites['protein_id'].map(overlap.set_index('protein_id')['same_protein']).to_numpy(),
           '(a) homologs >= 30% removed': ~sites['protein_id'].map(overlap.set_index('protein_id')['homolog_30']).to_numpy()}

rows = []
for subset, keep in SUBSETS.items():
    y_sub = sites.loc[keep, 'label'].to_numpy()
    for name, (col, cut) in SCORES.items():
        score = sites.loc[keep, col].to_numpy()
        called = score >= cut
        tp = int((called & (y_sub == 1)).sum())
        rows.append({'subset': subset, 'proteins': sites.loc[keep, 'protein_id'].nunique(), 'sites': int(keep.sum()),
                     'positive': int(y_sub.sum()), 'model': name, 'cut-off': cut,
                     'AUROC': roc_auc_score(y_sub, score), 'AUPRC': average_precision_score(y_sub, score),
                     'MCC': matthews_corrcoef(y_sub, called), 'precision': tp / max(called.sum(), 1),
                     'recall': tp / y_sub.sum()})
filtered = pd.DataFrame(rows)
filtered.to_csv(f'{OUT_DIR}/test_metrics_filtered.csv', index=False)

print(f'\nTest proteins: {len(overlap)}   same protein in their data: {overlap["same_protein"].sum()}   '
      f'homolog >= 30%: {overlap["homolog_30"].sum()}')
print('Highest identity to their proteins (test proteins), quantiles 0/25/50/75/100%:',
      np.round(overlap['max_identity'].quantile([0, 0.25, 0.5, 0.75, 1]).to_numpy(), 2))
print()
print(filtered.to_string(index=False, float_format='{:.4f}'.format))

# %%
import io
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_curve

# ---------------------------------------------------------------------------
# Comparison with the Seber RNN models on our test set (figure number set with the manuscript)
#   Columns: all test proteins; test proteins in their training data removed (same protein);
#   test proteins with >= 30% identity to their training proteins removed. In each column all
#   models are scored on the same sites.
#   Top row: ROC curves (A-C). Bottom row: precision-recall curves (D-F); grey line: the
#   positive fraction of the subset (no skill). Markers: each model's yes/no cut-off (0.60 for
#   the RNNs, the validation MCC cut-off for our model).
#   Frontiers figure rules: 180 mm wide, text >= 8 pt and lines >= 2 pt at final size,
#   solid lines only, panels labelled (A)-(F) outside the plot area, no red or green, models
#   told apart by colour, marker shape and the labelled values; PDF and RGB TIFF at 300 dpi.
# ---------------------------------------------------------------------------
FIG_DIR = 'results/figures/seber_comparison'
os.makedirs(FIG_DIR, exist_ok=True)
GREY = '#8a8984'
STYLE = {'Our model': ('#1c5cab', 'o'), 'RNN-[600,75]': ('#c26a0e', 's'), 'RNN-225': ('#6b3fa0', '^')}
COLUMNS = [('all test proteins', 'All test proteins'),
           ('(b) same protein removed', 'Same proteins removed'),
           ('(a) homologs >= 30% removed', 'Homologs (≥ 30%) removed')]

def save_figure(fig, name, width_mm):
    """PDF (vector, for the LaTeX manuscript) and RGB TIFF at 300 dpi (production format)."""
    assert abs(fig.get_size_inches()[0] * 25.4 - width_mm) < 1e-6
    fig.savefig(f'{FIG_DIR}/{name}.pdf')
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=300)
    Image.open(buffer).convert('RGB').save(f'{FIG_DIR}/{name}.tif', dpi=(300, 300))

def value_block(ax, header, values, top):
    """Metric name and one coloured line per model (best first), right-aligned from the top position."""
    ax.text(0.97, top, header, ha='right', va='top', transform=ax.transAxes, fontweight='bold')
    for k, (name, value, colour) in enumerate(sorted(values, key=lambda t: -t[1])):
        ax.text(0.97, top - 0.09 * (k + 1), f'{name} {value:.3f}', color=colour, ha='right', va='top',
                transform=ax.transAxes)

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'lines.linewidth': 2,
                     'axes.linewidth': 2, 'xtick.major.width': 2, 'ytick.major.width': 2,
                     'axes.spines.top': False, 'axes.spines.right': False})

fig, axes = plt.subplots(2, 3, figsize=(180 / 25.4, 128 / 25.4), layout='constrained')
for c, (subset, title) in enumerate(COLUMNS):
    keep = SUBSETS[subset]
    y_sub = sites.loc[keep, 'label'].to_numpy()
    roc_ax, pr_ax = axes[0, c], axes[1, c]
    roc_ax.plot([0, 1], [0, 1], color=GREY)
    pr_ax.axhline(y_sub.mean(), color=GREY)
    roc_text, pr_text = [], []
    for name, (col, cut) in SCORES.items():
        colour, marker = STYLE[name]
        score = sites.loc[keep, col].to_numpy()
        called = score >= cut
        fpr_cut = (called & (y_sub == 0)).sum() / (y_sub == 0).sum()
        rec_cut = (called & (y_sub == 1)).sum() / y_sub.sum()
        prec_cut = (called & (y_sub == 1)).sum() / max(called.sum(), 1)
        fpr, tpr, _ = roc_curve(y_sub, score)
        precision, recall, _ = precision_recall_curve(y_sub, score)
        roc_ax.plot(fpr, tpr, color=colour)
        roc_ax.plot(fpr_cut, rec_cut, marker=marker, color=colour, markersize=6, markeredgecolor='white',
                    markeredgewidth=0.8, linestyle='', zorder=4)
        pr_ax.plot(recall, precision, color=colour)
        pr_ax.plot(rec_cut, prec_cut, marker=marker, color=colour, markersize=6, markeredgecolor='white',
                   markeredgewidth=0.8, linestyle='', zorder=4)
        roc_text.append((name, roc_auc_score(y_sub, score), colour))
        pr_text.append((name, average_precision_score(y_sub, score), colour))
    value_block(roc_ax, 'AUROC', roc_text, top=0.12 + 0.09 * len(roc_text))
    value_block(pr_ax, 'AUPRC', pr_text, top=0.95)
    n_prot = sites.loc[keep, 'protein_id'].nunique()
    roc_ax.set_title(f'({chr(65 + c)}) {title}\n{n_prot} proteins, {y_sub.sum():,} positive sites', loc='left')
    pr_ax.set_title(f'({chr(68 + c)})', loc='left')
    roc_ax.set_xlabel('False positive rate')
    pr_ax.set_xlabel('Recall')
    for ax in (roc_ax, pr_ax):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
    if c == 0:
        roc_ax.set_ylabel('True positive rate (recall)')
        pr_ax.set_ylabel('Precision')
    else:
        roc_ax.tick_params(labelleft=False)
        pr_ax.tick_params(labelleft=False)

handles = [Line2D([], [], color=STYLE[n][0], marker=STYLE[n][1], markersize=6, markeredgecolor='white',
                  markeredgewidth=0.8) for n in SCORES]
handles.append(Line2D([], [], color=GREY))
fig.legend(handles, [*SCORES, 'No skill'], loc='outside lower center', ncol=4, frameon=False)

save_figure(fig, 'figure_seber_comparison', 180)
plt.show()

# %%
import os
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

OUT_DIR = 'data_processed/other_tools/Seber'
CUTOFF = 0.60
sites = pd.read_parquet(f'{OUT_DIR}/test_scores.parquet')
ours = pd.read_parquet('data_processed/final_model/test_scores.parquet')[['protein_id', 'position', 'probability']]
OUR_CUTOFF = torch.load('data_processed/final_model/final_model.pt', map_location='cpu')['cutoff']['probability']
sites = sites.merge(ours, on=['protein_id', 'position'], how='left', validate='one_to_one')
overlap = pd.read_csv(f'{OUT_DIR}/test_protein_overlap.csv')
SCORES = {'RNN-[600,75]': ('score_rnn600_75', CUTOFF), 'RNN-225': ('score_rnn225', CUTOFF),
          'Our model': ('probability', OUR_CUTOFF)}
SUBSETS = {'all test proteins': np.ones(len(sites), bool),
           '(b) same protein removed': ~sites['protein_id'].map(overlap.set_index('protein_id')['same_protein']).to_numpy(),
           '(a) homologs >= 30% removed': ~sites['protein_id'].map(overlap.set_index('protein_id')['homolog_30']).to_numpy()}

# %%
import json, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# Mammalian test proteins only (both Seber RNNs were trained on mammalian proteins only:
# Seber & Braatz 2024, Section 2.1, "remove non-mammalian proteins")
#   Each organism in the test set is looked up in NCBI Taxonomy (E-utilities); it is mammalian
#   if its lineage contains Mammalia. The organism of a protein is that of its kept UniProt ID.
#   Same three subsets, models and cut-offs as above, restricted to mammalian proteins.
# ---------------------------------------------------------------------------
EUTILS = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'

def ncbi_lineage(name):
    """Scientific name -> (taxid, lineage) from NCBI Taxonomy."""
    q = urllib.parse.urlencode({'db': 'taxonomy', 'term': f'"{name}"[Scientific Name]', 'retmode': 'json'})
    with urllib.request.urlopen(f'{EUTILS}/esearch.fcgi?{q}', timeout=60) as r:
        ids = json.load(r)['esearchresult']['idlist']
    assert len(ids) == 1, f'{name}: {len(ids)} NCBI Taxonomy matches'
    time.sleep(0.4)   # NCBI limit without an API key: 3 requests per second
    with urllib.request.urlopen(f'{EUTILS}/efetch.fcgi?db=taxonomy&id={ids[0]}&retmode=xml', timeout=60) as r:
        lineage = ET.fromstring(r.read()).findtext('Taxon/Lineage')
    time.sleep(0.4)
    return int(ids[0]), lineage

organisms = sites.groupby('organism').agg(proteins=('protein_id', 'nunique'), sites=('label', 'size'),
                                          positive=('label', 'sum')).sort_values('sites', ascending=False)
taxa = {name: ncbi_lineage(name) for name in organisms.index}
organisms['taxid'] = [taxa[n][0] for n in organisms.index]
organisms['mammal'] = [' Mammalia;' in f' {taxa[n][1]};' for n in organisms.index]
organisms.to_csv(f'{OUT_DIR}/test_organisms_ncbi.csv')
print('Test-set organisms (NCBI Taxonomy):')
print(organisms.to_string())

mammal = sites['organism'].isin(organisms.index[organisms['mammal']]).to_numpy()
rows = []
for subset, keep in SUBSETS.items():
    keep_m = keep & mammal
    y_sub = sites.loc[keep_m, 'label'].to_numpy()
    for name, (col, cut) in SCORES.items():
        score = sites.loc[keep_m, col].to_numpy()
        called = score >= cut
        tp = int((called & (y_sub == 1)).sum())
        rows.append({'subset': subset, 'proteins': sites.loc[keep_m, 'protein_id'].nunique(),
                     'sites': int(keep_m.sum()), 'positive': int(y_sub.sum()), 'model': name, 'cut-off': cut,
                     'AUROC': roc_auc_score(y_sub, score), 'AUPRC': average_precision_score(y_sub, score),
                     'MCC': matthews_corrcoef(y_sub, called), 'precision': tp / max(called.sum(), 1),
                     'recall': tp / y_sub.sum()})
mammal_metrics = pd.DataFrame(rows)
mammal_metrics.to_csv(f'{OUT_DIR}/test_metrics_mammalian.csv', index=False)
print('\nMammalian test proteins only:')
print(mammal_metrics.to_string(index=False, float_format='{:.4f}'.format))

# %%
import ast, time

# ---------------------------------------------------------------------------
# External validation sets: every S/T of every external protein (NB01)
#   As for the test set (cell 3): +/-20 windows from their get_nearby_AA() (one_hot_encode_csv.py,
#   unchanged), each model scoring all windows with its own Predict.py. Letters outside the 20
#   standard amino acids are encoded as zero rows by their encoder, as for the test set. Needs only
#   cell 0 and the compute-records cell (the helpers of cells 2-3 are defined again here).
#   Timing: wall-clock time of each model's Predict.py run (a new Python process: model loading,
#   one-hot encoding and scoring), saved as results/compute/NB08_{set}_{model}.json (compute-records
#   cell). If scores from an earlier run exist, the new scores are compared with them before they
#   are replaced.
# ---------------------------------------------------------------------------
WORK = '/content/seber_data'
REPO_2025 = MODELS['RNN-[600,75]']['dir']
os.makedirs(WORK, exist_ok=True)

def run_streamed(cmd, cwd):
    """Run a command and stream its output (including carriage-return progress lines)."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for ch in iter(lambda: proc.stdout.read(1), ''):
        sys.stdout.write(ch)
        if ch in '\r\n':
            sys.stdout.flush()
    assert proc.wait() == 0, f'{cmd} failed'

ohe_src = open(f'{REPO_2025}/one_hot_encode_csv.py', encoding='utf-8').read()
node = next(n for n in ast.parse(ohe_src).body if isinstance(n, ast.FunctionDef) and n.name == 'get_nearby_AA')
exec(ast.get_source_segment(ohe_src, node))

external = pd.read_parquet('data_processed/external_proteins.parquet').sort_values('seq_id').reset_index(drop=True)
ext_sites = pd.DataFrame([(sid, i + 1, aa, get_nearby_AA(seq, i, 20))
                          for sid, seq in zip(external['seq_id'], external['sequence'])
                          for i, aa in enumerate(seq) if aa in 'ST'],
                         columns=['seq_id', 'position', 'residue', 'window'])
assert (ext_sites['window'].str.len() <= 41).all()
off_centre = (ext_sites['position'] <= 20).to_numpy()
assert (ext_sites['window'].str[20][~off_centre] == ext_sites['residue'][~off_centre]).all()
extra = set(''.join(ext_sites['window'])) - set('ACDEFGHIKLMNPQRSTVWY')
print(f'External proteins: {len(external):,}   S/T windows: {len(ext_sites):,}   target off centre: {off_centre.sum():,}   '
      f'letters outside the 20 standard amino acids (zero rows): {sorted(extra)}')

for name, m in MODELS.items():
    key = 'rnn225' if name == 'RNN-225' else 'rnn600_75'
    in_csv = f'{WORK}/external_{key}.csv'
    pd.DataFrame({'Sequences': ext_sites['window']}).to_csv(in_csv, index=False)
    t0 = time.perf_counter()
    run_streamed([sys.executable, '-u', 'Predict.py', in_csv], cwd=m['dir'])
    seconds = time.perf_counter() - t0
    out = pd.read_csv(f'{WORK}/external_{key}_predictions.csv')
    assert out['Sequence'].tolist() == ext_sites['window'].tolist()
    ext_sites[f'score_{key}'] = out['Predicted O-GlcNAcylation Chance'].to_numpy()
    print()
    save_compute('NB08', f'external_{key}', seconds, proteins=len(external), st_sites=len(ext_sites))

if os.path.exists(f'{OUT_DIR}/external_scores.parquet'):
    previous = pd.read_parquet(f'{OUT_DIR}/external_scores.parquet')
    assert previous[['seq_id', 'position']].equals(ext_sites[['seq_id', 'position']])
    print('Previous run vs this run, max |diff|: ' + ', '.join(
        f'{c} {(previous[c] - ext_sites[c]).abs().max():.1e}' for c in ('score_rnn600_75', 'score_rnn225')))
ext_sites.drop(columns='window').to_parquet(f'{OUT_DIR}/external_scores.parquet', index=False)
print(f'\nSaved {len(ext_sites):,} S/T scores: {OUT_DIR}/external_scores.parquet')
print(ext_sites[['score_rnn600_75', 'score_rnn225']].describe().loc[['mean', '50%', 'min', 'max']].to_string(float_format='{:.4f}'.format))
