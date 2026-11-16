# %%
import os
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))

# Set before TensorFlow is first imported: tf.keras resolves to Keras 2 (tf-keras), so their
# tf.keras.models.load_model() call runs as written; TensorFlow allocates GPU memory as needed,
# leaving room for the PyTorch Ankh model.
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import subprocess, sys
from importlib.metadata import version
# Versions (decided for NB10): transformers 4.57.6 is the last release with the TensorFlow T5
# encoder their ProtT5 script uses (TensorFlow models were removed in 5.0.0), and ankh 1.10.0
# requires transformers < 5. tf-keras provides Keras 2, which transformers' TensorFlow models and
# their .h5 model need (current TensorFlow ships Keras 3).
TF_MINOR = version('tensorflow').rsplit('.', 1)[0]
subprocess.run(f'pip install -q "transformers==4.57.6" "ankh==1.10.0" "tf-keras~={TF_MINOR}.0" gdown biopython',
               shell=True, check=True)

import hashlib, json
import gdown
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Stack-OglyPred-PLM (Pakhrin lab; Bioinformatics 2024, doi:10.1093/bioinformatics/btae643),
# nucleocytoplasmic (GlcNAc) model, run with the authors' code and trained model
#   Code: github.com/PakhrinLab/Stack-OglyPred-PLM, pinned to its last commit (f9934d2).
#   Model and their independent-test features: the "NucleoCytoplasmic_dbPTM" folder of their
#   public Google Drive (linked from the repository README), downloaded once to OUT_DIR.
#   The stacked model takes two inputs per S/T site, its Ankh-large vector (1,536 values) and its
#   ProtT5-XL-U50 vector (1,024 values), and outputs two probabilities; a site is called positive
#   when the second one exceeds 0.5 (their notebook "Nucleocytoplasmic GlcNAc Final Program").
# ---------------------------------------------------------------------------
REPO_URL = 'https://github.com/PakhrinLab/Stack-OglyPred-PLM'
REPO_COMMIT = 'f9934d21144ebe34a110c0668904972e532b947c'   # 2024-09-25, "Add files via upload"
REPO = '/content/Stack-OglyPred-PLM'
OUT_DIR = 'data_processed/other_tools/Stack-OglyPred-PLM'
THEIRS = f'{OUT_DIR}/their_files'
os.makedirs(THEIRS, exist_ok=True)

if not os.path.exists(REPO):
    subprocess.run(f'git clone -q {REPO_URL} {REPO}', shell=True, check=True)
subprocess.run(f'git -C {REPO} checkout -q {REPO_COMMIT}', shell=True, check=True)

# Files of their "NucleoCytoplasmic_dbPTM" Drive folder (Google Drive file IDs)
DRIVE_FILES = {
    'Model_Nucleus_and_Cytoplasm_Stacked_2687032Model.h5': '1U900WeaCFZwgPcMj2Hli8zK4kG-dH5-6',
    'Ankh_Feature_1062_Positive_Independent_Testing_Site_Intracellular_from_183_Proteins.txt': '11l8yUyw7Pwy44C0LJxI6rvfywvd1nrfD',
    'Ankh_Feature_27031_Negative_Independent_Testing_Site_Intracellular_from_183_Proteins.txt': '1SsbdRnInUt0odCBcZ-HWdHxJFkCFatDo',
    'Nucleocytoplasmic_Independent_testing_Positive_1062_with_183_Proteins.csv': '1ZB6NiMcb4ToIjNDTAIWP5z4iEfTuB157',
    'Nucleocytoplasmic_Independent_testing_Negative_27031_with_183_Proteins.csv': '1Fp2Rs-5X0HbxkcIvzjyv1lU5jMQ7EhFU',
}
for name, file_id in DRIVE_FILES.items():
    path = f'{THEIRS}/{name}'
    if not os.path.exists(path):
        gdown.download(id=file_id, output=path, quiet=False)

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

print('\nTheir files:')
checksums = {}
for name in DRIVE_FILES:
    path = f'{THEIRS}/{name}'
    checksums[name] = sha256(path)
    print(f'  {name}: {os.path.getsize(path) / 1e6:,.1f} MB   sha256 {checksums[name][:16]}...')

import torch, torch._dynamo   # PyTorch and its compiler (triton) before TensorFlow: loading them after TensorFlow can crash the kernel
import tensorflow as tf
import transformers, ankh
assert transformers.__version__ == '4.57.6'
print(f'\ntensorflow {tf.__version__}   transformers {transformers.__version__}   torch {torch.__version__}   '
      f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else "none"}')
with open(f'{OUT_DIR}/provenance.json', 'w') as f:
    json.dump({'repository': REPO_URL, 'commit': REPO_COMMIT, 'drive_files': DRIVE_FILES, 'sha256': checksums,
               'tensorflow': tf.__version__, 'transformers': transformers.__version__, 'torch': torch.__version__},
              f, indent=1)

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
import json as _json
import torch
from sklearn.metrics import confusion_matrix

# Full precision throughout NB10: TF32 matrix multiplication (enabled by default in TensorFlow on
# A100 GPUs) changes scores by up to ~1e-3 and flips 3 of their 28,093 test sites at the 0.5
# cut-off; with it disabled their published result is reproduced exactly. Their embedding scripts
# request no GPU (SLURM jobs with CPUs only), so their runs were most likely in full precision.
tf.config.experimental.enable_tensor_float_32_execution(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ---------------------------------------------------------------------------
# Installation check: their independent test set, scored from their own precomputed features
#   Their notebook "Nucleocytoplasmic GlcNAc Final Program.ipynb" is run as released, keeping only
#   the cells that load the independent-test features and the model and print the results
#   (cells 0, 1, 7, 9 and 11; cells 3 and 5 load the training features, which are not needed).
#   One change, tagged in the saved copy: their working directory (os.chdir to their cluster
#   path) points to the folder holding the downloaded files.
#   Expected (saved output of their notebook): 183 proteins, 1,062 positive / 27,031 negative
#   sites; MCC 0.2824; confusion matrix [[22320, 4711], [252, 810]].
# ---------------------------------------------------------------------------
os.makedirs(f'{OUT_DIR}/code', exist_ok=True)

def patch(source, changes):
    """Apply line changes {old: new}; each changed line is tagged with its original text, and each
    change must match at least once."""
    lines = source.split('\n')
    for old, new in changes.items():
        hits = [i for i, line in enumerate(lines) if old in line]
        assert hits, f'not found: {old}'
        for i in hits:
            original = lines[i].strip()
            lines[i] = f'{lines[i].replace(old, new)}  # [modified] original: {original}'
    return '\n'.join(lines)

nb = _json.load(open(f'{REPO}/Nucleocytoplasmic GlcNAc Final Program.ipynb', encoding='utf-8'))
CELLS = [0, 1, 7, 9, 11]
assert all(nb['cells'][i]['cell_type'] == 'code' for i in CELLS)
check_src = '\n\n'.join(''.join(nb['cells'][i]['source']) for i in CELLS)
check_src = patch(check_src, {'"/homes/t326h379/Ankh_nucleocytoplasmic_dbptm_2231"': 'THEIRS_DIR'})
with open(f'{OUT_DIR}/code/installation_check_from_their_notebook.py', 'w', encoding='utf-8') as f:
    f.write(check_src)

check = {'THEIRS_DIR': os.path.abspath(THEIRS)}
cwd = os.getcwd()
try:
    exec(check_src, check)
finally:
    os.chdir(cwd)

# The two feature files must list the same sites in the same order (their code assumes it)
for ankh_df, t5_df in [(check['df_positive'], check['df_positive_testing']),
                       (check['df_negative'], check['df_negative_testing'])]:
    assert (ankh_df['PID'].astype(str).values == t5_df['PID'].astype(str).values).all()
    assert (ankh_df['Position'].values == t5_df['Position'].values).all()

cm = confusion_matrix(check['y_independent'], check['y_pred'])
expected = np.array([[22320, 4711], [252, 810]])
print(f'\nConfusion matrix: this run {cm.tolist()}   published {expected.tolist()}')
assert (cm == expected).all(), 'installation check failed'
print('Installation check passed: their published independent-test result is reproduced exactly.')

stacked_model = check['stacked_model']

# %%
import time, urllib.request
from transformers import TFT5EncoderModel, T5Tokenizer

# ---------------------------------------------------------------------------
# Embedding functions: their two scripts, as functions
#   ProtT5: analyze_Prot_GalNAc_T_833_Proteins.py; Ankh: analyze_ankh_OGalNAc_154.py (repository
#   commit f9934d2). Their lines are kept as written; every change is tagged:
#     [modified] / [added] / [removed], with the original line where one was replaced.
#   Changes, the same for both models:
#     - input: one sequence passed as an argument, instead of one FASTA file per SLURM job;
#     - output: the per-residue array is RETURNED (later reduced to the S/T rows the model uses and
#       saved as NumPy), instead of being written as text, one CSV per protein (decided for NB10:
#       identical vectors, ~35 GB of text avoided);
#     - each model is loaded once for all proteins, instead of once per SLURM job;
#     - Ankh runs on the GPU (their script runs it where the tensors are created, the CPU).
#       ProtT5 (TensorFlow) uses the GPU without any code change.
#   TF32 is off (installation check cell), so the GPU computes in full float32 precision.
# ---------------------------------------------------------------------------
#@ Stack-OglyPred-PLM analyze_Prot_GalNAc_T_833_Proteins.py 13-15 shift=0
def prott5_embedding(seq):  # [added] one sequence in, per-residue array out
    # [removed] FASTA reading: for seq_record in SeqIO.parse(name,"fasta"): placeholder = seq_record.id;
    #           seq = str(seq_record.seq); sequence_of_amino_acid_of_protein = seq; placeholder = placeholder.split("|")[1]
#@ Stack-OglyPred-PLM analyze_Prot_GalNAc_T_833_Proteins.py 21-21 shift=-4
#@ Stack-OglyPred-PLM analyze_Prot_GalNAc_T_833_Proteins.py 25-43 shift=-4
    # [removed] the per-residue CSV writing (filename = placeholder+"_Prot_Trans_"+".csv" ... fp.close())
    assert features.shape == (length_of_protein, 1024)  # [added] one row per residue
    return features  # [added] the per-residue array, written later as NumPy (S/T rows only)
#@ Stack-OglyPred-PLM analyze_ankh_OGalNAc_154.py 27-30 shift=0
ankh_model, ankh_tokenizer = ankh.load_large_model()  # [modified] original: {{Stack-OglyPred-PLM analyze_ankh_OGalNAc_154.py 32}}
ankh_model.eval()  # [modified] original: {{Stack-OglyPred-PLM analyze_ankh_OGalNAc_154.py 33}}
ankh_model = ankh_model.to('cuda')  # [added] run Ankh on the GPU

def ankh_embedding(sequence):  # [added] one sequence in, per-residue array out
    protein_sequences = [sequence]  # [modified] original: protein_sequences.append(Sequence) (read from a FASTA file)
    protein_sequences = [list(seq) for seq in protein_sequences]
    outputs = ankh_tokenizer.batch_encode_plus(protein_sequences,  # [modified] original: tokenizer.batch_encode_plus(
#@ Stack-OglyPred-PLM analyze_ankh_OGalNAc_154.py 37-41 shift=4
        embeddings = ankh_model(input_ids=outputs['input_ids'].to('cuda'),  # [modified] original: model(input_ids=outputs['input_ids'], attention_mask=outputs['attention_mask'])
                                attention_mask=outputs['attention_mask'].to('cuda'))
#@ Stack-OglyPred-PLM analyze_ankh_OGalNAc_154.py 45-45 shift=4
    # [removed] the per-residue CSV writing (filename = str(placeholder)+"_ANKH_"+".csv" ... fp.close())
    exact_embedding = exact_embedding.cpu().numpy()  # [added] back to the CPU as a NumPy array
    assert exact_embedding.shape == (len(sequence), 1536)  # [added] one row per residue
    return exact_embedding  # [added] the per-residue array, written later as NumPy (S/T rows only)

# ---------------------------------------------------------------------------
# Embedding check: our vectors vs their released vectors, for 10 of their independent-test proteins
#   Their feature files give, per S/T site, the UniProt accession, the 1-based position, the
#   residue and the vector. Proteins are taken in random order (seed 0); a protein is used if it
#   has fewer than 1,000 residues and its current UniProt sequence has the same S/T at every one of
#   their positions (otherwise the sequence changed since they ran).
# ---------------------------------------------------------------------------
N_PROTEINS, MAX_LEN = 10, 1000

ankh_sites = pd.concat([check['df_positive'], check['df_negative']], ignore_index=True)
t5_sites = pd.concat([check['df_positive_testing'], check['df_negative_testing']], ignore_index=True)
assert (ankh_sites['PID'].astype(str).values == t5_sites['PID'].astype(str).values).all()
assert (ankh_sites['Position'].values == t5_sites['Position'].values).all()
their_ankh = ankh_sites[list(range(1, 1537))].to_numpy(dtype=np.float64)
their_t5 = t5_sites.drop(columns=['Unnamed: 0', 'Position', 'PID', 'S or T']).to_numpy(dtype=np.float64)
assert their_t5.shape[1] == 1024

def uniprot_sequence(accession):
    with urllib.request.urlopen(f'https://rest.uniprot.org/uniprotkb/{accession}.fasta', timeout=60) as r:
        return ''.join(r.read().decode().split('\n')[1:])

rng = np.random.default_rng(0)
candidates = rng.permutation(sorted(ankh_sites['PID'].astype(str).unique()))
rows, used, skipped = [], [], []
for pid in candidates:
    if len(used) == N_PROTEINS:
        break
    try:
        seq = uniprot_sequence(pid)
    except Exception as e:
        skipped.append((pid, f'UniProt: {e}'))
        continue
    idx = np.flatnonzero(ankh_sites['PID'].astype(str).values == pid)
    pos = ankh_sites['Position'].values[idx].astype(int)
    res = ankh_sites['S_or_T'].values[idx]
    if len(seq) >= MAX_LEN:
        skipped.append((pid, f'{len(seq)} residues'))
        continue
    if pos.max() > len(seq) or any(seq[p - 1] != r for p, r in zip(pos, res)):
        skipped.append((pid, 'sequence differs from theirs at their S/T positions'))
        continue
    t0 = time.time()
    ours_t5 = prott5_embedding(seq)[pos - 1]
    t_t5 = time.time() - t0
    t0 = time.time()
    ours_ankh = ankh_embedding(seq)[pos - 1]
    t_ankh = time.time() - t0
    for name, ours, theirs in [('ProtT5', ours_t5, their_t5[idx]), ('Ankh', ours_ankh, their_ankh[idx])]:
        diff = np.abs(ours - theirs)
        cos = (ours * theirs).sum(1) / (np.linalg.norm(ours, axis=1) * np.linalg.norm(theirs, axis=1))
        rows.append({'protein': pid, 'length': len(seq), 'sites': len(idx), 'model': name,
                     'max_abs_diff': diff.max(), 'mean_abs_diff': diff.mean(), 'min_cosine': cos.min(),
                     'seconds': t_t5 if name == 'ProtT5' else t_ankh})
    used.append(pid)
    print(f'{pid}: {len(seq)} residues, {len(idx)} S/T sites   ProtT5 {t_t5:.1f} s   Ankh {t_ankh:.1f} s')

emb_check = pd.DataFrame(rows)
emb_check.to_csv(f'{OUT_DIR}/embedding_check.csv', index=False)
print(f'\nProteins skipped: {len(skipped)}')
for pid, why in skipped:
    print(f'  {pid}: {why}')
print()
print(emb_check.to_string(index=False, formatters={'max_abs_diff': '{:.2e}'.format, 'mean_abs_diff': '{:.2e}'.format,
                                                     'min_cosine': '{:.8f}'.format, 'seconds': '{:.1f}'.format}))
print('\nOver all checked sites:')
print(emb_check.groupby('model').agg(max_abs_diff=('max_abs_diff', 'max'), min_cosine=('min_cosine', 'min'))
      .to_string(formatters={'max_abs_diff': '{:.2e}'.format, 'min_cosine': '{:.8f}'.format}))

# %%
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

# ---------------------------------------------------------------------------
# Our test set: embeddings and scores
#   Sites: every S/T of the test proteins except masked residues (the same sites as the
#   embeddings notebook), in protein and position order.
#   Each protein is embedded at full length with both pLMs, as their scripts do (no truncation;
#   the longest test protein, 5,115 residues, fits on the GPU). Only the S/T rows the model uses are
#   kept, saved as NumPy float32 arrays (decided for NB10, instead of their per-residue text CSVs).
#   Score: the stacked model's second output. Yes/no call: their rule from "Nucleocytoplasmic
#   GlcNAc Final Program" (outputs > 0.5, then argmax), i.e. positive when the second output
#   exceeds 0.5. TF32 is off (installation check cell).
#   Timing: wall-clock time of the embeddings (both pLMs, all test proteins) plus the stacked
#   model, saved as results/compute/NB10_test.json (compute-records cell) when the embeddings are
#   computed in this run; when they are reloaded, no record is written. If scores from an earlier
#   run exist, the new scores are compared with them before they are replaced.
# ---------------------------------------------------------------------------
EMB_DIR = f'{OUT_DIR}/test_embeddings'
os.makedirs(EMB_DIR, exist_ok=True)

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
                     'label': int(i + 1 in positive)})
sites = pd.DataFrame(rows)

# Same test sites as the embeddings notebook
reference = pd.read_parquet('data_processed/embeddings/sites.parquet', columns=['split', 'label'])
reference = reference[(reference['split'] == 'test') & (reference['label'] != 'masked')]
assert len(sites) == len(reference) and sites['label'].sum() == (reference['label'] == 'positive').sum()
extra = set(''.join(test['sequence'])) - set('ACDEFGHIKLMNPQRSTVWY')
print(f'Test proteins: {len(test):,}   S/T sites: {len(sites):,} ({sites["label"].sum():,} positive)   '
      f'masked S/T excluded: {n_masked:,}   letters outside the 20 amino acids: {sorted(extra)}')

# Embeddings of the S/T sites (computed once; reloaded on reruns)
ANKH_FILE, T5_FILE = f'{EMB_DIR}/ankh_st.npy', f'{EMB_DIR}/prott5_st.npy'
if os.path.exists(ANKH_FILE) and os.path.exists(T5_FILE):
    X_ankh, X_t5 = np.load(ANKH_FILE), np.load(T5_FILE)
    saved = pd.read_parquet(f'{EMB_DIR}/sites.parquet')
    assert saved.equals(sites[['protein_id', 'position']].reset_index(drop=True)), 'saved embeddings are for other sites'
    embed_seconds = None
else:
    X_ankh = np.empty((len(sites), 1536), dtype=np.float32)
    X_t5 = np.empty((len(sites), 1024), dtype=np.float32)
    site_rows = sites.groupby('protein_id', sort=False).indices
    t0 = time.perf_counter()
    for pid, seq in tqdm(zip(test['protein_id'], test['sequence']), total=len(test), desc='Embedding test proteins'):
        if pid not in site_rows:   # every S/T of this protein is masked
            continue
        r = site_rows[pid]
        pos = sites['position'].values[r]
        assert all(seq[p - 1] in 'ST' for p in pos)
        X_t5[r] = prott5_embedding(seq)[pos - 1]
        X_ankh[r] = ankh_embedding(seq)[pos - 1]
    embed_seconds = time.perf_counter() - t0
    print(f'Embeddings computed in {embed_seconds / 60:.1f} min')
    np.save(ANKH_FILE, X_ankh)
    np.save(T5_FILE, X_t5)
    sites[['protein_id', 'position']].to_parquet(f'{EMB_DIR}/sites.parquet', index=False)
assert np.isfinite(X_ankh).all() and np.isfinite(X_t5).all()

# Scores
t0 = time.perf_counter()
probs = stacked_model.predict([X_ankh, X_t5], batch_size=4096, verbose=0)
predict_seconds = time.perf_counter() - t0
if embed_seconds is None:
    print(f'Embeddings reloaded from {EMB_DIR}: no compute record written (remove the folder to time a full run)')
else:
    save_compute('NB10', 'test', embed_seconds + predict_seconds, proteins=len(test), st_sites=len(sites))
assert probs.shape == (len(sites), 2)
called_their_rule = np.array([np.argmax(y, axis=None, out=None) for y in (probs > 0.5)]) == 1
print(f'Output rows summing to 1 (softmax): {np.allclose(probs.sum(axis=1), 1, atol=1e-5)}   '
      f'calls where their rule differs from "second output > 0.5": {(called_their_rule != (probs[:, 1] > 0.5)).sum()}')
sites['score'] = probs[:, 1]
sites['called'] = called_their_rule
if os.path.exists(f'{OUT_DIR}/test_scores.parquet'):
    previous = pd.read_parquet(f'{OUT_DIR}/test_scores.parquet')
    assert previous[['protein_id', 'position']].equals(sites[['protein_id', 'position']])
    print(f'Previous run vs this run: score max |diff| {(previous["score"] - sites["score"]).abs().max():.1e}   '
          f'calls changed {int((previous["called"] != sites["called"]).sum())}')
sites.to_parquet(f'{OUT_DIR}/test_scores.parquet', index=False)

y_true = sites['label'].to_numpy()
tp = int((sites['called'] & (y_true == 1)).sum())
metrics = pd.DataFrame([{'model': 'Stack-OglyPred-PLM (nucleocytoplasmic)', 'sites': len(sites),
                         'positive': int(y_true.sum()), 'AUROC': roc_auc_score(y_true, sites['score']),
                         'AUPRC': average_precision_score(y_true, sites['score']),
                         'MCC': matthews_corrcoef(y_true, sites['called']), 'precision': tp / max(sites['called'].sum(), 1),
                         'recall': tp / y_true.sum(), 'called positive': int(sites['called'].sum())}])
metrics.to_csv(f'{OUT_DIR}/test_metrics.csv', index=False)
print(f'\nAll test sites (positive fraction {y_true.mean():.4f}); yes/no calls at their cut-off (0.5):')
print(metrics.to_string(index=False, float_format='{:.4f}'.format))

# %%
import torch

# ---------------------------------------------------------------------------
# Overlap of our test proteins with their training proteins (decision 5b)
#   Their training proteins: every accession in their released 81-residue training windows
#   (GlacNAC_81_window_Training_Positive/Negative.fasta, "NucleoCytoplasmic" folder of their
#   Drive; headers "accession|position"), 1,638 proteins as in their paper (§2.1.1).
#   Their files hold windows, not full sequences, so the full sequences are downloaded from
#   UniProt by accession (current versions; a sequence may have changed since they ran). Entries
#   deleted from UniProt since then (Q5SWW7 and Q96S07, deleted in release 2025_04) are taken
#   from UniSave, UniProt's archive of entry versions, at their last version.
#   Two definitions of overlap, each removing whole test proteins (as for the Seber RNNs):
#     (a) homology, as in our own split: identity >= 30% to any of their training proteins, with
#         identity = identical residues / length of the shorter protein, from an exhaustive
#         MMseqs2 alignment (fake prefilter, --alignment-mode 3, E <= 0.001; E-values relative to
#         their training proteins);
#     (b) same protein: the UniProt accession (isoform suffix ignored) or the exact sequence is
#         among their training proteins.
#   Metrics on the remaining sites: Stack-OglyPred-PLM (cut-off 0.5) and our final model
#   (calibrated probability >= its validation MCC cut-off).
# ---------------------------------------------------------------------------
TRAIN_WINDOWS = {'GlacNAC_81_window_Training_Positive.fasta': '1IeT6yoVNSgF7EM_ybCGh4VxqaUkLq2St',
                 'GlacNAC_81_window_Training_Negative.fasta': '1f7W3Qm2hlCIVPeN9BCZgOBwjwUG4_rNN'}
for name, file_id in TRAIN_WINDOWS.items():
    if not os.path.exists(f'{THEIRS}/{name}'):
        gdown.download(id=file_id, output=f'{THEIRS}/{name}', quiet=False)

train_ids = sorted({line[1:].split('|')[0] for name in TRAIN_WINDOWS
                    for line in open(f'{THEIRS}/{name}') if line.startswith('>')})
print(f'Their training proteins: {len(train_ids):,}')
assert len(train_ids) == 1638

# Full sequences from UniProt (canonical entries in batches; isoforms one by one; deleted
# entries from UniSave at their last version)
TRAIN_FASTA = f'{OUT_DIR}/their_training_proteins_uniprot.fasta'

def read_fasta(path):
    seqs, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith('>'):
            name = line[1:].split('|')[1] if '|' in line else line[1:].split()[0]
            seqs[name] = ''
        elif name:
            seqs[name] += line
    return seqs

if not os.path.exists(TRAIN_FASTA):
    chunks = []
    canonical = [a for a in train_ids if '-' not in a]
    for i in tqdm(range(0, len(canonical), 200), desc='UniProt (canonical, batches of 200)'):
        url = 'https://rest.uniprot.org/uniprotkb/accessions?format=fasta&accessions=' + ','.join(canonical[i:i + 200])
        with urllib.request.urlopen(url, timeout=120) as r:
            chunks.append(r.read().decode())
    for a in tqdm([a for a in train_ids if '-' in a], desc='UniProt (isoforms)'):
        try:
            with urllib.request.urlopen(f'https://rest.uniprot.org/uniprotkb/{a}.fasta', timeout=60) as r:
                chunks.append(r.read().decode())
        except Exception as e:
            print(f'  {a}: {e}')
    retrieved = {line.split('|')[1] for line in ''.join(chunks).splitlines() if line.startswith('>')}
    for a in [a for a in train_ids if a not in retrieved]:
        with urllib.request.urlopen(f'https://rest.uniprot.org/unisave/{a}?format=json', timeout=60) as r:
            last = json.load(r)['results'][0]
        with urllib.request.urlopen(f'https://rest.uniprot.org/unisave/{a}?format=fasta&versions={last["entryVersion"]}',
                                    timeout=60) as r:
            chunks.append(r.read().decode())
        print(f'  {a}: not in current UniProt; UniSave version {last["entryVersion"]} '
              f'(last release {last["lastRelease"]})')
    with open(TRAIN_FASTA, 'w') as f:
        f.write(''.join(chunks))
train_seqs = read_fasta(TRAIN_FASTA)
missing = sorted(set(train_ids) - set(train_seqs))
print(f'Training sequences: {len(train_seqs):,}   not retrieved: {len(missing)} {missing[:10]}')
assert not missing

# MMseqs2: every test protein against every training protein
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

ALN_DIR = '/content/stack_overlap'
os.makedirs(ALN_DIR, exist_ok=True)
ALN_FILE = os.path.abspath(f'{OUT_DIR}/test_vs_training_alignments.tsv')   # MMseqs2 runs in ALN_DIR
if not os.path.exists(ALN_FILE):
    write_fasta(f'{ALN_DIR}/test.fasta', test['protein_id'], test['sequence'])
    write_fasta(f'{ALN_DIR}/theirs.fasta', list(train_seqs), list(train_seqs.values()))
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

# Per test protein: closest training protein, and both overlap flags
best = aln.sort_values('id_shortest', ascending=False).drop_duplicates('query').set_index('query')
overlap = test[['protein_id', 'organism', 'sequence']].copy()
overlap['max_identity'] = overlap['protein_id'].map(best['id_shortest']).fillna(0.0)
overlap['closest_training_protein'] = overlap['protein_id'].map(best['target'])
base = lambda ids: pd.Series(ids).astype(str).str.split('-').str[0]
overlap['same_protein'] = (base(overlap['protein_id']).isin(set(base(train_ids))).to_numpy()
                           | overlap['sequence'].isin(set(train_seqs.values())).to_numpy())
overlap['homolog_30'] = overlap['max_identity'] >= 0.30
assert (overlap.loc[overlap['sequence'].isin(set(train_seqs.values())), 'max_identity'] >= 0.99).all()
overlap.drop(columns='sequence').to_csv(f'{OUT_DIR}/test_protein_overlap.csv', index=False)

# Our final model's test scores and cut-off (NB05)
ours = pd.read_parquet('data_processed/final_model/test_scores.parquet')[['protein_id', 'position', 'probability']]
OUR_CUTOFF = torch.load('data_processed/final_model/final_model.pt', map_location='cpu')['cutoff']['probability']
scored = sites.merge(ours, on=['protein_id', 'position'], how='left', validate='one_to_one')
assert scored['probability'].notna().all()

flags = overlap.set_index('protein_id')
SUBSETS = {'all test proteins': np.ones(len(scored), bool),
           '(b) same protein removed': ~scored['protein_id'].map(flags['same_protein']).to_numpy(),
           '(a) homologs >= 30% removed': ~scored['protein_id'].map(flags['homolog_30']).to_numpy()}
SCORES = {'Stack-OglyPred-PLM': ('score', 0.5), 'Our model': ('probability', OUR_CUTOFF)}

rows = []
for subset, keep in SUBSETS.items():
    y_sub = scored.loc[keep, 'label'].to_numpy()
    for name, (col, cut) in SCORES.items():
        score = scored.loc[keep, col].to_numpy()
        called = score > cut if name == 'Stack-OglyPred-PLM' else score >= cut   # their rule: > 0.5
        tp = int((called & (y_sub == 1)).sum())
        rows.append({'subset': subset, 'proteins': scored.loc[keep, 'protein_id'].nunique(), 'sites': int(keep.sum()),
                     'positive': int(y_sub.sum()), 'model': name, 'cut-off': cut,
                     'AUROC': roc_auc_score(y_sub, score), 'AUPRC': average_precision_score(y_sub, score),
                     'MCC': matthews_corrcoef(y_sub, called), 'precision': tp / max(called.sum(), 1),
                     'recall': tp / y_sub.sum()})
filtered = pd.DataFrame(rows)
filtered.to_csv(f'{OUT_DIR}/test_metrics_filtered.csv', index=False)

print(f'\nTest proteins: {len(overlap)}   same protein as a training protein: {overlap["same_protein"].sum()}   '
      f'homolog >= 30%: {overlap["homolog_30"].sum()}')
print('Highest identity to their training proteins (test proteins), quantiles 0/25/50/75/100%:',
      np.round(overlap['max_identity'].quantile([0, 0.25, 0.5, 0.75, 1]).to_numpy(), 2))
print()
print(filtered.to_string(index=False, float_format='{:.4f}'.format))

# %%
# ---------------------------------------------------------------------------
# Human test proteins only (Stack-OglyPred-PLM was trained on human dbPTM proteins)
#   Same subsets, models and cut-offs as the previous cell, restricted to test proteins whose kept
#   UniProt ID is human. In each row both models are scored on the same sites.
# ---------------------------------------------------------------------------
human = (scored['organism'] == 'Homo sapiens').to_numpy()

rows = []
for subset, keep in SUBSETS.items():
    keep_h = keep & human
    y_sub = scored.loc[keep_h, 'label'].to_numpy()
    for name, (col, cut) in SCORES.items():
        score = scored.loc[keep_h, col].to_numpy()
        called = score > cut if name == 'Stack-OglyPred-PLM' else score >= cut   # their rule: > 0.5
        tp = int((called & (y_sub == 1)).sum())
        rows.append({'subset': subset, 'proteins': scored.loc[keep_h, 'protein_id'].nunique(),
                     'sites': int(keep_h.sum()), 'positive': int(y_sub.sum()), 'model': name, 'cut-off': cut,
                     'AUROC': roc_auc_score(y_sub, score), 'AUPRC': average_precision_score(y_sub, score),
                     'MCC': matthews_corrcoef(y_sub, called), 'precision': tp / max(called.sum(), 1),
                     'recall': tp / y_sub.sum()})
human_metrics = pd.DataFrame(rows)
human_metrics.to_csv(f'{OUT_DIR}/test_metrics_human.csv', index=False)
print('Human test proteins only:')
print(human_metrics.to_string(index=False, float_format='{:.4f}'.format))

# %%
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# External validation sets: every S/T of every external protein (NB01)
#   As for the test set (cell 4): each protein embedded at full length with both pLMs (their
#   scripts); score = the stacked model's second output; call = their rule (second output > 0.5).
#   Full-length embedding needs memory growing with the square of the length (ProtT5's attention
#   tables: ~33 GB each at 16,000 residues). Proteins up to GPU_MAX_LEN residues (the longest test
#   protein, embedded on this GPU in cell 4) are embedded here; longer ones are written to
#   long_proteins.parquet and embedded on a CPU server with the same code (their scripts ran on
#   CPUs), then merged.
#   Proteins are processed in chunks; the scores of each chunk are saved when complete and skipped
#   on a rerun. Only the scores are kept (the S/T embeddings would take ~5 GB).
#   Timing: wall-clock time of each chunk (embeddings and stacked model, excluding the write to
#   Drive), saved next to the chunk as chunk_NNN.json with the GPU name; the sum over chunks is
#   saved as results/compute/NB10_external_gpu_part.json (compute-records cell) once every chunk is
#   timed. Chunks are kept in external_runs/gpu_chunks.
# ---------------------------------------------------------------------------
EXT_RUNS = f'{OUT_DIR}/external_runs'
os.makedirs(EXT_RUNS, exist_ok=True)
CHUNK = 200          # proteins per saved chunk
GPU_MAX_LEN = 5115   # longest test protein

external = pd.read_parquet('data_processed/external_proteins.parquet').sort_values('seq_id').reset_index(drop=True)
external['length'] = external['sequence'].str.len()
external['n_st'] = external['sequence'].str.count('[ST]')
extra = set(''.join(external['sequence'])) - set('ACDEFGHIKLMNPQRSTVWY')
long_proteins = external[external['length'] > GPU_MAX_LEN]
long_proteins[['seq_id', 'sequence', 'length', 'n_st']].to_parquet(f'{EXT_RUNS}/long_proteins.parquet', index=False)
gpu_proteins = external[(external['length'] <= GPU_MAX_LEN) & (external['n_st'] > 0)].reset_index(drop=True)
print(f'External proteins: {len(external):,}   letters outside the 20 amino acids: {sorted(extra)}')
print(f'  here (<= {GPU_MAX_LEN:,} residues, with S/T): {len(gpu_proteins):,} proteins, {gpu_proteins["n_st"].sum():,} S/T')
print(f'  CPU server (> {GPU_MAX_LEN:,} residues): {len(long_proteins):,} proteins, {long_proteins["n_st"].sum():,} S/T, '
      f'lengths {long_proteins["length"].min():,}-{long_proteins["length"].max():,}')

n_chunks = -(-len(gpu_proteins) // CHUNK)
CHUNK_DIR = f'{EXT_RUNS}/gpu_chunks'
os.makedirs(CHUNK_DIR, exist_ok=True)
chunk_path = lambda c: f'{CHUNK_DIR}/chunk_{c:03d}.parquet'
timing_path = lambda c: f'{CHUNK_DIR}/chunk_{c:03d}.json'
done = [c for c in range(n_chunks) if os.path.exists(chunk_path(c))]
bar = tqdm(total=len(gpu_proteins), initial=sum(len(gpu_proteins.iloc[c * CHUNK:(c + 1) * CHUNK]) for c in done),
           desc='Embedding and scoring external proteins')
for c in range(n_chunks):
    if os.path.exists(chunk_path(c)):
        continue
    t0 = time.perf_counter()
    rows, ankh_rows, t5_rows = [], [], []
    for sid, seq in zip(gpu_proteins['seq_id'].iloc[c * CHUNK:(c + 1) * CHUNK], gpu_proteins['sequence'].iloc[c * CHUNK:(c + 1) * CHUNK]):
        pos = np.array([i + 1 for i, aa in enumerate(seq) if aa in 'ST'])
        t5_rows.append(prott5_embedding(seq)[pos - 1])
        ankh_rows.append(ankh_embedding(seq)[pos - 1])
        rows += [(sid, p, seq[p - 1]) for p in pos]
        bar.update(1)
    X_ankh, X_t5 = np.concatenate(ankh_rows).astype(np.float32), np.concatenate(t5_rows).astype(np.float32)
    assert np.isfinite(X_ankh).all() and np.isfinite(X_t5).all()
    probs = stacked_model.predict([X_ankh, X_t5], batch_size=4096, verbose=0)
    chunk = pd.DataFrame(rows, columns=['seq_id', 'position', 'residue'])
    chunk['score'] = probs[:, 1]
    chunk['called'] = np.array([np.argmax(y, axis=None, out=None) for y in (probs > 0.5)]) == 1
    seconds = time.perf_counter() - t0
    chunk.to_parquet(chunk_path(c) + '.part', index=False)
    with open(timing_path(c), 'w') as f:   # written before the chunk is complete: a chunk on Drive has its time
        json.dump({'seconds': round(seconds, 2), 'proteins': len(chunk['seq_id'].unique()),
                   'st_sites': len(chunk), 'gpu': gpu_name()}, f)
    os.replace(chunk_path(c) + '.part', chunk_path(c))
bar.close()

timings = [json.load(open(timing_path(c))) for c in range(n_chunks) if os.path.exists(timing_path(c))]
if len(timings) == n_chunks:
    save_compute('NB10', 'external_gpu_part', sum(t['seconds'] for t in timings),
                 proteins=len(gpu_proteins), st_sites=int(gpu_proteins['n_st'].sum()))
    print(f'GPU(s): {", ".join(sorted({t["gpu"] for t in timings}))}')
else:
    print(f'{n_chunks - len(timings)} of {n_chunks} chunks have no timing: no compute record written')

gpu_scores = pd.concat([pd.read_parquet(chunk_path(c)) for c in range(n_chunks)], ignore_index=True)
assert len(gpu_scores) == gpu_proteins['n_st'].sum()
assert not gpu_scores.duplicated(['seq_id', 'position']).any()
if os.path.exists(f'{EXT_RUNS}/scores_gpu_part.parquet'):
    previous = pd.read_parquet(f'{EXT_RUNS}/scores_gpu_part.parquet')
    assert previous[['seq_id', 'position']].equals(gpu_scores[['seq_id', 'position']])
    print(f'Previous run vs this run: score max |diff| {(previous["score"] - gpu_scores["score"]).abs().max():.1e}   '
          f'calls changed {int((previous["called"] != gpu_scores["called"]).sum())}')
gpu_scores.to_parquet(f'{EXT_RUNS}/scores_gpu_part.parquet', index=False)
print(f'Saved {len(gpu_scores):,} S/T scores ({int(gpu_scores["called"].sum()):,} called positive): '
      f'{EXT_RUNS}/scores_gpu_part.parquet; long proteins: {EXT_RUNS}/long_proteins.parquet')

# %%
import os
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))
import pandas as pd

# ---------------------------------------------------------------------------
# External validation sets: merge of the two parts
#   GPU part: proteins up to 5,115 residues, scored in this notebook (previous cell).
#   CPU part: the longer proteins, scored on the lab server with the same embedding code
#   (scripts/stack_oglypred_plm_long_proteins.py; its check against this notebook's GPU scores on
#   three proteins: max |score difference| 3.0e-06, no call changed).
# ---------------------------------------------------------------------------
OUT_DIR = 'data_processed/other_tools/Stack-OglyPred-PLM'
EXT_RUNS = f'{OUT_DIR}/external_runs'
gpu_part = pd.read_parquet(f'{EXT_RUNS}/scores_gpu_part.parquet').assign(device='GPU')
long_part = pd.read_parquet(f'{EXT_RUNS}/scores_long_part.parquet').assign(device='CPU')
assert set(long_part['seq_id']) == set(pd.read_parquet(f'{EXT_RUNS}/long_proteins.parquet')['seq_id'])
ext_scores = pd.concat([gpu_part, long_part], ignore_index=True).sort_values(['seq_id', 'position']).reset_index(drop=True)
external = pd.read_parquet('data_processed/external_proteins.parquet')
assert len(ext_scores) == external['sequence'].str.count('[ST]').sum()
assert not ext_scores.duplicated(['seq_id', 'position']).any()
ext_scores.to_parquet(f'{OUT_DIR}/external_scores.parquet', index=False)
print(ext_scores.groupby('device').agg(proteins=('seq_id', 'nunique'), sites=('score', 'size'),
                                       called=('called', 'sum')).to_string())
print(f'Saved {len(ext_scores):,} S/T scores: {OUT_DIR}/external_scores.parquet')
