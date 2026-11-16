# %%
import os
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))

import ast, hashlib, json, subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Our BSB 2025 model (conference paper, camera-ready version in old_article/),
# run with its released code and weights
#   Repository: github.com/LBS-UFMG/OGlcNAc, pinned to its last commit (c1d6d35, 2025-07-24).
#   Site representation: ESM2-8M (facebook/esm2_t6_8M_UR50D) last hidden layer at the site's
#   residue (320 values), from the whole protein (03_fetching_esm2_embeddings.py); proteins of
#   at most 1,022 residues only (the model's documented restriction).
#   Classifier: SimpleMLP 320 -> 18 -> 1 with dropout 0.5 and a sigmoid output
#   (04_training.py); released weights from epoch 47; a site is called positive at >= 0.5
#   (05_analysis.py).
# ---------------------------------------------------------------------------
REPO_URL = 'https://github.com/LBS-UFMG/OGlcNAc'
REPO_COMMIT = 'c1d6d351b1339562337fe1d7fc6b68462158d2ba'   # 2025-07-24, "Update README.md"
REPO = '/content/OGlcNAc_BSB2025'
WEIGHTS = f'{REPO}/models/model_MLP_320_18_1_with_dropout_epoch_47_weights.pth'
WEIGHTS_SHA256 = 'a12fd6cdeff644b7972dc7a86b75cbb4cfcae491fa0ac2b7ef6e1f055305921c'
OUT_DIR = 'data_processed/other_tools/BSB2025'
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(REPO):
    subprocess.run(f'git clone -q {REPO_URL} {REPO}', shell=True, check=True)
subprocess.run(f'git -C {REPO} checkout -q {REPO_COMMIT}', shell=True, check=True)
with open(WEIGHTS, 'rb') as f:
    assert hashlib.sha256(f.read()).hexdigest() == WEIGHTS_SHA256, 'weights file differs'

# Model class taken unchanged from 04_training.py
train_src = open(f'{REPO}/04_training.py', encoding='utf-8').read()
node = next(n for n in ast.parse(train_src).body if isinstance(n, ast.ClassDef) and n.name == 'SimpleMLP')
exec(ast.get_source_segment(train_src, node))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = False   # full float32 precision throughout
torch.backends.cudnn.allow_tf32 = False

# As in 05_analysis.py: input 320, hidden 18, output 1, dropout 0.5; weights of epoch 47
old_model = SimpleMLP(320, 18, 1, 0.5)
old_model.load_state_dict(torch.load(WEIGHTS, map_location=device))
old_model.to(device)
old_model.eval()
print(old_model)

with open(f'{OUT_DIR}/provenance.json', 'w') as f:
    json.dump({'repository': REPO_URL, 'commit': REPO_COMMIT, 'weights': os.path.basename(WEIGHTS),
               'weights_sha256': WEIGHTS_SHA256, 'torch': torch.__version__}, f, indent=1)

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
import subprocess
subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv'])
subprocess.run(['free', '-g'])
subprocess.run(['df', '-h', '/root'])

# %%
import h5py
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------------
# Installation check: the released validation and test embeddings
#   data_processed/03_seqs_esm2_t6_8M_UR50D_{validation,testing}.h5 (balanced sets, one row per
#   site, with the ESM2-8M embedding), scored with the epoch-47 weights; prediction = probability
#   >= 0.5 and accuracy / precision / recall / F1 from the confusion matrix, as in 05_analysis.py.
#   Expected: the two rows of BSB 2025 Table 1 (validation 76.91 / 77.41 / 76.01 / 76.70 %;
#   testing 76.06 / 77.66 / 73.17 / 75.35 %). The released "testing" file reproduces the row
#   labelled Validation and the "validation" file the row labelled Testing (labels swapped
#   between the paper's table and the released files).
# ---------------------------------------------------------------------------
PUBLISHED = {'validation': (76.91, 77.41, 76.01, 76.70), 'testing': (76.06, 77.66, 73.17, 75.35)}
METRICS = ['accuracy', 'precision', 'recall', 'F1']

rows = []
for split in PUBLISHED:
    with h5py.File(f'{REPO}/data_processed/03_seqs_esm2_t6_8M_UR50D_{split}.h5', 'r') as f:
        X = torch.from_numpy(f['embedding'][:]).float().to(device)
        y = f['OGlcNAc'][:]
    with torch.no_grad():
        prob = old_model(X).squeeze(1).cpu().numpy()
    pred = (prob >= 0.5).astype(int)
    TN, FP, FN, TP = confusion_matrix(y, pred).ravel()
    precision, recall = TP / (TP + FP), TP / (TP + FN)
    ours = (100 * (TP + TN) / len(y), 100 * precision, 100 * recall, 100 * 2 * precision * recall / (precision + recall))
    match = [name for name, pub in PUBLISHED.items() if all(round(o, 2) == p for o, p in zip(ours, pub))]
    rows.append({'released file': split, 'sites': len(y), **dict(zip(METRICS, ours)),
                 'matches Table 1 row': match[0] if match else 'none'})

check = pd.DataFrame(rows)
check.to_csv(f'{OUT_DIR}/installation_check.csv', index=False)
print(check.to_string(index=False, float_format='{:.2f}'.format))
assert sorted(check['matches Table 1 row']) == ['testing', 'validation'], 'Table 1 not reproduced'
print('\nInstallation check passed: both rows of BSB 2025 Table 1 are reproduced exactly '
      '(with the validation/testing labels swapped between the paper and the released files).')

# %%
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

# ---------------------------------------------------------------------------
# Embedding function: 03_fetching_esm2_embeddings.py (repository commit c1d6d35), as a function
#   Their lines are kept as written; every change is tagged [modified] / [added] / [removed], with
#   the original line where one was replaced. Changes:
#     - input: a list of sequences and, for each, the 1-based positions of its sites, instead of
#       the rows of their data frames;
#     - output: the site embeddings are RETURNED as a NumPy array in input order, instead of being
#       stored in the data frame and written to CSV/HDF5 files.
#   Kept from their script: ESM2-8M through AutoModelForMaskedLM, batches of 32 sequences padded
#   to the longest in the batch (sequences taken in order of first appearance), last hidden layer,
#   row index = 1-based position (index 0 is the <cls> token). TF32 is off (cell 0).
# ---------------------------------------------------------------------------
model_name = 'facebook/esm2_t6_8M_UR50D' #smallest model
tokenizer = AutoTokenizer.from_pretrained(model_name)
esm_model = AutoModelForMaskedLM.from_pretrained(model_name)  # [modified] original: model = AutoModelForMaskedLM.from_pretrained(model_name)
esm_model.to(device)  # [modified] original: model.to(device)
esm_model.eval()  # [modified] original: model.eval()

batch_size = 32

def esm2_site_embeddings(sequences, positions):  # [added] sequences with their 1-based site positions in, embeddings out
    seqs = list(dict.fromkeys(sequences))  # [modified] original: seqs = list(df['sequence'].unique())
    site_rows = {}  # [added] sequence -> rows of the output that belong to it
    for k, s in enumerate(sequences):  # [added]
        site_rows.setdefault(s, []).append(k)  # [added]
    out = np.empty((len(sequences), esm_model.config.hidden_size), dtype=np.float32)  # [added]
    for i in tqdm(range(0, len(seqs), batch_size)):
        batch_seqs = seqs[i:i+batch_size]

        inputs = tokenizer(batch_seqs, return_tensors='pt', padding=True).to(device)
        with torch.no_grad():
            outputs = esm_model(**inputs, output_hidden_states=True) #** unpacks inputs  # [modified] original: outputs = model(**inputs, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1] #shape is (batch size, max length of sequence in batch + 2, embedding dimension)

        #for loop only in the rows of df where df['sequence'] is one of batch_seqs
        for seq in batch_seqs:
            for j in site_rows[seq]:  # [modified] original: for j, row in df[df['sequence'] == seq].iterrows():
                out[j] = last_hidden_states[batch_seqs.index(seq), positions[j], :].cpu().numpy()  # [modified] original: df.at[j, 'embedding'] = last_hidden_states[batch_seqs.index(seq), df.at[j, 'position_in_protein'], :].cpu().numpy()
    # [removed] writing the data frames to CSV and HDF5 files
    return out  # [added]

# ---------------------------------------------------------------------------
# Embedding check: recompute the embeddings of their released test file and compare
#   03_seqs_esm2_t6_8M_UR50D_testing.h5 stores, per site, the sequence, the position and the
#   embedding they computed; the sequences are processed in the file's order, as in their script.
# ---------------------------------------------------------------------------
with h5py.File(f'{REPO}/data_processed/03_seqs_esm2_t6_8M_UR50D_testing.h5', 'r') as f:
    their_seqs = [s.decode() for s in f['sequence'][:]]
    their_pos = f['position_in_protein'][:].astype(int)
    their_emb = f['embedding'][:].astype(np.float64)
ours_emb = esm2_site_embeddings(their_seqs, their_pos).astype(np.float64)
diff = np.abs(ours_emb - their_emb)
cos = (ours_emb * their_emb).sum(1) / (np.linalg.norm(ours_emb, axis=1) * np.linalg.norm(their_emb, axis=1))
print(f'Embedding check, {len(their_seqs):,} sites of their test file: max |diff| {diff.max():.2e}   '
      f'mean |diff| {diff.mean():.2e}   min cosine {cos.min():.8f}')

# ---------------------------------------------------------------------------
# Our test set: proteins of at most 1,022 residues (the model's documented restriction)
#   Sites: every S/T of those test proteins except masked residues, in protein and position
#   order. Score: the model's sigmoid output; call at >= 0.5. Our final model is scored on the
#   same sites (calibrated probability >= its validation MCC cut-off).
#   Timing: wall-clock time of the ESM2-8M site embeddings plus the classifier, saved as
#   results/compute/NB11_test.json (compute-records cell). If scores from an earlier run exist,
#   the new scores are compared with them before they are replaced.
# ---------------------------------------------------------------------------
MAX_LEN = 1022
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
                     'label': int(i + 1 in positive), 'length': len(seq)})
all_sites = pd.DataFrame(rows)
reference = pd.read_parquet('data_processed/embeddings/sites.parquet', columns=['split', 'label'])
reference = reference[(reference['split'] == 'test') & (reference['label'] != 'masked')]
assert len(all_sites) == len(reference)

sites = all_sites[all_sites['length'] <= MAX_LEN].reset_index(drop=True)
seq_of = dict(zip(test['protein_id'], test['sequence']))
print(f'\nTest proteins: {len(test):,} ({all_sites["protein_id"].nunique():,} with scored sites); '
      f'<= {MAX_LEN:,} residues: {sites["protein_id"].nunique():,}   '
      f'sites kept: {len(sites):,} of {len(all_sites):,} ({sites["label"].sum():,} of {all_sites["label"].sum():,} positive)')

t0 = time.perf_counter()
X_old = esm2_site_embeddings([seq_of[p] for p in sites['protein_id']], sites['position'].to_numpy())
with torch.no_grad():
    sites['score'] = old_model(torch.from_numpy(X_old).float().to(device)).squeeze(1).cpu().numpy()
save_compute('NB11', 'test', time.perf_counter() - t0, proteins=sites['protein_id'].nunique(), st_sites=len(sites))
sites['called'] = sites['score'] >= 0.5

ours = pd.read_parquet('data_processed/final_model/test_scores.parquet')[['protein_id', 'position', 'probability']]
OUR_CUTOFF = torch.load('data_processed/final_model/final_model.pt', map_location='cpu')['cutoff']['probability']
sites = sites.merge(ours, on=['protein_id', 'position'], how='left', validate='one_to_one')
assert sites['probability'].notna().all()
if os.path.exists(f'{OUT_DIR}/test_scores.parquet'):
    previous = pd.read_parquet(f'{OUT_DIR}/test_scores.parquet')
    assert previous[['protein_id', 'position']].equals(sites[['protein_id', 'position']])
    print(f'Previous run vs this run: score max |diff| {(previous["score"] - sites["score"]).abs().max():.1e}   '
          f'calls changed {int((previous["called"] != sites["called"]).sum())}')
sites.to_parquet(f'{OUT_DIR}/test_scores.parquet', index=False)

y_true = sites['label'].to_numpy()
rows = []
for name, col, cut in [('BSB 2025 model', 'score', 0.5), ('Our model', 'probability', OUR_CUTOFF)]:
    score = sites[col].to_numpy()
    called = score >= cut
    tp = int((called & (y_true == 1)).sum())
    rows.append({'model': name, 'cut-off': cut, 'AUROC': roc_auc_score(y_true, score),
                 'AUPRC': average_precision_score(y_true, score), 'MCC': matthews_corrcoef(y_true, called),
                 'precision': tp / max(called.sum(), 1), 'recall': tp / y_true.sum(), 'called positive': int(called.sum())})
metrics = pd.DataFrame(rows)
metrics.to_csv(f'{OUT_DIR}/test_metrics.csv', index=False)
print(f'\nTest sites on proteins <= {MAX_LEN:,} residues (positive fraction {y_true.mean():.4f}):')
print(metrics.to_string(index=False, float_format='{:.4f}'.format))

# %%
import subprocess

# ---------------------------------------------------------------------------
# Overlap of our test proteins with the proteins the BSB 2025 model has seen (decision 5b)
#   Seen proteins: every accession in its released training and validation splits
#   (data_processed/02_training_dataset_balanced.csv, 02_validation_dataset_balanced.csv); the
#   training split fitted the weights and the validation split selected epoch 47. Sequences are
#   taken from those files (the sequences the model was trained on).
#   Two definitions of overlap, each removing whole test proteins (as for the other tools):
#     (a) homology, as in our own split: identity >= 30% to any seen protein, with identity =
#         identical residues / length of the shorter protein, from an exhaustive MMseqs2 alignment
#         (fake prefilter, --alignment-mode 3, E <= 0.001; E-values relative to the seen proteins);
#     (b) same protein: the UniProt accession (isoform suffix ignored) or the exact sequence is
#         among the seen proteins.
#   Metrics on the remaining sites (test proteins <= 1,022 residues): BSB 2025 model (>= 0.5) and
#   our final model (calibrated probability >= its validation MCC cut-off).
# ---------------------------------------------------------------------------
seen = pd.concat([pd.read_csv(f'{REPO}/data_processed/02_{s}_dataset_balanced.csv', usecols=['accession', 'sequence'])
                  for s in ('training', 'validation')]).drop_duplicates()
assert seen['accession'].is_unique, 'one accession with two sequences'
print(f'Proteins seen by the BSB 2025 model (training + validation splits): {len(seen):,}')

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

ALN_DIR = '/content/bsb2025_overlap'
os.makedirs(ALN_DIR, exist_ok=True)
ALN_FILE = os.path.abspath(f'{OUT_DIR}/test_vs_seen_alignments.tsv')   # MMseqs2 runs in ALN_DIR
if not os.path.exists(ALN_FILE):
    write_fasta(f'{ALN_DIR}/test.fasta', test['protein_id'], test['sequence'])
    write_fasta(f'{ALN_DIR}/seen.fasta', seen['accession'], seen['sequence'])
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

# Per test protein: closest seen protein, and both overlap flags
best = aln.sort_values('id_shortest', ascending=False).drop_duplicates('query').set_index('query')
overlap = test[['protein_id', 'organism', 'sequence']].copy()
overlap['length'] = overlap['sequence'].str.len()
overlap['max_identity'] = overlap['protein_id'].map(best['id_shortest']).fillna(0.0)
overlap['closest_seen_protein'] = overlap['protein_id'].map(best['target'])
base = lambda ids: pd.Series(ids).astype(str).str.split('-').str[0]
overlap['same_protein'] = (base(overlap['protein_id']).isin(set(base(seen['accession']))).to_numpy()
                           | overlap['sequence'].isin(set(seen['sequence'])).to_numpy())
overlap['homolog_30'] = overlap['max_identity'] >= 0.30
assert (overlap.loc[overlap['sequence'].isin(set(seen['sequence'])), 'max_identity'] >= 0.99).all()
overlap.drop(columns='sequence').to_csv(f'{OUT_DIR}/test_protein_overlap.csv', index=False)

flags = overlap.set_index('protein_id')
SUBSETS = {'all test proteins <= 1,022': np.ones(len(sites), bool),
           '(b) same protein removed': ~sites['protein_id'].map(flags['same_protein']).to_numpy(),
           '(a) homologs >= 30% removed': ~sites['protein_id'].map(flags['homolog_30']).to_numpy()}
SCORES = {'BSB 2025 model': ('score', 0.5), 'Our model': ('probability', OUR_CUTOFF)}

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

short = overlap[overlap['length'] <= 1022]
print(f'\nTest proteins <= 1,022 residues: {len(short)}   same protein as a seen protein: {short["same_protein"].sum()}   '
      f'homolog >= 30%: {short["homolog_30"].sum()}')
print('Highest identity to seen proteins (test proteins <= 1,022), quantiles 0/25/50/75/100%:',
      np.round(short['max_identity'].quantile([0, 0.25, 0.5, 0.75, 1]).to_numpy(), 2))
print()
print(filtered.to_string(index=False, float_format='{:.4f}'.format))

# %%
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ---------------------------------------------------------------------------
# External validation sets: every S/T of every external protein (NB01), at full length
#   On our test set the model was evaluated on proteins of at most 1,022 residues (its documented
#   restriction; cell 4). The external sets test every tool on all proteins, so here every protein
#   is embedded at full length with their code (03_fetching_esm2_embeddings.py has no length check
#   or truncation; ESM-2 uses rotary position embeddings, so longer inputs run).
#   ESM2-8M site embeddings from cell 4's embedding function (repeated here so that only cell 0 is
#   needed; their lines kept, changes tagged), scored with the epoch-47 weights; call at >= 0.5.
#   Proteins of at most 1,022 residues: their code as is (batches of 32 padded to the longest).
#   Longer proteins: one protein per batch. Padding a batch to the longest protein (35,213 residues)
#   would need terabytes of GPU memory (with padding, the attention table of every protein is
#   materialised); without padding, the memory-efficient attention kernel is used. Padding is
#   masked, so embeddings do not depend on the batch beyond floating-point rounding.
#   Timing: wall-clock time of the site embeddings (both parts) plus the classifier, saved as
#   results/compute/NB11_external.json (compute-records cell). If scores from an earlier run
#   exist, the new scores are compared with them before they are replaced.
# ---------------------------------------------------------------------------
model_name = 'facebook/esm2_t6_8M_UR50D' #smallest model
tokenizer = AutoTokenizer.from_pretrained(model_name)
esm_model = AutoModelForMaskedLM.from_pretrained(model_name)  # [modified] original: model = AutoModelForMaskedLM.from_pretrained(model_name)
esm_model.to(device)  # [modified] original: model.to(device)
esm_model.eval()  # [modified] original: model.eval()

batch_size = 32

def esm2_site_embeddings(sequences, positions):  # [added] sequences with their 1-based site positions in, embeddings out
    seqs = list(dict.fromkeys(sequences))  # [modified] original: seqs = list(df['sequence'].unique())
    site_rows = {}  # [added] sequence -> rows of the output that belong to it
    for k, s in enumerate(sequences):  # [added]
        site_rows.setdefault(s, []).append(k)  # [added]
    out = np.empty((len(sequences), esm_model.config.hidden_size), dtype=np.float32)  # [added]
    for i in tqdm(range(0, len(seqs), batch_size)):
        batch_seqs = seqs[i:i+batch_size]

        inputs = tokenizer(batch_seqs, return_tensors='pt', padding=True).to(device)
        with torch.no_grad():
            outputs = esm_model(**inputs, output_hidden_states=True) #** unpacks inputs  # [modified] original: outputs = model(**inputs, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1] #shape is (batch size, max length of sequence in batch + 2, embedding dimension)

        #for loop only in the rows of df where df['sequence'] is one of batch_seqs
        for seq in batch_seqs:
            for j in site_rows[seq]:  # [modified] original: for j, row in df[df['sequence'] == seq].iterrows():
                out[j] = last_hidden_states[batch_seqs.index(seq), positions[j], :].cpu().numpy()  # [modified] original: df.at[j, 'embedding'] = last_hidden_states[batch_seqs.index(seq), df.at[j, 'position_in_protein'], :].cpu().numpy()
    # [removed] writing the data frames to CSV and HDF5 files
    return out  # [added]

print(f'ESM-2 attention implementation: {esm_model.config._attn_implementation}')
external = pd.read_parquet('data_processed/external_proteins.parquet').sort_values('seq_id').reset_index(drop=True)
ext_sites = pd.DataFrame([(sid, i + 1, aa) for sid, seq in zip(external['seq_id'], external['sequence'])
                          for i, aa in enumerate(seq) if aa in 'ST'], columns=['seq_id', 'position', 'residue'])
lengths = external['sequence'].str.len()
print(f'External proteins: {len(external):,} (longer than 1,022 residues: {(lengths > 1022).sum():,}; '
      f'longest {lengths.max():,})   S/T: {len(ext_sites):,}')

seq_of = dict(zip(external['seq_id'], external['sequence']))
long_site = ext_sites['seq_id'].map(lengths.set_axis(external['seq_id']) > 1022).to_numpy()
t0 = time.perf_counter()
X_ext = np.empty((len(ext_sites), esm_model.config.hidden_size), dtype=np.float32)
X_ext[~long_site] = esm2_site_embeddings([seq_of[s] for s in ext_sites['seq_id'][~long_site]],
                                         ext_sites['position'].to_numpy()[~long_site])
batch_size = 1  # [modified] original: batch_size = 32 (proteins longer than 1,022 residues only)
X_ext[long_site] = esm2_site_embeddings([seq_of[s] for s in ext_sites['seq_id'][long_site]],
                                        ext_sites['position'].to_numpy()[long_site])
batch_size = 32
assert np.isfinite(X_ext).all()
with torch.no_grad():
    ext_sites['score'] = old_model(torch.from_numpy(X_ext).float().to(device)).squeeze(1).cpu().numpy()
save_compute('NB11', 'external', time.perf_counter() - t0, proteins=len(external), st_sites=len(ext_sites))
ext_sites['called'] = ext_sites['score'] >= 0.5
ext_sites['length_over_1022'] = long_site
if os.path.exists(f'{OUT_DIR}/external_scores.parquet'):
    previous = pd.read_parquet(f'{OUT_DIR}/external_scores.parquet')
    assert previous[['seq_id', 'position']].equals(ext_sites[['seq_id', 'position']])
    print(f'Previous run vs this run: score max |diff| {(previous["score"] - ext_sites["score"]).abs().max():.1e}   '
          f'calls changed {int((previous["called"] != ext_sites["called"]).sum())}')
ext_sites.to_parquet(f'{OUT_DIR}/external_scores.parquet', index=False)
print(ext_sites.groupby('length_over_1022').agg(sites=('score', 'size'), called=('called', 'sum'),
                                                mean_score=('score', 'mean')).to_string(float_format='{:.4f}'.format))
print(f'Saved {len(ext_sites):,} S/T scores: {OUT_DIR}/external_scores.parquet')
