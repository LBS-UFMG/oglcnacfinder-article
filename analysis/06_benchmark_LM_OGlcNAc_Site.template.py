# %%
import os
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))

import ast, subprocess
#@ LM-OGlcNAc-Site predict.py 9-10 shift=0

# ---------------------------------------------------------------------------
# LM-OGlcNAc-Site (Pokharel et al. 2023, Int J Mol Sci 24:16000), run with the authors'
# trained models
#   Code and models: github.com/KCLabMTU/LM-OGlcNAc-Site, pinned to its last commit.
#   Evaluated on our test set: every S/T of the test proteins except masked residues.
# ---------------------------------------------------------------------------
REPO_URL = 'https://github.com/KCLabMTU/LM-OGlcNAc-Site'
REPO_COMMIT = 'ff032df049c75812a9ad070580c60ea19cb8d941'   # 2024-06-28, "Update predict.py"
REPO = '/content/LM-OGlcNAc-Site'
OUT_DIR = 'data_processed/other_tools/LM-OGlcNAc-Site'
MAX_LEN = 1021   # longest sequence predict.py passes to the pLMs
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(REPO):
    subprocess.run(f'git clone -q {REPO_URL} {REPO}', shell=True, check=True)
subprocess.run(f'git -C {REPO} checkout -q {REPO_COMMIT}', shell=True, check=True)

# ---------------------------------------------------------------------------
# Window for proteins longer than 1,021 residues
#   predict.py calls `sequence = get_1021_string(sequence)`, which stores the returned pair
#   (site, window) as the sequence, so long proteins fail. We call the function once per
#   site instead, as the paper describes (a window centred on the site, section 4.2.2):
#   sites within 510 residues of an end use the first or last 1,021 residues. The same
#   window feeds all three pLMs, as in predict.py. The paper states 1,023 residues
#   (511 per side); the code uses 1,021 (510 per side), and we follow the code.
#   The function places the target at `new_site` only when given a 0-based index, so it is
#   called with the 0-based index (checked below for every site).
#   Its source is taken from the repository's predict.py and executed unchanged
#   (predict.py cannot be imported: it loads all models and predicts at import).
# ---------------------------------------------------------------------------
source = open(f'{REPO}/predict.py').read()
node = next(n for n in ast.parse(source).body
            if isinstance(n, ast.FunctionDef) and n.name == 'get_1021_string')
exec(ast.get_source_segment(source, node))

proteins = pd.read_parquet('data_processed/atlas5_proteins.parquet')
test = proteins[proteins['split'] == 'test']

rows, n_masked = [], 0
for pid, seq, positive, masked in zip(test['protein_id'], test['sequence'], test['positive'], test['masked']):
    positive, masked = set(positive), set(masked)
    for i, aa in enumerate(seq):
        if aa not in 'ST':
            continue
        if i + 1 in masked:
            n_masked += 1
            continue
        if len(seq) > MAX_LEN:
            new_site, window = get_1021_string(seq, i)
        else:
            new_site, window = i, seq
        start = i - new_site
        assert len(window) <= MAX_LEN and window == seq[start:start + len(window)] and window[new_site] == aa
        rows.append({'protein_id': pid, 'position': i + 1, 'residue': aa, 'label': int(i + 1 in positive),
                     'window_start': start, 'window_length': len(window), 'site_in_window': new_site})
sites = pd.DataFrame(rows)

# Same test sites as the embeddings notebook
reference = pd.read_parquet('data_processed/embeddings/sites.parquet', columns=['split', 'label'])
reference = reference[(reference['split'] == 'test') & (reference['label'] != 'masked')]
assert len(sites) == len(reference) and sites['label'].sum() == (reference['label'] == 'positive').sum()

sites.to_parquet(f'{OUT_DIR}/test_sites.parquet', index=False)

n_pos = sites['label'].sum()
print(f'Test proteins: {len(test):,}   S/T sites scored: {len(sites):,} '
      f'({n_pos:,} positive, {len(sites) - n_pos:,} negative)   masked S/T excluded: {n_masked:,}')

# Forward passes: one per distinct window (sites sharing a window share its embeddings)
long_ids = set(test.loc[test['length'] > MAX_LEN, 'protein_id'])
group = lambda df: np.where(df['protein_id'].isin(long_ids), f'length > {MAX_LEN}', f'length <= {MAX_LEN}')
windows = sites.drop_duplicates(['protein_id', 'window_start'])
summary = (sites.groupby(group(sites)).agg(proteins=('protein_id', 'nunique'), sites=('label', 'size'),
                                           positive=('label', 'sum'))
           .join(windows.groupby(group(windows)).agg(**{'forward passes per pLM': ('window_length', 'size'),
                                                         'residues per pLM': ('window_length', 'sum')})))
summary.loc['total'] = summary.sum()
print('\n' + summary.to_string(formatters={c: '{:,}'.format for c in summary.columns}))

# %%
import faulthandler, sys
faulthandler.enable(file=sys.__stderr__, all_threads=True)

# Set before TensorFlow is first imported: tf.keras resolves to Keras 2 (tf-keras), so their
# .h5 networks load with tensorflow.keras.models.load_model as written; TensorFlow allocates GPU
# memory as needed, leaving room for the three PyTorch pLMs.
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from importlib.metadata import version
# Versions: transformers 4.57.6 and ankh 1.10.0 as in NB10 (ankh 1.10.0 requires transformers < 5);
# fair-esm 2.0.0 and ankh 1.10.0 as in their requirements.txt; tf-keras provides Keras 2, which their
# .h5 networks need (current TensorFlow ships Keras 3).
TF_MINOR = version('tensorflow').rsplit('.', 1)[0]
subprocess.run(f'pip install -q "transformers==4.57.6" "ankh==1.10.0" "fair-esm==2.0.0" "tf-keras~={TF_MINOR}.0" biopython',
               shell=True, check=True)

import gc, hashlib, json, re
# Import order as in scripts/lm_oglcnac_site_external.py: PyTorch, transformers (without TensorFlow),
# fair-esm, ankh and sentencepiece before TensorFlow. With TensorFlow imported first, loading the ProtT5
# tokenizer aborted Python (Colab runtime 2026.07: std::system_error in sentencepiece LoadFromFile).
os.environ['USE_TF'] = '0'   # transformers: do not import TensorFlow
import torch
import torch._dynamo   # loads triton before TensorFlow
import sentencepiece
import transformers, esm, ankh
from transformers import T5EncoderModel, T5Tokenizer
import tensorflow as tf
from tensorflow.keras.models import load_model
assert transformers.__version__ == '4.57.6'
assert torch.cuda.is_available(), 'this notebook needs a GPU runtime'

# Full float32 precision on the GPU (their predict.py runs on the CPU)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
tf.config.experimental.enable_tensor_float_32_execution(False)

# Their three trained networks (repository commit ff032df)
MODEL_FILES = {'ankh': 'models/ankh_model_ann.h5', 'esm2': 'models/esm3B_model_ann.h5',
               'prott5': 'models/protT5_model_ann.h5'}
MODEL_SHA256 = {'ankh': '0526d7f22986734aa2e8ddbb14c628bdec2c187eb03de09b665a41e986a8988b',
                'esm2': '26808225d0df32b70bfbc02e2a27d0b79edea2abff44cf7f0ecbd1a56b48bff7',
                'prott5': 'b723ee0bd2f500f96b437e5c57159b49afcba34ffcae6095a89498fdc6f1c9f7'}
for name, path in MODEL_FILES.items():
    with open(f'{REPO}/{path}', 'rb') as f:
        assert hashlib.sha256(f.read()).hexdigest() == MODEL_SHA256[name], f'{path} differs'

# ---------------------------------------------------------------------------
# predict.py (repository commit ff032df), model loading and embedding functions
#   Their lines are kept as written; every change is tagged [modified] / [added] / [removed],
#   with the original line where one was replaced. Changes:
#     - the pLMs run on the GPU (their script hard-codes the CPU; its commented-out line selects
#       CUDA when available);
#     - the ESM-2 output is moved back to the CPU as a NumPy array;
#     - model and file paths are prefixed with the repository folder.
# ---------------------------------------------------------------------------
prot_t5_model_path = f'{REPO}/models/protT5_model_ann.h5'  # [modified] original: {{LM-OGlcNAc-Site predict.py 61}}
esm2_model_path = f'{REPO}/models/esm3B_model_ann.h5'  # [modified] original: {{LM-OGlcNAc-Site predict.py 62}}
ankh_model_path = f'{REPO}/models/ankh_model_ann.h5'  # [modified] original: {{LM-OGlcNAc-Site predict.py 63}}
#@ LM-OGlcNAc-Site predict.py 64-67 shift=0
#@ LM-OGlcNAc-Site predict.py 79-87 shift=0
device = torch.device('cuda:0')  # [modified] original: {{LM-OGlcNAc-Site predict.py 88}}
#@ LM-OGlcNAc-Site predict.py 89-95 shift=0
ankh_pretrained_model = ankh_pretrained_model.to(device)  # [added] Ankh on the GPU (their input ids are moved to `device`)
#@ LM-OGlcNAc-Site predict.py 96-101 shift=0
pretrained_model_esm = pretrained_model_esm.to(device)  # [added] ESM-2 on the GPU

#@ LM-OGlcNAc-Site predict.py 131-199 shift=0
def get_esm2_3B_features(sequence, layer=36):  # [modified] original: def get_esm2_3B_features(sequence):  (layer: see the installation check)
#@ LM-OGlcNAc-Site predict.py 201-212 shift=0
    batch_tokens = batch_tokens.to(device)  # [added] input on the GPU
#@ LM-OGlcNAc-Site predict.py 213-215 shift=0
        results = pretrained_model_esm(batch_tokens, repr_layers=[layer], return_contacts=True)  # [modified] original: {{LM-OGlcNAc-Site predict.py 216}}
    token_representations = results["representations"][layer]  # [modified] original: {{LM-OGlcNAc-Site predict.py 217}}
#@ LM-OGlcNAc-Site predict.py 218-219 shift=0
    return token_representations[:,1:-1,:][0].cpu().numpy()  # [modified] original: {{LM-OGlcNAc-Site predict.py 220}}


#@ LM-OGlcNAc-Site predict.py 225-228 shift=0

print(f'tensorflow {tf.__version__}   transformers {transformers.__version__}   torch {torch.__version__}   '
      f'ankh {version("ankh")}   fair-esm {version("fair-esm")}   GPU: {torch.cuda.get_device_name()}')
print(f'GPU memory allocated by PyTorch after loading the three pLMs: {torch.cuda.memory_allocated() / 1e9:.1f} GB')
with open(f'{OUT_DIR}/provenance.json', 'w') as f:
    json.dump({'repository': REPO_URL, 'commit': REPO_COMMIT, 'model_sha256': MODEL_SHA256,
               'tensorflow': tf.__version__, 'transformers': transformers.__version__, 'torch': torch.__version__,
               'ankh': version('ankh'), 'fair-esm': version('fair-esm'), 'gpu': torch.cuda.get_device_name()},
              f, indent=1)

# %%
import time
from Bio import SeqIO
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# predict.py main loop (repository commit ff032df), as a function
#   Their lines are kept as written; every change is tagged [modified] / [added] / [removed],
#   with the original line where one was replaced. Changes:
#     - input FASTA, output data frame and ESM-2 layer passed as arguments;
#     - the S/T embedding rows are also returned (used for the network timing below);
#     - the CSV writing is left to the caller.
# ---------------------------------------------------------------------------
def run_predict(input_fasta_file, esm_layer=36):  # [added] their main loop as a function
#@ LM-OGlcNAc-Site predict.py 70-71 shift=4
    st_rows = {'ankh': [], 'esm2': [], 'prott5': []}  # [added] S/T embedding rows, in output order
#@ LM-OGlcNAc-Site predict.py 229-243 shift=4
        esm_all = get_esm2_3B_features(sequence, layer=esm_layer)  # [modified] original: {{LM-OGlcNAc-Site predict.py 244}}
#@ LM-OGlcNAc-Site predict.py 245-258 shift=4
                st_rows['ankh'].append(X_test_ankh); st_rows['esm2'].append(X_test_esm); st_rows['prott5'].append(X_test_pt5)  # [added]
#@ LM-OGlcNAc-Site predict.py 259-270 shift=4
    # [removed] results_df.to_csv(output_csv_file, index = False) and the two print lines
    return results_df, {k: np.stack(v) for k, v in st_rows.items()}  # [added]

# ---------------------------------------------------------------------------
# Installation check: their example input and output
#   input/sequence.fasta (2 proteins, both <= 1,021 residues) and output/results.csv (59 S/T
#   sites), as released. results.csv was last written at commit e0fa511 (2024-04-24), when
#   predict.py took ESM-2 layer 33; the released predict.py (ff032df) takes layer 36, the last
#   layer. Both layers are run: the Ankh and ProtT5 columns do not depend on the layer.
#   Their output was computed on the CPU and is printed at float32 precision.
# ---------------------------------------------------------------------------
PROB_COLS = {'ankh': 'ankh_prob(Th = 0.496)', 'prott5': 'prot_t5_prob(Th = 0.669)', 'esm2': 'esm2_prob(Th = 0.797)'}
expected = pd.read_csv(f'{REPO}/output/results.csv')

runs = {}
for layer in (36, 33):
    runs[layer], st = run_predict(f'{REPO}/input/sequence.fasta', esm_layer=layer)
    assert (runs[layer]['prot_desc'].values == expected['prot_desc'].values).all()
    assert (runs[layer]['position'].astype(int).values == expected['position'].values).all()

rows = []
for layer, res in runs.items():
    for name, col in PROB_COLS.items():
        diff = np.abs(res[col].astype(float).values - expected[col].values)
        rows.append({'ESM-2 layer': layer, 'column': col, 'max |diff|': diff.max(), 'mean |diff|': diff.mean()})
    rows.append({'ESM-2 layer': layer, 'column': 'final_prediction', 'max |diff|': np.nan, 'mean |diff|': np.nan,
                 'calls differing': int((res['final_prediction'].astype(int).values != expected['final_prediction'].values).sum())})
check = pd.DataFrame(rows)
check.to_csv(f'{OUT_DIR}/installation_check.csv', index=False)
print(f'Their example: {len(expected)} S/T sites, {expected["final_prediction"].sum()} called positive\n')
fmt = lambda f: (lambda v: '' if pd.isna(v) else f.format(v))
print(check.to_string(index=False, formatters={'max |diff|': fmt('{:.2e}'), 'mean |diff|': fmt('{:.2e}'),
                                               'calls differing': fmt('{:.0f}')}))

# ---------------------------------------------------------------------------
# Timing, for the estimate of the full run
#   pLMs: one 1,021-residue window of a test protein (the length of ~98% of the test windows),
#   each pLM timed separately (median of 3 passes after one warm-up pass).
#   Networks: their per-site predict() calls (one call per site and network), on the S/T rows of
#   the example run above.
# ---------------------------------------------------------------------------
def gpu_seconds(fn, seq, n=3):
    fn(seq); torch.cuda.synchronize()
    times = []
    for _ in range(n):
        t0 = time.time(); fn(seq); torch.cuda.synchronize(); times.append(time.time() - t0)
    return float(np.median(times))

seq_of = dict(zip(test['protein_id'], test['sequence']))
w = sites[sites['window_length'] == MAX_LEN].iloc[0]
window = seq_of[w['protein_id']][w['window_start']:w['window_start'] + MAX_LEN]
t_plm = {'ProtT5': gpu_seconds(get_protT5_features, window), 'Ankh': gpu_seconds(get_ankh_features, window),
         'ESM-2 3B': gpu_seconds(get_esm2_3B_features, window)}

windows = sites.drop_duplicates(['protein_id', 'window_start'])
n_full = int((windows['window_length'] == MAX_LEN).sum())
per_window = sum(t_plm.values())
plm_hours = per_window * windows['window_length'].sum() / MAX_LEN / 3600   # time scaled by residues

NETS = {'ankh': (ankh_model_ann, 1536), 'esm2': (esm2_model_ann, 2560), 'prott5': (prot_t5_model_ann, 1024)}
t0 = time.time()
for name, (net, dim) in NETS.items():
    for x in st[name]:
        net.predict(np.array(x.reshape(1, dim)), verbose = 0)
per_site = (time.time() - t0) / len(st['ankh'])   # three predict() calls per site
net_hours = per_site * len(sites) / 3600

print(f'\npLM seconds per 1,021-residue window: ' + '   '.join(f'{k} {v:.2f}' for k, v in t_plm.items())
      + f'   total {per_window:.2f}')
print(f'Windows: {len(windows):,} ({n_full:,} of {MAX_LEN:,} residues)   estimated pLM time: {plm_hours:.1f} h')
print(f'Networks, their per-site predict(): {per_site * 1000:.0f} ms per site (3 calls)   '
      f'estimated for {len(sites):,} sites: {net_hours:.1f} h')

# %%
import shutil, time

# ---------------------------------------------------------------------------
# ESM-2 3B with two layers from one forward pass (decided: layers 33 and 36)
#   get_esm2_3B_features (cell 2) returning the representations of several layers. The forward
#   pass does not depend on which layers are returned; checked below against the single-layer
#   function for one window.
# ---------------------------------------------------------------------------
ESM_LAYERS = (33, 36)

def get_esm2_3B_features_layers(sequence, layers=ESM_LAYERS):  # [added] get_esm2_3B_features with several layers
#@ LM-OGlcNAc-Site predict.py 208-212 shift=0
    batch_tokens = batch_tokens.to(device)  # [added] input on the GPU

#@ LM-OGlcNAc-Site predict.py 215-215 shift=0
        results = pretrained_model_esm(batch_tokens, repr_layers=list(layers), return_contacts=True)  # [modified] original: {{LM-OGlcNAc-Site predict.py 216}}

    # return only residue level embeddings
    return {layer: results["representations"][layer][:,1:-1,:][0].cpu().numpy() for layer in layers}  # [modified] original: token_representations = results["representations"][36]; return token_representations[:,1:-1,:][0]

# ---------------------------------------------------------------------------
# S/T embeddings of the test sites
#   One forward pass per window and pLM, as in predict.py (no batching); sites sharing a window
#   share its pass. Only the S/T rows of the sites are kept: Ankh (1,536), ProtT5 (1,024) and
#   ESM-2 layers 33 and 36 (2,560 each), float32, with `site_row` = row of `sites` (cell 1).
#   Windows are processed in chunks of CHUNK; each chunk is written to Drive when complete
#   (written under a temporary name, then renamed), and chunks already on Drive are skipped, so
#   the cell resumes after a disconnection.
#   Timing: wall-clock seconds of each chunk's embedding loop (the three pLMs, excluding the
#   writes to Drive), saved next to the chunk as chunk_NNN.json with the GPU name; the total is
#   the sum over chunks, so it survives disconnections. Their predict.py runs the pLMs on the
#   CPU; these times are for the GPU setup used here.
# ---------------------------------------------------------------------------
EMB_DIR = f'{OUT_DIR}/embeddings'
CHUNK = 500
os.makedirs(EMB_DIR, exist_ok=True)

seq_of = dict(zip(test['protein_id'], test['sequence']))
windows = (sites.drop_duplicates(['protein_id', 'window_start'])[['protein_id', 'window_start', 'window_length']]
           .reset_index(drop=True))
rows_of = sites.groupby(['protein_id', 'window_start']).indices
site_in_window = sites['site_in_window'].to_numpy()
n_chunks = -(-len(windows) // CHUNK)
chunk_path = lambda c: f'{EMB_DIR}/chunk_{c:03d}.npz'
timing_path = lambda c: f'{EMB_DIR}/chunk_{c:03d}.json'

w = windows.iloc[0]
window = seq_of[w['protein_id']][w['window_start']:w['window_start'] + w['window_length']]
both = get_esm2_3B_features_layers(window)
for layer in ESM_LAYERS:
    diff = np.abs(both[layer] - get_esm2_3B_features(window, layer=layer)).max()
    print(f'ESM-2 layer {layer}, two-layer vs single-layer function (window of {len(window):,} residues): '
          f'max |diff| {diff:.2e}')
    assert diff < 1e-5

done = [c for c in range(n_chunks) if os.path.exists(chunk_path(c))]
bar = tqdm(total=len(windows), initial=sum(len(windows.iloc[c * CHUNK:(c + 1) * CHUNK]) for c in done), unit='window')
for c in range(n_chunks):
    if os.path.exists(chunk_path(c)):
        continue
    out = {'site_row': [], 'ankh': [], 'prott5': [], **{f'esm2_layer{l}': [] for l in ESM_LAYERS}}
    chunk_windows = windows.iloc[c * CHUNK:(c + 1) * CHUNK]
    t0 = time.perf_counter()
    for w in chunk_windows.itertuples():
        window = seq_of[w.protein_id][w.window_start:w.window_start + w.window_length]
        pt5_all = get_protT5_features(window)
        ankh_all = get_ankh_features(window)
        esm_all = get_esm2_3B_features_layers(window)
        assert pt5_all.shape == (len(window), 1024) and ankh_all.shape == (len(window), 1536)
        assert all(esm_all[l].shape == (len(window), 2560) for l in ESM_LAYERS)
        rows = rows_of[(w.protein_id, w.window_start)]
        idx = site_in_window[rows]
        assert all(window[i] in 'ST' for i in idx)
        out['site_row'].append(rows)
        out['ankh'].append(ankh_all[idx])
        out['prott5'].append(pt5_all[idx])
        for l in ESM_LAYERS:
            out[f'esm2_layer{l}'].append(esm_all[l][idx])
        bar.update(1)
    seconds = time.perf_counter() - t0
    tmp = f'/content/chunk_{c:03d}.npz'
    np.savez(tmp, **{k: np.concatenate(v) for k, v in out.items()})
    shutil.move(tmp, chunk_path(c) + '.part')
    with open(timing_path(c), 'w') as f:   # written before the chunk is complete: a chunk on Drive has its time
        json.dump({'seconds': round(seconds, 2), 'windows': len(chunk_windows),
                   'gpu': torch.cuda.get_device_name()}, f)
    os.replace(chunk_path(c) + '.part', chunk_path(c))
bar.close()

# Every site embedded exactly once
covered = np.concatenate([np.load(chunk_path(c))['site_row'] for c in range(n_chunks)])
assert np.array_equal(np.sort(covered), np.arange(len(sites)))
size = sum(os.path.getsize(chunk_path(c)) for c in range(n_chunks))
print(f'S/T embeddings of {len(sites):,} sites in {n_chunks} chunks ({size / 1e9:.1f} GB): {EMB_DIR}')

# Embedding time: sum over chunks
timings = [json.load(open(timing_path(c))) for c in range(n_chunks) if os.path.exists(timing_path(c))]
embed_seconds = sum(t['seconds'] for t in timings)
embed_gpus = sorted({t['gpu'] for t in timings})
print(f'pLM embedding time: {embed_seconds / 3600:.2f} h for {sum(t["windows"] for t in timings):,} windows '
      f'({len(timings)} of {n_chunks} chunks timed; GPU: {", ".join(embed_gpus)})')
if len(timings) < n_chunks:
    print(f'{n_chunks - len(timings)} chunks were embedded before timing was added: remove {EMB_DIR} '
          'and rerun this cell for the full time')

# %%
import time
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

# ---------------------------------------------------------------------------
# The three networks applied once to all saved S/T embeddings
#   predict.py calls predict() once per site and network (~2.3 h for our test set); the networks
#   score each row independently, so all rows are passed in one predict() call per network.
#   Checked on their example (cell 3): batched vs per-site outputs.
# ---------------------------------------------------------------------------
NETWORKS = {'ankh': ankh_model_ann, 'prott5': prot_t5_model_ann, 'esm2': esm2_model_ann}
for name, net in NETWORKS.items():
    batched = net.predict(st[name], verbose=0)[:, 0]
    per_site = runs[33][PROB_COLS[name]].astype(float).to_numpy()   # st holds the layer-33 ESM-2 rows
    diff = np.abs(batched - per_site).max()
    print(f'{name}: batched vs per-site predict() on their example, max |diff| {diff:.2e}')
    assert diff < 1e-5

# ---------------------------------------------------------------------------
# Scores of our test sites
#   Per network: its probability; a network votes positive when its probability exceeds its
#   threshold (Ankh 0.496, ESM-2 0.797, ProtT5 0.669; strict >, as in predict.py).
#   LM-OGlcNAc-Site call: >= 2 of 3 votes (decision-level fusion). Continuous score for AUROC and
#   AUPRC: mean of the three probabilities (score-level fusion); their Table 2 reports the same
#   ROC-AUC, 0.8116, for decision-level and three-model score-level fusion.
#   Computed with ESM-2 layer 33 and with layer 36 (decided: both).
#   Timing: wall-clock seconds of the four predict() calls; saved with the embedding time of the
#   previous cell in compute_test.json. If scores from an earlier run exist, the new scores are
#   compared with them before they are replaced.
# ---------------------------------------------------------------------------
DIMS = {'ankh': 1536, 'prott5': 1024, **{f'esm2_layer{l}': 2560 for l in ESM_LAYERS}}
X = {k: np.full((len(sites), d), np.nan, np.float32) for k, d in DIMS.items()}
for c in range(n_chunks):
    with np.load(chunk_path(c)) as chunk:
        for k in X:
            X[k][chunk['site_row']] = chunk[k]
assert not any(np.isnan(x).any() for x in X.values())

scores = sites[['protein_id', 'position', 'residue', 'label']].copy()
t0 = time.perf_counter()
scores['ankh_prob'] = NETWORKS['ankh'].predict(X['ankh'], batch_size=1024, verbose=0)[:, 0]
scores['prott5_prob'] = NETWORKS['prott5'].predict(X['prott5'], batch_size=1024, verbose=0)[:, 0]
for l in ESM_LAYERS:
    scores[f'esm2_layer{l}_prob'] = NETWORKS['esm2'].predict(X[f'esm2_layer{l}'], batch_size=1024, verbose=0)[:, 0]
network_seconds = time.perf_counter() - t0
for l in ESM_LAYERS:
    votes = ((scores['ankh_prob'] > cutoff_threshold_ankh).astype(int)
             + (scores[f'esm2_layer{l}_prob'] > cutoff_threshold_esm).astype(int)
             + (scores['prott5_prob'] > cutoff_threshold_prott5).astype(int))
    scores[f'call_layer{l}'] = (votes > 1).astype(int)
    scores[f'score_layer{l}'] = scores[['ankh_prob', 'prott5_prob', f'esm2_layer{l}_prob']].mean(axis=1)

ours = pd.read_parquet('data_processed/final_model/test_scores.parquet')[['protein_id', 'position', 'probability']]
OUR_CUTOFF = torch.load('data_processed/final_model/final_model.pt', map_location='cpu')['cutoff']['probability']
scores = scores.merge(ours, on=['protein_id', 'position'], how='left', validate='one_to_one')
assert scores['probability'].notna().all()
SCORE_COLS = (['ankh_prob', 'prott5_prob'] + [f'esm2_layer{l}_prob' for l in ESM_LAYERS]
              + [f'score_layer{l}' for l in ESM_LAYERS])
if os.path.exists(f'{OUT_DIR}/test_scores.parquet'):
    previous = pd.read_parquet(f'{OUT_DIR}/test_scores.parquet')
    assert previous[['protein_id', 'position']].equals(scores[['protein_id', 'position']])
    diff = (previous[SCORE_COLS] - scores[SCORE_COLS]).abs().max()
    print('Previous run vs this run, max |diff| per score: '
          + ', '.join(f'{c} {d:.1e}' for c, d in diff.items()))
scores.to_parquet(f'{OUT_DIR}/test_scores.parquet', index=False)

compute = {'windows': len(windows), 'sites': len(sites), 'gpu': torch.cuda.get_device_name(),
           'embedding_seconds': round(embed_seconds, 1), 'embedding_chunks_timed': len(timings),
           'chunks': n_chunks, 'embedding_gpus': embed_gpus, 'network_seconds': round(network_seconds, 1)}
with open(f'{OUT_DIR}/compute_test.json', 'w') as f:
    json.dump(compute, f, indent=1)
print(f'Compute on our test set: pLM embeddings {embed_seconds / 3600:.2f} h '
      f'({len(timings)} of {n_chunks} chunks timed), networks {network_seconds:.1f} s; GPU {compute["gpu"]}')

# ---------------------------------------------------------------------------
# Metrics on all test sites (no leakage filter: their training data are not released)
# ---------------------------------------------------------------------------
y_true = scores['label'].to_numpy()
def metrics(name, score, called):
    tp = int((called & (y_true == 1)).sum())
    return {'model': name, 'AUROC': roc_auc_score(y_true, score), 'AUPRC': average_precision_score(y_true, score),
            'MCC': matthews_corrcoef(y_true, called), 'precision': tp / max(called.sum(), 1),
            'recall': tp / y_true.sum(), 'called positive': int(called.sum())}

rows = []
for l in ESM_LAYERS:
    rows.append(metrics(f'LM-OGlcNAc-Site (ESM-2 layer {l})', scores[f'score_layer{l}'].to_numpy(),
                        scores[f'call_layer{l}'].to_numpy() == 1))
rows.append(metrics('Our model', scores['probability'].to_numpy(), scores['probability'].to_numpy() >= OUR_CUTOFF))
rows.append(metrics('  Ankh network', scores['ankh_prob'].to_numpy(), scores['ankh_prob'].to_numpy() > cutoff_threshold_ankh))
rows.append(metrics('  ProtT5 network', scores['prott5_prob'].to_numpy(), scores['prott5_prob'].to_numpy() > cutoff_threshold_prott5))
for l in ESM_LAYERS:
    p = scores[f'esm2_layer{l}_prob'].to_numpy()
    rows.append(metrics(f'  ESM-2 network, layer {l}', p, p > cutoff_threshold_esm))
table = pd.DataFrame(rows)
table.to_csv(f'{OUT_DIR}/test_metrics.csv', index=False)
print(f'\nAll test sites: {len(scores):,} ({y_true.sum():,} positive, fraction {y_true.mean():.4f})')
print(table.to_string(index=False, float_format='{:.4f}'.format))

# %%
import urllib.request

# ---------------------------------------------------------------------------
# ESM-2 layer check against their published predictions: human galectins (Table 5)
#   Pokharel et al. 2023 (Int J Mol Sci 24:16000, Table 5) list the sites LM-OGlcNAc-Site
#   predicted in 12 human galectins, with the NCBI RefSeq protein of each. The paper predates
#   the change of predict.py from ESM-2 layer 33 to layer 36 (commit ff032df, 2024-06-28).
#   The sequences are retrieved from NCBI at the listed versions; each protein (all <= 1,021
#   residues) is embedded whole and scored as in cells 4-5, with layer 33 and with layer 36.
#   A site is called when >= 2 of the 3 networks exceed their thresholds, as in predict.py.
# ---------------------------------------------------------------------------
TABLE5 = {   # galectin: (NCBI RefSeq protein, sites predicted by LM-OGlcNAc-Site in Table 5)
    'Galectin-1': ('NP_002296.1', 'T71, S84'),
    'Galectin-2': ('NP_006489.1', 'T21, S80, T85'),
    'Galectin-3': ('NP_002297.2', 'S6, S12, S14, S40, S84, S91, S92, S96, T98, T104'),
    'Galectin-4': ('NP_006140.1', 'S86, S92, T217, S258'),
    'Galectin-7': ('NP_002298.1', 'S2, T18, S31, S69'),
    'Galectin-8': ('NP_006490.3', 'S4, T160, S171, S178, T180, S192, T198, T201, S203, T207, T211, T215'),
    'Galectin-9': ('NP_033665.1', 'S4, S6, T152, S160, S161, S165, T193, T195'),
    'Galectin-10': ('NP_001819.2', 'S13'),
    'Galectin-12': ('NP_001136007.2', 'S2, S221'),
    'Galectin-13': ('NP_037400.1', 'S2, S13'),
    'Galectin-14': ('NP_064514.1', 'S2, S13, S138'),
    'Galectin-16': ('NP_001177370.2', 'S2, S13'),
}
ids = ','.join(acc for acc, _ in TABLE5.values())
url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=protein&id={ids}&rettype=fasta&retmode=text'
with urllib.request.urlopen(url, timeout=60) as r:
    fasta = r.read().decode()
seq_by_acc = {}
for record in fasta.strip().split('>')[1:]:
    header, *lines = record.strip().split('\n')
    seq_by_acc[header.split()[0]] = ''.join(lines)
assert set(seq_by_acc) == {acc for acc, _ in TABLE5.values()}
with open(f'{OUT_DIR}/galectins_table5.fasta', 'w') as f:
    f.write(fasta)

# Sites are compared by position; a published residue letter that differs from the sequence is
# listed in the last column.
rows = []
for galectin, (acc, published) in TABLE5.items():
    seq = seq_by_acc[acc]
    assert len(seq) <= MAX_LEN
    label = lambda positions: ', '.join(f'{seq[p - 1]}{p}' for p in sorted(positions))
    not_st = [s for s in published.split(', ') if seq[int(s[1:]) - 1] != s[0]]
    published = {int(s[1:]) for s in published.split(', ')}
    idx = np.array([i for i, aa in enumerate(seq) if aa in 'ST'])
    p_ankh = ankh_model_ann.predict(get_ankh_features(seq)[idx], verbose=0)[:, 0]
    p_t5 = prot_t5_model_ann.predict(get_protT5_features(seq)[idx], verbose=0)[:, 0]
    esm_all = get_esm2_3B_features_layers(seq)
    for l in ESM_LAYERS:
        p_esm = esm2_model_ann.predict(esm_all[l][idx], verbose=0)[:, 0]
        votes = ((p_ankh > cutoff_threshold_ankh).astype(int) + (p_esm > cutoff_threshold_esm).astype(int)
                 + (p_t5 > cutoff_threshold_prott5).astype(int))
        called = {i + 1 for i, v in zip(idx, votes) if v > 1}
        rows.append({'galectin': galectin, 'accession': acc, 'length': len(seq), 'ESM-2 layer': l,
                     'published': len(published), 'called': len(called), 'in both': len(called & published),
                     'published only': label(published - called), 'called only': label(called - published),
                     'published letter differs': ', '.join(not_st)})

galectins = pd.DataFrame(rows)
galectins.to_csv(f'{OUT_DIR}/galectin_check.csv', index=False)
for l in ESM_LAYERS:
    g = galectins[galectins['ESM-2 layer'] == l]
    exact = int(((g['published only'] == '') & (g['called only'] == '')).sum())
    print(f'\nESM-2 layer {l}: proteins with the published calls reproduced exactly: {exact} of {len(g)}   '
          f'sites published {g["published"].sum()}, called {g["called"].sum()}, in both {g["in both"].sum()}')
    print(g.drop(columns=['ESM-2 layer', 'accession']).to_string(index=False))

# %%
import torch
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

# Scores saved by the network cell (GPU session)
scores = pd.read_parquet(f'{OUT_DIR}/test_scores.parquet')
OUR_CUTOFF = torch.load('data_processed/final_model/final_model.pt', map_location='cpu')['cutoff']['probability']
print(f'{len(scores):,} scored sites reloaded   our cut-off {OUR_CUTOFF:.4f}')

# Atlas 4.0 files from the interrupted run: remove any incomplete download
ATLAS4_SIZES = {'Atlas 4.0_unambiguous sites_20241217.csv': 6102722, 'Atlas 4.0_ambiguous sites_20241217.csv': 1895593}
for name, size in ATLAS4_SIZES.items():
    path = f'data_processed/other_tools/Atlas4.0/{name}'
    if os.path.exists(path) and os.path.getsize(path) != size:
        os.remove(path)
        print(f'removed incomplete {name}')
    print(f'{name}: {"present" if os.path.exists(path) else "will be downloaded"}')

# %%
import json, re, time, urllib.error, urllib.parse, urllib.request

# ---------------------------------------------------------------------------
# Overlap of our test proteins with LM-OGlcNAc-Site's training data (decision 5b)
#   Their training proteins (4,826; O-GlcNAcAtlas 2.0, proteins with unambiguous sites; paper
#   section 4.1) are not released, and Atlas 2.0 is not offered for download. The archived
#   Atlas 4.0 release (oglcnac.org, files dated 2024-12-17) is used as a superset: every protein
#   of its unambiguous and ambiguous datasets, all species (assumption: Atlas 2.0 proteins are
#   contained in Atlas 4.0). Test proteins not overlapping Atlas 4.0 cannot be among their
#   training proteins under that assumption; proteins added to the Atlas after 2.0 are also
#   removed, which reduces the subset but does not favour either model.
#   Sequences, by accession source (as in NB01): UniProt accessions from UniProt (current
#   versions; deleted entries from UniSave at their last version); rice MSU IDs from the MSU v7
#   proteome (NB01); NCBI GI and GenBank accessions from NCBI; FlyBase genes from the UniProt
#   entries cross-referenced to them; PDB chains are not used (as in NB01).
#   Overlap, each removing whole test proteins (as for the other tools):
#     (a) homology: identity >= 30% to any Atlas 4.0 protein, identity = identical residues /
#         length of the shorter protein, exhaustive MMseqs2 alignment (fake prefilter,
#         --alignment-mode 3, E <= 0.001; E-values relative to the Atlas 4.0 proteins);
#     (b) same protein: accession (isoform suffix ignored) or exact sequence in Atlas 4.0.
#   Files shared with NB07 (O-GlcNAcPRED-DL, also trained on Atlas 2.0) in ATLAS4_DIR.
# ---------------------------------------------------------------------------
ATLAS4_DIR = 'data_processed/other_tools/Atlas4.0'
ATLAS4_FILES = ['Atlas 4.0_unambiguous sites_20241217.csv', 'Atlas 4.0_ambiguous sites_20241217.csv']
os.makedirs(ATLAS4_DIR, exist_ok=True)
for name in ATLAS4_FILES:   # downloaded with curl (the server refuses Python's default client)
    if not os.path.exists(f'{ATLAS4_DIR}/{name}'):
        subprocess.run(['curl', '-sSfL', '-o', f'{ATLAS4_DIR}/{name}',
                        'https://oglcnac.org/static/dataset/' + urllib.parse.quote(name)], check=True)
atlas4 = pd.concat([pd.read_csv(f'{ATLAS4_DIR}/{name}', dtype=str).assign(dataset=name.split('_')[1].split()[0])
                    for name in ATLAS4_FILES])
atlas4['source'] = atlas4['accession_source'].str.lower()
accessions = atlas4[['source', 'accession']].dropna().drop_duplicates()
print(f'Atlas 4.0: {len(atlas4):,} site records, {len(accessions):,} distinct accessions')
print(accessions['source'].value_counts().to_string())

def read_fasta_text(text, key=lambda header: header.split()[0]):
    seqs, name = {}, None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('>'):
            name = key(line[1:])
            seqs[name] = ''
        elif name:
            seqs[name] += line
    return seqs

def get(url, timeout=120, attempts=4):
    """Download a URL; network errors, timeouts and rate limits (HTTP 429) are retried."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read().decode()
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == attempts - 1:
                raise
            time.sleep(10 * (attempt + 1))
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(10 * (attempt + 1))

ATLAS4_FASTA = f'{ATLAS4_DIR}/atlas4_proteins.fasta'   # header: >source|accession|retrieved entry
if not os.path.exists(ATLAS4_FASTA):
    records = []   # (source, accession, retrieved entry, sequence)
    ids = lambda s: sorted(accessions.loc[accessions['source'] == s, 'accession'])
    # UniProt: canonical accessions in batches; isoforms and other identifiers one by one;
    # deleted entries from UniSave
    uniprot = ids('uniprot')
    canonical = [a for a in uniprot if re.fullmatch(r'[A-Z0-9]{6,10}', a)]
    got = {}
    for i in tqdm(range(0, len(canonical), 200), desc='UniProt (canonical, batches of 200)'):
        text = get('https://rest.uniprot.org/uniprotkb/accessions?format=fasta&accessions=' + ','.join(canonical[i:i + 200]))
        got.update(read_fasta_text(text, key=lambda h: h.split('|')[1]))
    for a in tqdm([a for a in uniprot if a not in set(canonical)], desc='UniProt (isoforms and others)'):
        try:
            got[a] = ''.join(get(f'https://rest.uniprot.org/uniprotkb/{urllib.parse.quote(a)}.fasta', 60).splitlines()[1:])
        except Exception as e:
            print(f'  {a}: {e}')
    for a in [a for a in uniprot if a not in got]:   # secondary accessions (redirected), then UniSave
        try:
            got[a] = ''.join(get(f'https://rest.uniprot.org/uniprotkb/{a}.fasta', 60).splitlines()[1:])
            assert got[a]
            continue
        except Exception:
            got.pop(a, None)
        try:
            versions = json.loads(get(f'https://rest.uniprot.org/unisave/{a}?format=json', 60))['results']
            text = get(f'https://rest.uniprot.org/unisave/{a}?format=fasta&versions={versions[0]["entryVersion"]}', 60)
            got[a] = ''.join(text.splitlines()[1:])
            print(f'  {a}: not in current UniProt; UniSave version {versions[0]["entryVersion"]}')
        except Exception as e:
            print(f'  {a}: not retrieved ({e})')
    records += [('uniprot', a, a, got[a]) for a in uniprot if a in got]
    # Rice: MSU v7 proteome (downloaded in NB01)
    msu7 = {k: v.rstrip('*') for k, v in read_fasta_text(open('data/msu7_all.pep').read()).items()}
    records += [('msu id', a, a, msu7[a]) for a in ids('msu id') if a in msu7]
    # NCBI GI numbers and GenBank accessions, one by one (NCBI allows 3 requests per second)
    for source in ('ncbi', 'genbank'):
        for a in ids(source):
            time.sleep(0.5)
            text = get(f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=protein&rettype=fasta&retmode=text&id={a}', 60)
            records += [(source, a, e, s) for e, s in read_fasta_text(text).items()]
    # FlyBase genes: every UniProt entry cross-referenced to the gene
    for a in ids('flybase'):
        text = get('https://rest.uniprot.org/uniprotkb/search?format=fasta&query=' + urllib.parse.quote(f'xref:flybase-{a}'))
        records += [('flybase', a, e, s) for e, s in read_fasta_text(text, key=lambda h: h.split('|')[1]).items()]
    with open(ATLAS4_FASTA, 'w') as f:
        for source, a, e, s in records:
            f.write(f'>{source}|{a}|{e}\n{s}\n')

atlas4_seqs = read_fasta_text(open(ATLAS4_FASTA).read(), key=lambda h: h)
retrieved = {tuple(h.split('|')[:2]) for h in atlas4_seqs}
unresolved = accessions[[(s, a) not in retrieved for s, a in zip(accessions['source'], accessions['accession'])]]
print(f'\nAtlas 4.0 sequences: {len(atlas4_seqs):,}   accessions without a sequence: {len(unresolved)}')
print(unresolved.groupby('source')['accession'].apply(lambda x: ', '.join(x[:6]) + (' ...' if len(x) > 6 else '')).to_string())

# MMseqs2: every test protein against every Atlas 4.0 protein (run_streamed, write_fasta: as in NB08/NB10)
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

ALN_DIR = '/content/atlas4_overlap'
os.makedirs(ALN_DIR, exist_ok=True)
ALN_FILE = os.path.abspath(f'{ATLAS4_DIR}/test_vs_atlas4_alignments.tsv')   # MMseqs2 runs in ALN_DIR
if not os.path.exists(ALN_FILE):
    write_fasta(f'{ALN_DIR}/test.fasta', test['protein_id'], test['sequence'])
    write_fasta(f'{ALN_DIR}/atlas4.fasta', list(atlas4_seqs), list(atlas4_seqs.values()))
    run_streamed([MMSEQS, 'createdb', f'{ALN_DIR}/test.fasta', f'{ALN_DIR}/qdb'], cwd=ALN_DIR)
    run_streamed([MMSEQS, 'createdb', f'{ALN_DIR}/atlas4.fasta', f'{ALN_DIR}/tdb'], cwd=ALN_DIR)
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

# Per test protein: closest Atlas 4.0 protein, and both overlap flags
best = aln.sort_values('id_shortest', ascending=False).drop_duplicates('query').set_index('query')
overlap = test[['protein_id', 'organism', 'sequence']].copy()
overlap['max_identity'] = overlap['protein_id'].map(best['id_shortest']).fillna(0.0)
overlap['closest_atlas4_protein'] = overlap['protein_id'].map(best['target'])
base = lambda ids: pd.Series(ids).astype(str).str.split('-').str[0]
overlap['same_protein'] = (base(overlap['protein_id']).isin(set(base(accessions['accession']))).to_numpy()
                           | overlap['sequence'].isin(set(atlas4_seqs.values())).to_numpy())
overlap['homolog_30'] = overlap['max_identity'] >= 0.30
assert (overlap.loc[overlap['sequence'].isin(set(atlas4_seqs.values())), 'max_identity'] >= 0.99).all()
overlap.drop(columns='sequence').to_csv(f'{ATLAS4_DIR}/test_protein_overlap.csv', index=False)

flags = overlap.set_index('protein_id')
SUBSETS = {'all test proteins': np.ones(len(scores), bool),
           '(b) same protein removed': ~scores['protein_id'].map(flags['same_protein']).to_numpy(),
           '(a) homologs >= 30% removed': ~scores['protein_id'].map(flags['homolog_30']).to_numpy()}
MODELS = {f'LM-OGlcNAc-Site (layer {l})': (f'score_layer{l}', scores[f'call_layer{l}'].to_numpy() == 1)
          for l in (33, 36)}
MODELS['Our model'] = ('probability', scores['probability'].to_numpy() >= OUR_CUTOFF)

rows = []
for subset, keep in SUBSETS.items():
    y_sub = scores.loc[keep, 'label'].to_numpy()
    for name, (col, called_all) in MODELS.items():
        score, called = scores.loc[keep, col].to_numpy(), called_all[keep]
        tp = int((called & (y_sub == 1)).sum())
        rows.append({'subset': subset, 'proteins': scores.loc[keep, 'protein_id'].nunique(), 'sites': int(keep.sum()),
                     'positive': int(y_sub.sum()), 'model': name,
                     'AUROC': roc_auc_score(y_sub, score), 'AUPRC': average_precision_score(y_sub, score),
                     'MCC': matthews_corrcoef(y_sub, called), 'precision': tp / max(called.sum(), 1),
                     'recall': tp / y_sub.sum()})
filtered = pd.DataFrame(rows)
filtered.to_csv(f'{OUT_DIR}/test_metrics_filtered.csv', index=False)

print(f'\nTest proteins: {len(overlap)}   same protein as an Atlas 4.0 protein: {overlap["same_protein"].sum()}   '
      f'homolog >= 30%: {overlap["homolog_30"].sum()}')
print('Highest identity to Atlas 4.0 proteins (test proteins), quantiles 0/25/50/75/100%:',
      np.round(overlap['max_identity'].quantile([0, 0.25, 0.5, 0.75, 1]).to_numpy(), 2))
print()
print(filtered.to_string(index=False, float_format='{:.4f}'.format))

# %%
import os, urllib.request

# ---------------------------------------------------------------------------
# Recovered training data of LM-OGlcNAc-Site
#   The repository held its training data (data/sites/) from commit 7a2ee22 (2022-11-14,
#   "Add initial files") until commit e33f67c (2023-07-12, "Stand alone prediction version"),
#   which deleted them and added the released models. The files are downloaded from the
#   repository at commit 7a2ee22.
# ---------------------------------------------------------------------------
SNAP_COMMIT = '7a2ee22961b33ce43f4e0ec8e454ffecccb89e2e'
SNAP = f'{OUT_DIR}/training_data_git'
SNAP_FILES = ['data/sites/train_protein_ids.txt', 'data/sites/test_protein_ids.txt',
              'data/sites/our_data/train_positive_unambiguous_list_cleaned.csv',
              'data/sites/our_data/train_negative_unambiguous_list_cleaned.csv']
os.makedirs(SNAP, exist_ok=True)
for path in SNAP_FILES:
    url = f'https://raw.githubusercontent.com/KCLabMTU/LM-OGlcNAc-Site/{SNAP_COMMIT}/{path}'
    urllib.request.urlretrieve(url, f'{SNAP}/{os.path.basename(path)}')
    print(f'{os.path.basename(path)}: {os.path.getsize(f"{SNAP}/{os.path.basename(path)}"):,} bytes')

# %%
import re
from sklearn.metrics import average_precision_score, roc_auc_score

# ---------------------------------------------------------------------------
# Check: were the released models trained on the recovered data snapshot?
#   The repository held LM-OGlcNAc-Site's training data (data/sites/) from commit 7a2ee22
#   (2022-11-14) until commit e33f67c (2023-07-12), which deleted them and added the released
#   models. The files are downloaded in the previous cell. A model scores the proteins and
#   sites it was trained on better than unseen ones, so two comparisons are made:
#     (1) test proteins grouped by the snapshot's protein lists (training ids, test ids,
#         neither), both models scored per group. Our model saw none of these proteins, so its
#         score gives each group's difficulty; the signal is LM-OGlcNAc-Site's advantage over
#         ours in the training-list group compared with the other groups.
#     (2) our test positives on proteins of the snapshot's training list, split by their label
#         in the snapshot's site files: positive there, or negative there (reported after
#         Atlas 2.0). Each site's score is expressed as a percentile among all test sites of
#         the same model, so the two models can be compared.
#   Proteins are matched by UniProt accession (isoform suffix ignored) over all IDs merged into
#   each test protein; snapshot sites by accession and position.
# ---------------------------------------------------------------------------
SNAP = f'{OUT_DIR}/training_data_git'
base = lambda acc: acc.split('-')[0]
read_ids = lambda name: {base(a) for a in re.findall(r"'([^']+)'", open(f'{SNAP}/{name}').read())}
train_ids, test_ids = read_ids('train_protein_ids.txt'), read_ids('test_protein_ids.txt')
snap_sites = {label: pd.read_csv(f'{SNAP}/train_{label}_unambiguous_list_cleaned.csv', index_col=0)
              for label in ('positive', 'negative')}
print(f'Snapshot: training ids {len(train_ids):,}, test ids {len(test_ids):,}, ids in both lists '
      f'{len(train_ids & test_ids)}; positive sites {len(snap_sites["positive"]):,}, '
      f'negative sites {len(snap_sites["negative"]):,}')

scores = pd.read_parquet(f'{OUT_DIR}/test_scores.parquet')   # cell 6: theirs and ours per test site
SCORE = 'score_layer33'   # LM-OGlcNAc-Site main version (ESM-2 layer 33)
test = proteins[proteins['split'] == 'test']
accs_of = {pid: {base(a) for a in ids.split('; ')} for pid, ids in zip(test['protein_id'], test['merged_ids'])}
assert set(scores['protein_id']) <= set(accs_of)

GROUPS = ['snapshot training list', 'snapshot test list', 'neither list']
def group_of(accs):
    if accs & train_ids:
        return GROUPS[0]
    return GROUPS[1] if accs & test_ids else GROUPS[2]
scores['group'] = scores['protein_id'].map(lambda pid: group_of(accs_of[pid]))

# (1) Per protein group
rows = []
for g in GROUPS:
    sub = scores[scores['group'] == g]
    y = sub['label'].to_numpy()
    row = {'group': g, 'proteins': sub['protein_id'].nunique(), 'sites': len(sub), 'positives': int(y.sum())}
    if 0 < y.sum() < len(y):
        for name, col in [('LM-OGlcNAc-Site', SCORE), ('ours', 'probability')]:
            row[f'AUROC {name}'] = roc_auc_score(y, sub[col])
            row[f'AUPRC {name}'] = average_precision_score(y, sub[col])
        row['AUROC theirs - ours'] = row['AUROC LM-OGlcNAc-Site'] - row['AUROC ours']
    rows.append(row)
by_group = pd.DataFrame(rows)
by_group.to_csv(f'{OUT_DIR}/snapshot_check_by_group.csv', index=False)
print('\n(1) Test proteins grouped by the snapshot protein lists')
print(by_group.to_string(index=False, float_format='{:.4f}'.format))

# (2) Positives on training-list proteins, by their label in the snapshot site files
scores['percentile LM-OGlcNAc-Site'] = scores[SCORE].rank(pct=True)
scores['percentile ours'] = scores['probability'].rank(pct=True)
snap_label = {}
for label, df in snap_sites.items():
    for acc, pos in zip(df['accession'], df['position_in_protein']):
        snap_label.setdefault((base(acc), int(pos)), label)   # a site listed as both stays positive
def site_label(pid, pos):
    found = {snap_label.get((acc, pos)) for acc in accs_of[pid]} - {None}
    if 'positive' in found:
        return 'positive in snapshot'
    return 'negative in snapshot' if found else 'not in snapshot site files'
pos_train = scores[(scores['group'] == GROUPS[0]) & (scores['label'] == 1)].copy()
pos_train['snapshot label'] = [site_label(p, q) for p, q in zip(pos_train['protein_id'], pos_train['position'])]
by_site = (pos_train.groupby('snapshot label')
           .agg(sites=('position', 'size'), proteins=('protein_id', 'nunique'),
                **{'median percentile LM-OGlcNAc-Site': ('percentile LM-OGlcNAc-Site', 'median'),
                   'median percentile ours': ('percentile ours', 'median')})
           .reset_index())
by_site.to_csv(f'{OUT_DIR}/snapshot_check_by_site.csv', index=False)
print('\n(2) Our test positives on proteins of the snapshot training list, by their snapshot label '
      '(score percentile among all test sites of the same model)')
print(by_site.to_string(index=False, float_format='{:.3f}'.format))

# Snapshot sites on these proteins that do not fall on an S/T of our sequence (sequence changes)
st_of = {(pid, pos) for pid, pos in zip(scores['protein_id'], scores['position'])}
masked_of = {pid: set(m) for pid, m in zip(test['protein_id'], test['masked'])}
positions_of = {}
for acc, pos in snap_label:
    positions_of.setdefault(acc, set()).add(pos)
n_snap = n_off = 0
for pid, accs in accs_of.items():
    if group_of(accs) != GROUPS[0]:
        continue
    for pos in set().union(*(positions_of.get(acc, set()) for acc in accs)):
        n_snap += 1
        n_off += (pid, pos) not in st_of and pos not in masked_of[pid]
print(f'\nSnapshot sites on these proteins: {n_snap:,}; not on an S/T of our sequence: {n_off:,}')

# %%
import json, re, subprocess, time, urllib.error, urllib.parse, urllib.request
import torch
from tqdm.auto import tqdm
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

# ---------------------------------------------------------------------------
# Overlap of our test proteins with LM-OGlcNAc-Site's recovered training data
#   Replaces the Atlas 4.0 superset of cell 7 (whose files are kept unchanged). The data
#   snapshot (cell 8, commit 7a2ee22) holds 5,434 proteins: training list 4,752, test list 529,
#   and 153 proteins in the site files only. Which of them the released models saw is not
#   recorded (the snapshot is 8 months older than the models, and the paper reports 4,826
#   training proteins), so all 5,434 are treated as seen. Cell 9 showed that the released models
#   behave as if trained on the snapshot (advantage over our model largest on its training list).
#   Sequences: UniProt accessions from UniProt (current versions; secondary accessions followed;
#   deleted entries from UniSave at their last version). The 3 PDB chains (3J61_A, _C, _K) have
#   no sequence: entry 3J61 is no longer in the PDB (checked 2026-10-19).
#   Overlap, each removing whole test proteins (same criteria as cell 7):
#     (a) homology: identity >= 30% to any snapshot protein, identity = identical residues /
#         length of the shorter protein, exhaustive MMseqs2 alignment (fake prefilter,
#         --alignment-mode 3, E <= 0.001);
#     (b) same protein: accession (isoform suffix ignored) or exact sequence in the snapshot;
#         (a) also removes these, also when the snapshot sequence could not be retrieved.
#   New files only, in SNAP and OUT_DIR (names ending in _snapshot).
# ---------------------------------------------------------------------------
SNAP = f'{OUT_DIR}/training_data_git'
OUR_CUTOFF = torch.load('data_processed/final_model/final_model.pt', map_location='cpu')['cutoff']['probability']
scores = pd.read_parquet(f'{OUT_DIR}/test_scores.parquet')
test = proteins[proteins['split'] == 'test']

snap_accessions = set(re.findall(r"'([^']+)'", open(f'{SNAP}/train_protein_ids.txt').read()))
snap_accessions |= set(re.findall(r"'([^']+)'", open(f'{SNAP}/test_protein_ids.txt').read()))
for label in ('positive', 'negative'):
    snap_accessions |= set(pd.read_csv(f'{SNAP}/train_{label}_unambiguous_list_cleaned.csv',
                                       usecols=['accession'])['accession'])
pdb_chains = sorted(a for a in snap_accessions if re.fullmatch(r'[0-9][A-Z0-9]{3}_[A-Za-z0-9]+', a))
uniprot = sorted(snap_accessions - set(pdb_chains))
print(f'Snapshot proteins: {len(snap_accessions):,} (UniProt {len(uniprot):,}, PDB chains {len(pdb_chains)})')

def read_fasta_text(text, key=lambda header: header.split()[0]):
    seqs, name = {}, None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('>'):
            name = key(line[1:])
            seqs[name] = ''
        elif name:
            seqs[name] += line
    return seqs

def get(url, timeout=120, attempts=4):
    """Download a URL; network errors, timeouts and rate limits (HTTP 429) are retried."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read().decode()
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == attempts - 1:
                raise
            time.sleep(10 * (attempt + 1))
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(10 * (attempt + 1))

SNAP_FASTA = f'{SNAP}/snapshot_proteins.fasta'   # header: >accession
if not os.path.exists(SNAP_FASTA):
    got = {}
    canonical = [a for a in uniprot if re.fullmatch(r'[A-Z0-9]{6,10}', a)]
    for i in tqdm(range(0, len(canonical), 200), desc='UniProt (canonical, batches of 200)'):
        text = get('https://rest.uniprot.org/uniprotkb/accessions?format=fasta&accessions=' + ','.join(canonical[i:i + 200]))
        got.update(read_fasta_text(text, key=lambda h: h.split('|')[1]))
    for a in tqdm([a for a in uniprot if a not in got], desc='UniProt (isoforms, secondary, deleted)'):
        try:   # isoforms, and secondary accessions (redirected to the current entry)
            got[a] = ''.join(get(f'https://rest.uniprot.org/uniprotkb/{urllib.parse.quote(a)}.fasta', 60).splitlines()[1:])
            assert got[a]
            continue
        except Exception:
            got.pop(a, None)
        try:   # deleted entries: last UniSave version
            versions = json.loads(get(f'https://rest.uniprot.org/unisave/{a}?format=json', 60))['results']
            text = get(f'https://rest.uniprot.org/unisave/{a}?format=fasta&versions={versions[0]["entryVersion"]}', 60)
            got[a] = ''.join(text.splitlines()[1:])
            print(f'  {a}: not in current UniProt; UniSave version {versions[0]["entryVersion"]}')
        except Exception as e:
            print(f'  {a}: not retrieved ({e})')
    with open(SNAP_FASTA, 'w') as f:
        for a in sorted(got):
            f.write(f'>{a}\n{got[a]}\n')

snap_seqs = read_fasta_text(open(SNAP_FASTA).read())
missing = sorted(snap_accessions - set(snap_seqs))
print(f'Sequences retrieved: {len(snap_seqs):,}   without a sequence: {len(missing)}'
      + (f' ({", ".join(missing[:10])}{" ..." if len(missing) > 10 else ""})' if missing else ''))

# MMseqs2: every test protein against every snapshot protein (as in cell 7)
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

ALN_DIR = '/content/snapshot_overlap'
os.makedirs(ALN_DIR, exist_ok=True)
ALN_FILE = os.path.abspath(f'{SNAP}/test_vs_snapshot_alignments.tsv')   # MMseqs2 runs in ALN_DIR
if not os.path.exists(ALN_FILE):
    if not os.path.exists(MMSEQS):
        subprocess.run('wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -O /content/mmseqs.tar.gz '
                       '&& tar xzf /content/mmseqs.tar.gz -C /content', shell=True, check=True)
    print('MMseqs2 version:', subprocess.run([MMSEQS, 'version'], capture_output=True, text=True).stdout.strip())
    write_fasta(f'{ALN_DIR}/test.fasta', test['protein_id'], test['sequence'])
    write_fasta(f'{ALN_DIR}/snapshot.fasta', list(snap_seqs), list(snap_seqs.values()))
    run_streamed([MMSEQS, 'createdb', f'{ALN_DIR}/test.fasta', f'{ALN_DIR}/qdb'], cwd=ALN_DIR)
    run_streamed([MMSEQS, 'createdb', f'{ALN_DIR}/snapshot.fasta', f'{ALN_DIR}/tdb'], cwd=ALN_DIR)
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

# Per test protein: closest snapshot protein, and both overlap flags
best = aln.sort_values('id_shortest', ascending=False).drop_duplicates('query').set_index('query')
overlap = test[['protein_id', 'organism', 'merged_ids', 'sequence']].copy()
overlap['max_identity'] = overlap['protein_id'].map(best['id_shortest']).fillna(0.0)
overlap['closest_snapshot_protein'] = overlap['protein_id'].map(best['target'])
base = lambda acc: acc.split('-')[0]
snap_base = {base(a) for a in snap_accessions}
overlap['same_protein'] = (np.array([bool({base(a) for a in ids.split('; ')} & snap_base) for ids in overlap['merged_ids']])
                           | overlap['sequence'].isin(set(snap_seqs.values())).to_numpy())
overlap['homolog_30'] = (overlap['max_identity'] >= 0.30) | overlap['same_protein']   # (a) includes (b)
assert (overlap.loc[overlap['sequence'].isin(set(snap_seqs.values())), 'max_identity'] >= 0.99).all()
overlap.drop(columns='sequence').to_csv(f'{SNAP}/test_protein_overlap_snapshot.csv', index=False)

flags = overlap.set_index('protein_id')
SUBSETS = {'all test proteins': np.ones(len(scores), bool),
           '(b) same protein removed': ~scores['protein_id'].map(flags['same_protein']).to_numpy(bool),
           '(a) homologs >= 30% removed': ~scores['protein_id'].map(flags['homolog_30']).to_numpy(bool)}
MODELS = {f'LM-OGlcNAc-Site (layer {l})': (f'score_layer{l}', scores[f'call_layer{l}'].to_numpy() == 1)
          for l in (33, 36)}
MODELS['Our model'] = ('probability', scores['probability'].to_numpy() >= OUR_CUTOFF)

rows = []
for subset, keep in SUBSETS.items():
    y_sub = scores.loc[keep, 'label'].to_numpy()
    for name, (col, called_all) in MODELS.items():
        score, called = scores.loc[keep, col].to_numpy(), called_all[keep]
        tp = int((called & (y_sub == 1)).sum())
        rows.append({'subset': subset, 'proteins': scores.loc[keep, 'protein_id'].nunique(), 'sites': int(keep.sum()),
                     'positive': int(y_sub.sum()), 'model': name,
                     'AUROC': roc_auc_score(y_sub, score), 'AUPRC': average_precision_score(y_sub, score),
                     'MCC': matthews_corrcoef(y_sub, called), 'precision': tp / max(called.sum(), 1),
                     'recall': tp / y_sub.sum()})
filtered = pd.DataFrame(rows)
filtered.to_csv(f'{OUT_DIR}/test_metrics_filtered_snapshot.csv', index=False)

atlas4 = pd.read_csv('data_processed/other_tools/Atlas4.0/test_protein_overlap.csv')   # cell 7, for comparison
print(f'\nTest proteins: {len(overlap)}   same protein as a snapshot protein: {overlap["same_protein"].sum()} '
      f'(Atlas 4.0: {atlas4["same_protein"].sum()})   homolog >= 30%: {overlap["homolog_30"].sum()} '
      f'(Atlas 4.0: {atlas4["homolog_30"].sum()})')
print('Highest identity to snapshot proteins (test proteins), quantiles 0/25/50/75/100%:',
      np.round(overlap['max_identity'].quantile([0, 0.25, 0.5, 0.75, 1]).to_numpy(), 2))
print()
print(filtered.to_string(index=False, float_format='{:.4f}'.format))

# %%
import glob

# ---------------------------------------------------------------------------
# External validation: LM-OGlcNAc-Site on the lab GPU
#   Scoring every S/T of the external proteins (NB01) takes one forward pass per S/T for proteins
#   longer than 1,021 residues (~333,000 windows; estimated 70-90 h on one A100), so it runs as a
#   script (scripts/lm_oglcnac_site_external.py, written by the next cell) on the lab server.
#   This cell writes the script's inputs to EXT_DIR/input:
#     external_proteins.parquet: seq_id and sequence of every external protein (NB01);
#     check_sites.parquet: the S/T of 16 test-set windows (8 of proteins <= 1,021 residues, 8 of
#       1,021 residues from longer proteins) with the probabilities computed in this notebook
#       (test_scores.parquet), which the script must reproduce before scoring.
# ---------------------------------------------------------------------------
EXT_DIR = f'{OUT_DIR}/external_run'
os.makedirs(f'{EXT_DIR}/input', exist_ok=True)
os.makedirs('scripts', exist_ok=True)

found = glob.glob('data_processed/external_proteins.parquet') + glob.glob('data_processed/*/external_proteins.parquet')
assert len(found) == 1, f'external_proteins.parquet: {found}'
external = pd.read_parquet(found[0], columns=['seq_id', 'sequence'])
assert external['seq_id'].is_unique
external.to_parquet(f'{EXT_DIR}/input/external_proteins.parquet', index=False)
print(f'{found[0]}: {len(external):,} proteins, '
      f'{sum(s.count("S") + s.count("T") for s in external["sequence"]):,} S/T')

test_sites = pd.read_parquet(f'{OUT_DIR}/test_sites.parquet')
test_probs = pd.read_parquet(f'{OUT_DIR}/test_scores.parquet')
PROB_COLS = ['ankh_prob', 'prott5_prob', 'esm2_layer33_prob', 'esm2_layer36_prob']
check = test_sites.merge(test_probs[['protein_id', 'position'] + PROB_COLS], on=['protein_id', 'position'],
                         how='left', validate='one_to_one')
assert check[PROB_COLS].notna().all().all()
windows = check.drop_duplicates(['protein_id', 'window_start'])
chosen = pd.concat([windows[windows['window_length'] < MAX_LEN].sort_values('protein_id').head(8),
                    windows[windows['window_length'] == MAX_LEN].sample(8, random_state=0)])
check = check.merge(chosen[['protein_id', 'window_start']], on=['protein_id', 'window_start'])
seq_of = dict(zip(proteins['protein_id'], proteins['sequence']))
check['window'] = [seq_of[p][s:s + n] for p, s, n in zip(check['protein_id'], check['window_start'], check['window_length'])]
assert all(w[i] in 'ST' for w, i in zip(check['window'], check['site_in_window']))
check[['protein_id', 'position', 'window', 'site_in_window'] + PROB_COLS].to_parquet(f'{EXT_DIR}/input/check_sites.parquet', index=False)
print(f'check_sites.parquet: {check["window"].nunique()} windows, {len(check):,} S/T')

# %%
# The notebook wrote the script scripts/lm_oglcnac_site_external.py from this cell; the script is in
# this repository (scripts/) and is run on a GPU server, as described in its docstring.
