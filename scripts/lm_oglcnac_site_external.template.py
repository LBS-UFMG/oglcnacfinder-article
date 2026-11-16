"""LM-OGlcNAc-Site on the external validation proteins (every S/T), with the authors' trained models.

LM-OGlcNAc-Site (Pokharel et al. 2023, Int J Mol Sci 24:16000; github.com/KCLabMTU/LM-OGlcNAc-Site,
commit ff032df) scores each S/T with three networks applied to ProtT5-XL, Ankh-large and ESM-2 3B
embeddings. This script runs it as in NB06 (benchmark on our test set):
  - proteins longer than 1,021 residues: one window per S/T from their get_1021_string(), called
    with the 0-based index (predict.py passes the whole protein and fails on long proteins);
  - one forward pass per window and pLM, no batching (as in predict.py); float32, TF32 off;
  - ESM-2 layers 33 and 36 from one forward pass (layer 33: the version of their paper and example
    output; layer 36: the released predict.py);
#   - the three networks applied to the S/T embeddings of each chunk in batches (row-independent;
#     NB06 checked batched against per-site predict()).
Their lines are kept as written; every change is tagged [modified] / [added] / [removed].

Before scoring, two checks: (1) their example input against their example output; (2) windows of
our test set against the probabilities computed in NB06 (Colab G4 GPU). Windows are processed in
chunks; each chunk is saved when complete and skipped on a restart, so the run can be resumed.

Libraries: PyTorch, transformers, fair-esm and ankh are imported before TensorFlow, and transformers
is told not to load TensorFlow (USE_TF=0): loading triton (PyTorch) after TensorFlow crashes Python.
TensorFlow runs on the CPU only.

Usage:
    python lm_oglcnac_site_external.py --repo LM-OGlcNAc-Site --input-dir external_run/input \
        --out-dir external_run/output [--device cuda:0] [--max-gpu-memory GB] [--check-only]
"""
import os
os.environ['USE_TF'] = '0'                 # transformers: do not import TensorFlow
os.environ['TF_USE_LEGACY_KERAS'] = '1'    # tf.keras = Keras 2 (tf-keras): their .h5 networks
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import argparse, ast, gc, hashlib, json, re, subprocess, sys, time
from importlib.metadata import version

#@ LM-OGlcNAc-Site predict.py 9-10 shift=0
import torch
import torch._dynamo   # loads triton before TensorFlow
import transformers, esm, ankh
from transformers import T5EncoderModel, T5Tokenizer
from tqdm.auto import tqdm
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')   # TensorFlow on the CPU; the GPU is left to PyTorch
from tensorflow.keras.models import load_model

REPO_COMMIT = 'ff032df049c75812a9ad070580c60ea19cb8d941'   # 2024-06-28, "Update predict.py"
MODEL_FILES = {'ankh': 'models/ankh_model_ann.h5', 'esm2': 'models/esm3B_model_ann.h5',
               'prott5': 'models/protT5_model_ann.h5'}
MODEL_SHA256 = {'ankh': '0526d7f22986734aa2e8ddbb14c628bdec2c187eb03de09b665a41e986a8988b',
                'esm2': '26808225d0df32b70bfbc02e2a27d0b79edea2abff44cf7f0ecbd1a56b48bff7',
                'prott5': 'b723ee0bd2f500f96b437e5c57159b49afcba34ffcae6095a89498fdc6f1c9f7'}
MAX_LEN = 1021        # longest sequence predict.py passes to the pLMs
ESM_LAYERS = (33, 36)
CHECK_TOL = 1e-4      # largest accepted |difference| in the checks (other GPU and library versions)

parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
parser.add_argument('--repo', required=True, help='clone of github.com/KCLabMTU/LM-OGlcNAc-Site')
parser.add_argument('--input-dir', required=True, help='external_proteins.parquet and check_sites.parquet')
parser.add_argument('--out-dir', required=True)
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--chunk', type=int, default=500, help='windows per saved chunk')
parser.add_argument('--check-only', action='store_true', help='run the two checks and stop')
parser.add_argument('--max-gpu-memory', type=float, default=None,
                    help='GB of GPU memory this process may use (leaves the rest to other users)')
#@ LM-OGlcNAc-Site predict.py 43-43 shift=-4
REPO = args.repo
os.makedirs(f'{args.out_dir}/chunks', exist_ok=True)

head = subprocess.run(['git', '-C', REPO, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
assert head == REPO_COMMIT, f'repository at {head}, expected {REPO_COMMIT}'
for name, path in MODEL_FILES.items():
    with open(f'{REPO}/{path}', 'rb') as f:
        assert hashlib.sha256(f.read()).hexdigest() == MODEL_SHA256[name], f'{path} differs'

# Full float32 precision (their predict.py runs on the CPU)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
tf.config.experimental.enable_tensor_float_32_execution(False)

# get_1021_string, taken from predict.py and executed unchanged (predict.py cannot be imported: it
# loads all models and predicts at import)
source = open(f'{REPO}/predict.py').read()
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'get_1021_string')
exec(ast.get_source_segment(source, node))

# ---------------------------------------------------------------------------
# predict.py (repository commit ff032df), model loading and embedding functions, as in NB06
#   Changes: the pLMs run on the GPU (their script hard-codes the CPU); the ESM-2 output is moved
#   back to the CPU as a NumPy array; ESM-2 returns layers 33 and 36 from one forward pass; ESM-2
#   runs without contact prediction (return_contacts=False): the contacts are not used, and keeping
#   the attention maps for them raised the peak GPU memory to 57 GB; ESM-2 uses rotary embeddings,
#   so its attention is computed the same way either way (checked below, check 3); model paths are
#   prefixed with the repository folder.
# ---------------------------------------------------------------------------
prot_t5_model_path = f'{REPO}/models/protT5_model_ann.h5'  # [modified] original: {{LM-OGlcNAc-Site predict.py 61}}
esm2_model_path = f'{REPO}/models/esm3B_model_ann.h5'  # [modified] original: {{LM-OGlcNAc-Site predict.py 62}}
ankh_model_path = f'{REPO}/models/ankh_model_ann.h5'  # [modified] original: {{LM-OGlcNAc-Site predict.py 63}}
#@ LM-OGlcNAc-Site predict.py 64-67 shift=0
#@ LM-OGlcNAc-Site predict.py 79-87 shift=0
device = torch.device(args.device)  # [modified] original: {{LM-OGlcNAc-Site predict.py 88}}
if args.max_gpu_memory:  # [added] memory cap on a shared GPU
    torch.cuda.set_per_process_memory_fraction(args.max_gpu_memory * 1e9 / torch.cuda.get_device_properties(device).total_memory, device)
#@ LM-OGlcNAc-Site predict.py 89-95 shift=0
ankh_pretrained_model = ankh_pretrained_model.to(device)  # [added] Ankh on the GPU (their input ids are moved to `device`)
#@ LM-OGlcNAc-Site predict.py 96-101 shift=0
pretrained_model_esm = pretrained_model_esm.to(device)  # [added] ESM-2 on the GPU

#@ LM-OGlcNAc-Site predict.py 131-199 shift=0
def get_esm2_3B_features_layers(sequence, layers=ESM_LAYERS, return_contacts=False):  # [modified] original: {{LM-OGlcNAc-Site predict.py 200}}
#@ LM-OGlcNAc-Site predict.py 208-212 shift=0
    batch_tokens = batch_tokens.to(device)  # [added] input on the GPU

#@ LM-OGlcNAc-Site predict.py 215-215 shift=0
        results = pretrained_model_esm(batch_tokens, repr_layers=list(layers), return_contacts=return_contacts)  # [modified] original: {{LM-OGlcNAc-Site predict.py 216}}

    # return only residue level embeddings
    return {layer: results["representations"][layer][:,1:-1,:][0].cpu().numpy() for layer in layers}  # [modified] original: token_representations = results["representations"][36]; return token_representations[:,1:-1,:][0]


#@ LM-OGlcNAc-Site predict.py 225-229 shift=0

def st_embeddings(window, idx):  # [added] embeddings of the S/T rows `idx` (0-based) of one window
    pt5_all = get_protT5_features(window)
    ankh_all = get_ankh_features(window)
    esm_all = get_esm2_3B_features_layers(window)
    assert pt5_all.shape == (len(window), 1024) and ankh_all.shape == (len(window), 1536)
    assert all(esm_all[l].shape == (len(window), 2560) for l in ESM_LAYERS)
    assert all(window[i] in 'ST' for i in idx)
    return {'ankh': ankh_all[idx], 'prott5': pt5_all[idx], **{f'esm2_layer{l}': esm_all[l][idx] for l in ESM_LAYERS}}


def network_probabilities(emb):  # [added] the three networks on stacked S/T embeddings, in batches
    out = {'ankh_prob': ankh_model_ann.predict(emb['ankh'], batch_size=1024, verbose=0)[:, 0],
           'prott5_prob': prot_t5_model_ann.predict(emb['prott5'], batch_size=1024, verbose=0)[:, 0]}
    for l in ESM_LAYERS:
        out[f'esm2_layer{l}_prob'] = esm2_model_ann.predict(emb[f'esm2_layer{l}'], batch_size=1024, verbose=0)[:, 0]
    return out


def stack(embeddings):  # [added] concatenate the per-window embedding rows
    return {k: np.concatenate([e[k] for e in embeddings]) for k in embeddings[0]}

PROB_COLS = ['ankh_prob', 'prott5_prob'] + [f'esm2_layer{l}_prob' for l in ESM_LAYERS]
provenance = {'repository': 'https://github.com/KCLabMTU/LM-OGlcNAc-Site', 'commit': REPO_COMMIT,
              'model_sha256': MODEL_SHA256, 'python': sys.version.split()[0], 'torch': torch.__version__,
              'tensorflow': tf.__version__, 'transformers': transformers.__version__, 'ankh': version('ankh'),
              'fair-esm': version('fair-esm'), 'gpu': torch.cuda.get_device_name(device)}
print('Versions: ' + '   '.join(f'{k} {v}' for k, v in provenance.items() if k not in ('repository', 'model_sha256')),
      flush=True)

# ---------------------------------------------------------------------------
# Check 1: their example input (2 proteins, 59 S/T) against their example output (layer 33;
# written at commit e0fa511 on the CPU, printed at float32 precision)
# ---------------------------------------------------------------------------
expected = pd.read_csv(f'{REPO}/output/results.csv')
example = {}
for record in open(f'{REPO}/input/sequence.fasta').read().split('>')[1:]:
    header, *lines = record.strip().split('\n')
    example[header.split()[0]] = ''.join(lines)
rows = []
for pid, seq in example.items():
    assert len(seq) <= MAX_LEN
    idx = np.array([i for i, aa in enumerate(seq) if aa in 'ST'])
    probs = network_probabilities(st_embeddings(seq, idx))
    rows.append(pd.DataFrame({'prot_desc': pid, 'position': idx + 1, **probs}))
ours = pd.concat(rows, ignore_index=True)
assert (ours['prot_desc'].values == expected['prot_desc'].values).all()
assert (ours['position'].values == expected['position'].values).all()
check1 = {name: float(np.abs(ours[name].values - expected[col].values).max()) for name, col in
          [('ankh_prob', 'ankh_prob(Th = 0.496)'), ('prott5_prob', 'prot_t5_prob(Th = 0.669)'),
           ('esm2_layer33_prob', 'esm2_prob(Th = 0.797)')]}
print('Check 1, their example (max |diff|): ' + '   '.join(f'{k} {v:.2e}' for k, v in check1.items()), flush=True)
assert max(check1.values()) < CHECK_TOL

# ---------------------------------------------------------------------------
# Check 2: windows of our test set against the probabilities computed in NB06
# ---------------------------------------------------------------------------
check = pd.read_parquet(f'{args.input_dir}/check_sites.parquet')
groups = list(check.groupby('window', sort=False))
probs = network_probabilities(stack([st_embeddings(w, sub['site_in_window'].to_numpy()) for w, sub in groups]))
expected_check = pd.concat([sub for _, sub in groups])
diffs = {c: float(np.abs(probs[c] - expected_check[c].to_numpy()).max()) for c in PROB_COLS}
print(f'Check 2, {check["window"].nunique()} test-set windows, {len(check)} S/T (max |diff| vs NB06): '
      + '   '.join(f'{k} {v:.2e}' for k, v in diffs.items()), flush=True)
assert max(diffs.values()) < CHECK_TOL
gpu_peak = {'allocated GB': torch.cuda.max_memory_allocated(device) / 1e9, 'reserved GB': torch.cuda.max_memory_reserved(device) / 1e9}
provenance['gpu memory, checks 1-2'] = gpu_peak
print('GPU memory, peak in checks 1-2 (as in the run): ' + '   '.join(f'{k} {v:.1f}' for k, v in gpu_peak.items()), flush=True)
# ---------------------------------------------------------------------------
# Check 3 (with --check-only): ESM-2 embeddings without contact prediction (used here) against
# with contact prediction (their predict.py), on the 1,021-residue windows of check 2
# ---------------------------------------------------------------------------
check3 = None
if args.check_only:
    check3 = {l: 0.0 for l in ESM_LAYERS}
    for w in check.loc[check['window'].str.len() == MAX_LEN, 'window'].unique():
        off, on = get_esm2_3B_features_layers(w), get_esm2_3B_features_layers(w, return_contacts=True)
        for l in ESM_LAYERS:
            check3[l] = max(check3[l], float(np.abs(off[l] - on[l]).max()))
    print(f'Check 3, ESM-2 embeddings without vs with contact prediction (max |diff|): '
          + '   '.join(f'layer {l} {v:.2e}' for l, v in check3.items()), flush=True)
    assert max(check3.values()) < CHECK_TOL
provenance['checks'] = {'their example': check1, 'NB06 test windows': diffs,
                        'ESM-2 contacts off vs on': check3}
with open(f'{args.out_dir}/provenance.json', 'w') as f:
    json.dump(provenance, f, indent=1)
if args.check_only:
    sys.exit(0)

# ---------------------------------------------------------------------------
# Sites and windows: every S/T of every external protein
# ---------------------------------------------------------------------------
proteins = pd.read_parquet(f'{args.input_dir}/external_proteins.parquet', columns=['seq_id', 'sequence'])
proteins = proteins.sort_values('seq_id').reset_index(drop=True)
rows = []
for sid, seq in zip(proteins['seq_id'], proteins['sequence']):
    for i, aa in enumerate(seq):
        if aa not in 'ST':
            continue
        if len(seq) > MAX_LEN:
            new_site, window = get_1021_string(seq, i)
        else:
            new_site, window = i, seq
        start = i - new_site
        assert len(window) <= MAX_LEN and window == seq[start:start + len(window)] and window[new_site] == aa
        rows.append((sid, i + 1, aa, start, len(window), new_site))
sites = pd.DataFrame(rows, columns=['seq_id', 'position', 'residue', 'window_start', 'window_length', 'site_in_window'])
windows = (sites.drop_duplicates(['seq_id', 'window_start'])[['seq_id', 'window_start', 'window_length']]
           .reset_index(drop=True))
rows_of = sites.groupby(['seq_id', 'window_start'], sort=False).indices
seq_of = dict(zip(proteins['seq_id'], proteins['sequence']))
n_chunks = -(-len(windows) // args.chunk)
plan = hashlib.sha256(pd.util.hash_pandas_object(windows, index=False).values.tobytes()).hexdigest()
print(f'External proteins: {len(proteins):,}   S/T: {len(sites):,}   windows (forward passes per pLM): '
      f'{len(windows):,} in {n_chunks} chunks of {args.chunk}', flush=True)

# The window plan must be identical when a run is resumed
plan_file = f'{args.out_dir}/window_plan.json'
if os.path.exists(plan_file):
    previous = json.load(open(plan_file))
    assert previous['sha256'] == plan, 'window plan differs from the interrupted run'
    assert previous['chunk'] == args.chunk, f'resume with --chunk {previous["chunk"]}, as the interrupted run'
else:
    json.dump({'sha256': plan, 'windows': len(windows), 'sites': len(sites), 'chunk': args.chunk}, open(plan_file, 'w'))

chunk_path = lambda c: f'{args.out_dir}/chunks/chunk_{c:04d}.parquet'
done = [c for c in range(n_chunks) if os.path.exists(chunk_path(c))]
bar = tqdm(total=len(windows), initial=sum(len(windows.iloc[c * args.chunk:(c + 1) * args.chunk]) for c in done),
           unit='window', mininterval=60, file=sys.stdout)
site_in_window = sites['site_in_window'].to_numpy()
for c in range(n_chunks):
    if os.path.exists(chunk_path(c)):
        continue
    site_rows, embeddings = [], []
    for w in windows.iloc[c * args.chunk:(c + 1) * args.chunk].itertuples():
        window = seq_of[w.seq_id][w.window_start:w.window_start + w.window_length]
        rows = rows_of[(w.seq_id, w.window_start)]
        site_rows.append(rows)
        embeddings.append(st_embeddings(window, site_in_window[rows]))
        bar.update(1)
    chunk = pd.DataFrame({'site_row': np.concatenate(site_rows), **network_probabilities(stack(embeddings))})
    chunk.to_parquet(chunk_path(c) + '.part', index=False)
    os.replace(chunk_path(c) + '.part', chunk_path(c))   # complete chunks only
bar.close()

# ---------------------------------------------------------------------------
# All sites: probabilities, LM-OGlcNAc-Site call (>= 2 of 3 networks above their thresholds, as in
# predict.py) and continuous score (mean of the three probabilities), for each ESM-2 layer (as NB06)
# ---------------------------------------------------------------------------
probs = pd.concat([pd.read_parquet(chunk_path(c)) for c in range(n_chunks)]).set_index('site_row').sort_index()
assert np.array_equal(probs.index.to_numpy(), np.arange(len(sites))), 'every site exactly once'
scores = pd.concat([sites[['seq_id', 'position', 'residue']], probs.reset_index(drop=True)], axis=1)
for l in ESM_LAYERS:
    votes = ((scores['ankh_prob'] > cutoff_threshold_ankh).astype(int)
             + (scores[f'esm2_layer{l}_prob'] > cutoff_threshold_esm).astype(int)
             + (scores['prott5_prob'] > cutoff_threshold_prott5).astype(int))
    scores[f'call_layer{l}'] = (votes > 1).astype(int)
    scores[f'score_layer{l}'] = scores[['ankh_prob', 'prott5_prob', f'esm2_layer{l}_prob']].mean(axis=1)
scores.to_parquet(f'{args.out_dir}/external_scores.parquet', index=False)
provenance['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
with open(f'{args.out_dir}/provenance.json', 'w') as f:
    json.dump(provenance, f, indent=1)
print(f'Saved {len(scores):,} S/T scores: {args.out_dir}/external_scores.parquet', flush=True)
