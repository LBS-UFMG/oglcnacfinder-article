# %%
import os
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# ESM2 windows and the site table (shared by every pLM)
#   Proteins of up to 1,022 residues (1,024 tokens with the start and end tokens) are
#   embedded whole. Longer proteins are cut into the fewest windows of 1,022 residues that
#   overlap by at least 256, spaced evenly from the protein's first to its last residue.
#   Every window is stored in full, so residues in an overlap are stored once per window.
#   Embedding-matrix rows: windows in protein order, residues in sequence order.
#   Each S/T is assigned the window where it is most central (largest distance to the
#   nearer window edge); the neighbours used for its window mean are read from that window.
# ---------------------------------------------------------------------------
MAX_RES, OVERLAP = 1022, 256
EMB_DIR = 'data_processed/embeddings'
os.makedirs(EMB_DIR, exist_ok=True)

def window_starts(length):
    """0-based start of each window: the fewest windows of MAX_RES residues overlapping by
    at least OVERLAP, spaced evenly from the protein's first to its last residue."""
    if length <= MAX_RES:
        return [0]
    n = int(np.ceil((length - OVERLAP) / (MAX_RES - OVERLAP)))
    return [round(i * (length - MAX_RES) / (n - 1)) for i in range(n)]

proteins = pd.read_parquet('data_processed/atlas5_proteins.parquet')
assert proteins['split'].notna().all() and not proteins['protein_id'].duplicated().any()

windows = pd.DataFrame([{'protein_id': pid, 'window': w, 'first_position': s + 1, 'length': min(L, MAX_RES)}
                        for pid, L in zip(proteins['protein_id'], proteins['length'])
                        for w, s in enumerate(window_starts(L))])
windows['row'] = windows['length'].cumsum() - windows['length']
windows_of = {pid: g.to_dict('records') for pid, g in windows.groupby('protein_id', sort=False)}

sites, min_room = [], np.inf
for pid, seq, split, pos, msk in zip(proteins['protein_id'], proteins['sequence'], proteins['split'],
                                     proteins['positive'], proteins['masked']):
    pos, msk, ws, L = set(pos), set(msk), windows_of[pid], len(seq)
    for p, aa in enumerate(seq, start=1):
        if aa not in 'ST':
            continue
        inside = [w for w in ws if w['first_position'] <= p < w['first_position'] + w['length']]
        w = max(inside, key=lambda w: min(p - w['first_position'], w['first_position'] + w['length'] - 1 - p))
        if len(ws) > 1:
            # Context available on each side within the chosen window (protein ends excluded)
            last = w['first_position'] + w['length'] - 1
            left = p - w['first_position'] if w['first_position'] > 1 else np.inf
            right = last - p if last < L else np.inf
            min_room = min(min_room, left, right)
        sites.append({'protein_id': pid, 'position': p, 'residue': aa, 'split': split,
                      'label': 'positive' if p in pos else 'masked' if p in msk else 'negative',
                      'window': w['window'], 'row': w['row'] + p - w['first_position']})
sites = pd.DataFrame(sites)

# The site table must reproduce the label counts of the dataset
counts = sites['label'].value_counts()
assert counts['positive'] == proteins['positive'].str.len().sum()
assert counts['masked'] == proteins['masked'].str.len().sum()
assert counts['negative'] == proteins['n_negative'].sum()

n_long = (proteins['length'] > MAX_RES).sum()
print(f'Proteins: {len(proteins):,}   longer than {MAX_RES}: {n_long:,}   windows: {len(windows):,}')
print(f'Embedding rows: {windows["length"].sum():,}   residues: {proteins["length"].sum():,}   '
      f'extra rows from overlaps: {windows["length"].sum() - proteins["length"].sum():,}')
print(f'Smallest context on either side of a target S/T in a multi-window protein: {min_room:.0f} residues')
print('\n', sites.groupby(['split', 'label']).size().unstack().loc[['train', 'val', 'test']].to_string())

windows.to_parquet(f'{EMB_DIR}/windows.parquet', index=False)
sites.to_parquet(f'{EMB_DIR}/sites.parquet', index=False)

# %%
from pathlib import Path

import torch
import transformers
from huggingface_hub import try_to_load_from_cache
from tqdm.auto import tqdm
from transformers import AutoTokenizer, EsmModel, EsmForMaskedLM

# ---------------------------------------------------------------------------
# ESM2 per-residue embeddings: last layer, float32
#   Each window is embedded as its own sequence (start token + residues + end token); the
#   last-layer hidden state of token i (1-based) is the embedding of the window's i-th residue.
#   Windows are batched by length (longest first) under a token budget; padding is excluded
#   through the attention mask. TF32 matrix multiplication is disabled to keep full float32.
# ---------------------------------------------------------------------------
PLMS = {'ESM2-8M':   'facebook/esm2_t6_8M_UR50D',
        'ESM2-35M':  'facebook/esm2_t12_35M_UR50D',
        'ESM2-150M': 'facebook/esm2_t30_150M_UR50D',
        'ESM2-650M': 'facebook/esm2_t33_650M_UR50D'}
DEVICE = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

seq_of = dict(zip(proteins['protein_id'], proteins['sequence']))
window_seqs = [seq_of[p][f - 1:f - 1 + n]
               for p, f, n in zip(windows['protein_id'], windows['first_position'], windows['length'])]
assert [len(s) for s in window_seqs] == windows['length'].tolist()

def model_revision(path):
    """Commit hash of the cached model files (Hugging Face snapshot directory name)."""
    return Path(try_to_load_from_cache(path, 'config.json')).parent.name

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

# Check 1: one token per residue, framed by the start and end tokens, no unknown tokens
tokenizers = {name: AutoTokenizer.from_pretrained(path) for name, path in PLMS.items()}
assert all(t.get_vocab() == tokenizers['ESM2-8M'].get_vocab() for t in tokenizers.values())
tok = tokenizers['ESM2-8M']
for s in window_seqs:
    ids = tok(s)['input_ids']
    assert len(ids) == len(s) + 2 and ids[0] == tok.cls_token_id and ids[-1] == tok.eos_token_id
    assert tok.unk_token_id not in ids
letters = pd.Series(list(''.join(proteins['sequence']))).value_counts()
print('Check 1 passed. Residue letters outside the 20 standard:',
      letters.drop(list('ACDEFGHIKLMNPQRSTVWY'), errors='ignore').to_dict())

# Checks 2 and 3 on three windows of different lengths, for every model
test_seqs = [window_seqs[k] for k in np.argsort(windows['length'].values)[[0, len(windows) // 2, -1]]]
for name, path in PLMS.items():
    model = EsmModel.from_pretrained(path, add_pooling_layer=False).to(DEVICE).eval()
    mlm = EsmForMaskedLM.from_pretrained(path).to(DEVICE).eval()
    with torch.no_grad():
        alone = [model(**tok(s, return_tensors='pt').to(DEVICE)).last_hidden_state[0, 1:len(s) + 1]
                 for s in test_seqs]
        bsb = [mlm(**tok(s, return_tensors='pt').to(DEVICE), output_hidden_states=True).hidden_states[-1][0, 1:len(s) + 1]
               for s in test_seqs]
    batched = np.zeros((sum(map(len, test_seqs)), model.config.hidden_size), dtype=np.float32)
    starts = np.cumsum([0] + [len(s) for s in test_seqs[:-1]])
    embed_windows(model, tok, test_seqs, batched, starts, max_tokens=10**6)
    diff_bsb = max((a - b).abs().max().item() for a, b in zip(alone, bsb))
    diff_batch = max(np.abs(a.cpu().numpy() - batched[r:r + len(s)]).max()
                     for a, r, s in zip(alone, starts, test_seqs))
    print(f'{name:10s} dim {model.config.hidden_size:5d}   revision {model_revision(path)[:10]}   '
          f'max |EsmModel - MaskedLM hidden_states[-1]|: {diff_bsb:.1e}   max |alone - batched|: {diff_batch:.1e}')
    del model, mlm
    torch.cuda.empty_cache()

print(f'\ntorch {torch.__version__}   transformers {transformers.__version__}   GPU {torch.cuda.get_device_name()}')

# %%
import hashlib, json, shutil, time
from datetime import date

# ---------------------------------------------------------------------------
# Extract and save the embeddings of every window, one matrix per model
#   The matrix is written on the Colab local disk, checked, then copied to Drive. A JSON
#   file written after the copy records the model revision, the matrix shape and a checksum
#   of the windows; a model whose JSON matches the current windows is skipped on later runs.
#   Checks: every row was written (no all-zero rows) and all values are finite; 20 random
#   windows re-embedded one at a time match the stored rows.
# ---------------------------------------------------------------------------
LOCAL_DIR = '/content/embeddings'
MAX_TOKENS = {'ESM2-8M': 65536, 'ESM2-35M': 65536, 'ESM2-150M': 65536, 'ESM2-650M': 32768}
CHUNK = 500_000   # rows per chunk when scanning a matrix

rows = windows['row'].to_numpy()
n_rows = int(windows['length'].sum())
windows_checksum = hashlib.sha256('\n'.join(f'{r}\t{s}' for r, s in zip(rows, window_seqs)).encode()).hexdigest()
spot = np.random.default_rng(0).choice(len(windows), 20, replace=False)

def copy_with_progress(src, dst, chunk=64 * 2**20):
    """Copy a file in chunks, showing progress."""
    with open(src, 'rb') as fi, open(dst, 'wb') as fo, \
         tqdm(total=os.path.getsize(src), unit='B', unit_scale=True, desc='copy to Drive') as bar:
        while block := fi.read(chunk):
            fo.write(block)
            bar.update(len(block))

for name, path in PLMS.items():
    drive_dir, local_dir = f'{EMB_DIR}/{name}', f'{LOCAL_DIR}/{name}'
    meta_file = f'{drive_dir}/residues.json'
    if os.path.exists(meta_file) and json.load(open(meta_file))['windows_checksum'] == windows_checksum:
        print(f'{name}: already extracted, skipped')
        continue

    model = EsmModel.from_pretrained(path, add_pooling_layer=False).to(DEVICE).eval()
    dim = model.config.hidden_size
    os.makedirs(local_dir, exist_ok=True)
    assert shutil.disk_usage('/content').free > 1.1 * n_rows * dim * 4, 'not enough local disk'
    local_file = f'{local_dir}/residues.npy'
    out = np.lib.format.open_memmap(local_file, mode='w+', dtype=np.float32, shape=(n_rows, dim))

    start = time.time()
    embed_windows(model, tok, window_seqs, out, rows, MAX_TOKENS[name])
    out.flush()
    minutes = (time.time() - start) / 60

    for i in tqdm(range(0, n_rows, CHUNK), desc='checking rows'):
        block = out[i:i + CHUNK]
        assert np.isfinite(block).all() and not (block == 0).all(axis=1).any()
    with torch.no_grad():
        spot_diff = max(np.abs(model(**tok(window_seqs[k], return_tensors='pt').to(DEVICE)).last_hidden_state[0, 1:-1].cpu().numpy()
                               - out[rows[k]:rows[k] + len(window_seqs[k])]).max() for k in spot)
    assert spot_diff < 1e-4, spot_diff
    del out, model
    torch.cuda.empty_cache()

    os.makedirs(drive_dir, exist_ok=True)
    copy_with_progress(local_file, f'{drive_dir}/residues.npy')
    assert os.path.getsize(f'{drive_dir}/residues.npy') == os.path.getsize(local_file)
    with open(meta_file, 'w') as f:
        json.dump({'model': path, 'revision': model_revision(path), 'layer': 'last', 'dtype': 'float32',
                   'shape': [n_rows, dim], 'windows_checksum': windows_checksum,
                   'max_tokens_per_batch': MAX_TOKENS[name], 'extraction_minutes': round(minutes, 1),
                   'spot_check_max_abs_diff': float(spot_diff), 'torch': torch.__version__,
                   'transformers': transformers.__version__, 'gpu': torch.cuda.get_device_name(),
                   'date': str(date.today())}, f, indent=1)
    os.remove(local_file)
    print(f'{name}: {n_rows:,} x {dim}   {minutes:.1f} min   spot check max |diff| {spot_diff:.1e}   saved\n')

# %%
from torch.utils.flop_counter import FlopCounterMode

# ---------------------------------------------------------------------------
# Compute used for the extraction: GPU, GPU time and floating-point operations
#   GPU time: wall-clock time of embed_windows per model (recorded in its JSON file).
#   FLOPs: counted with PyTorch's FlopCounterMode, which counts matrix multiplications
#   (2 FLOPs per multiply-add); element-wise operations (softmax, layer norm, activations)
#   are not counted. A forward pass of B sequences padded to T tokens costs B·(a·T + b·T²);
#   a and b are measured on two lengths, checked on a third, and summed over the batches
#   embed_windows formed (same batching rule, padding included).
# ---------------------------------------------------------------------------
def batch_shapes(lengths, max_tokens):
    """(sequences, padded tokens) of each batch, with the batching rule of embed_windows."""
    shapes, n, longest = [], 0, 0
    for L in sorted(lengths, reverse=True):
        if n and (n + 1) * (longest + 2) > max_tokens:
            shapes.append((n, longest + 2))
            n = 0
        if n == 0:
            longest = L
        n += 1
    shapes.append((n, longest + 2))
    return shapes

def forward_flops(model, n_tokens):
    """FLOPs of one forward pass of a single sequence of n_tokens tokens."""
    toks = tok('A' * (n_tokens - 2), return_tensors='pt').to(DEVICE)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(**toks)
    return counter.get_total_flops()

compute = []
for name, path in PLMS.items():
    meta = json.load(open(f'{EMB_DIR}/{name}/residues.json'))
    model = EsmModel.from_pretrained(path, add_pooling_layer=False).to(DEVICE).eval()
    f256, f1024, f600 = (forward_flops(model, T) for T in (256, 1024, 600))
    b = (f1024 / 1024 - f256 / 256) / (1024 - 256)
    a = f256 / 256 - b * 256
    assert abs(a * 600 + b * 600**2 - f600) / f600 < 1e-6, 'FLOPs are not quadratic in length'
    shapes = batch_shapes(windows['length'].tolist(), meta['max_tokens_per_batch'])
    assert sum(n for n, _ in shapes) == len(windows)
    compute.append({'model': name, 'parameters': sum(p.numel() for p in model.parameters()),
                    'GPU': meta['gpu'], 'batches': len(shapes), 'GPU minutes': meta['extraction_minutes'],
                    'FLOPs': sum(n * (a * T + b * T**2) for n, T in shapes)})
    del model
    torch.cuda.empty_cache()

compute = pd.DataFrame(compute)
compute.loc[len(compute)] = {'model': 'total', 'GPU': compute['GPU'].iloc[0], 'batches': compute['batches'].sum(),
                             'GPU minutes': compute['GPU minutes'].sum(), 'FLOPs': compute['FLOPs'].sum()}
compute['parameters'] = compute['parameters'].astype('Int64')   # integer column; the total is left empty
compute['batches'] = compute['batches'].astype(int)
print(compute.to_string(index=False, formatters={'FLOPs': '{:.2e}'.format}))
compute.to_csv(f'{EMB_DIR}/compute_extraction.csv', index=False)

# %%
import hashlib, json, os, shutil, time
from datetime import date

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Classifier input features, precomputed for every site (row i = row i of sites.parquet)
#   target.npy     embedding of the target S/T
#   mean_pmK.npy   mean embedding of the residues from K before to K after the target
#                  (target included), for K = 5, 10, 15, 20, 25, 35
#   Residues are read from the window assigned to the site. Near a protein end the window
#   ends there and the mean uses the residues that exist (no padding). Means are computed
#   from float64 cumulative sums per window and stored in float32.
#   The residue matrix is copied to the Colab local disk; the features are written there,
#   checked, and copied to Drive. A JSON file written last marks a model as done.
# ---------------------------------------------------------------------------
HALF_WIDTHS = [5, 10, 15, 20, 25, 35]
PLM_NAMES = ['ESM2-8M', 'ESM2-35M', 'ESM2-150M', 'ESM2-650M']
EMB_DIR, FEAT_DIR, LOCAL_DIR = 'data_processed/embeddings', 'data_processed/features', '/content/features'

windows = pd.read_parquet(f'{EMB_DIR}/windows.parquet')
sites = pd.read_parquet(f'{EMB_DIR}/sites.parquet')
sites_checksum = hashlib.sha256(sites[['protein_id', 'position', 'row']].to_csv(index=False).encode()).hexdigest()

# Window of each site: first row, length, and the site's offset inside it
w = windows.rename(columns={'row': 'window_row', 'length': 'window_length'})
site_win = sites.merge(w[['protein_id', 'window', 'window_row', 'window_length', 'first_position']],
                       on=['protein_id', 'window'], how='left', validate='many_to_one')
assert (site_win['row'].values == sites['row'].values).all()
offset = (site_win['row'] - site_win['window_row']).to_numpy()
window_length = site_win['window_length'].to_numpy()

# A mean may be cut short only at a protein end, never at an inner window edge
protein_length = (windows['first_position'] + windows['length'] - 1).groupby(windows['protein_id']).max()
at_start = (site_win['first_position'] == 1).to_numpy()
at_end = (site_win['first_position'] + site_win['window_length'] - 1
          == site_win['protein_id'].map(protein_length)).to_numpy()
K = max(HALF_WIDTHS)
assert ((offset - K >= 0) | at_start).all() and ((offset + K < window_length) | at_end).all()
print(f'Sites: {len(sites):,}   whose ±{K} window reaches a protein end: '
      f'{((offset - K < 0) | (offset + K >= window_length)).sum():,}')

sites_of_window = site_win.groupby('window_row').indices
length_of_window = dict(zip(windows['row'], windows['length']))
spot = np.random.default_rng(0).choice(len(sites), 2000, replace=False)

def copy_with_progress(src, dst, chunk=64 * 2**20):
    """Copy a file in chunks, showing progress."""
    with open(src, 'rb') as fi, open(dst, 'wb') as fo, \
         tqdm(total=os.path.getsize(src), unit='B', unit_scale=True, desc=f'copy {os.path.basename(src)}') as bar:
        while block := fi.read(chunk):
            fo.write(block)
            bar.update(len(block))

for name in PLM_NAMES:
    emb_meta = json.load(open(f'{EMB_DIR}/{name}/residues.json'))
    out_dir, local_dir = f'{FEAT_DIR}/{name}', f'{LOCAL_DIR}/{name}'
    meta_file = f'{out_dir}/features.json'
    if os.path.exists(meta_file):
        meta = json.load(open(meta_file))
        if meta['windows_checksum'] == emb_meta['windows_checksum'] and meta['sites_checksum'] == sites_checksum:
            print(f'{name}: features already computed, skipped')
            continue

    n_rows, dim = emb_meta['shape']
    out_bytes = len(sites) * dim * 4 * (1 + len(HALF_WIDTHS))
    os.makedirs(local_dir, exist_ok=True)
    assert shutil.disk_usage('/content').free > 1.1 * (n_rows * dim * 4 + out_bytes), 'not enough local disk'

    local_residues = f'{local_dir}/residues.npy'
    copy_with_progress(f'{EMB_DIR}/{name}/residues.npy', local_residues)
    M = np.load(local_residues, mmap_mode='r')
    assert M.shape == (n_rows, dim)

    files = ['target'] + [f'mean_pm{k:02d}' for k in HALF_WIDTHS]
    outs = {f: np.lib.format.open_memmap(f'{local_dir}/{f}.npy', mode='w+', dtype=np.float32,
                                         shape=(len(sites), dim)) for f in files}
    start = time.time()
    for row0 in tqdm(sorted(sites_of_window), desc=f'{name} features'):
        idx = sites_of_window[row0]
        block = np.asarray(M[row0:row0 + length_of_window[row0]])
        L = len(block)
        csum = np.zeros((L + 1, dim))
        np.cumsum(block, axis=0, dtype=np.float64, out=csum[1:])
        i = offset[idx]
        outs['target'][idx] = block[i]
        for k in HALF_WIDTHS:
            lo, hi = np.maximum(i - k, 0), np.minimum(i + k + 1, L)
            outs[f'mean_pm{k:02d}'][idx] = (csum[hi] - csum[lo]) / (hi - lo)[:, None]
    minutes = (time.time() - start) / 60

    # Checks: target = residue matrix row; means recomputed directly for 2,000 random sites
    rows = sites['row'].to_numpy()
    assert np.array_equal(outs['target'][spot], M[rows[spot]])
    max_diff = 0.0
    for s in spot:
        r0, L, i = site_win.at[s, 'window_row'], window_length[s], offset[s]
        for k in HALF_WIDTHS:
            direct = M[r0 + max(i - k, 0):r0 + min(i + k + 1, L)].astype(np.float64).mean(axis=0)
            max_diff = max(max_diff, np.abs(outs[f'mean_pm{k:02d}'][s] - direct).max())
    assert max_diff < 1e-5, max_diff
    for f in files:
        assert np.isfinite(outs[f]).all()
        outs[f].flush()
    del outs, M

    os.makedirs(out_dir, exist_ok=True)
    for f in files:
        copy_with_progress(f'{local_dir}/{f}.npy', f'{out_dir}/{f}.npy')
        assert os.path.getsize(f'{out_dir}/{f}.npy') == os.path.getsize(f'{local_dir}/{f}.npy')
    with open(meta_file, 'w') as fh:
        json.dump({'plm': name, 'files': files, 'half_widths': HALF_WIDTHS, 'shape': [len(sites), dim],
                   'dtype': 'float32', 'windows_checksum': emb_meta['windows_checksum'],
                   'sites_checksum': sites_checksum, 'spot_check_max_abs_diff': float(max_diff),
                   'minutes': round(minutes, 1), 'date': str(date.today())}, fh, indent=1)
    shutil.rmtree(local_dir)
    print(f'{name}: {len(sites):,} sites x {dim}   {minutes:.1f} min   spot check max |diff| {max_diff:.1e}   saved\n')

# %%
import hashlib, json, os, shutil, time
from datetime import date

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Classifier input features, precomputed for every site (row i = row i of sites.parquet)
#   target.npy     embedding of the target S/T
#   mean_pmK.npy   mean embedding of the residues from K before to K after the target
#                  (target included), for K = 5, 10, 15, 20, 25, 35
#   Residues are read from the window assigned to the site. Near a protein end the window
#   ends there and the mean uses the residues that exist (no padding). Means are computed
#   from float64 cumulative sums per window and stored in float32.
#   The residue matrix is copied to the Colab local disk; the features are written there,
#   checked, and copied to Drive. A JSON file written last marks a model as done.
# ---------------------------------------------------------------------------
HALF_WIDTHS = [5, 10, 15, 20, 25, 35]
PLM_NAMES = ['ESM2-8M', 'ESM2-35M', 'ESM2-150M', 'ESM2-650M']
EMB_DIR, FEAT_DIR, LOCAL_DIR = 'data_processed/embeddings', 'data_processed/features', '/content/features'

windows = pd.read_parquet(f'{EMB_DIR}/windows.parquet')
sites = pd.read_parquet(f'{EMB_DIR}/sites.parquet')
sites_checksum = hashlib.sha256(sites[['protein_id', 'position', 'row']].to_csv(index=False).encode()).hexdigest()

# Window of each site: first row, length, and the site's offset inside it
w = windows.rename(columns={'row': 'window_row', 'length': 'window_length'})
site_win = sites.merge(w[['protein_id', 'window', 'window_row', 'window_length', 'first_position']],
                       on=['protein_id', 'window'], how='left', validate='many_to_one')
assert (site_win['row'].values == sites['row'].values).all()
offset = (site_win['row'] - site_win['window_row']).to_numpy()
window_length = site_win['window_length'].to_numpy()

# A mean may be cut short only at a protein end, never at an inner window edge
protein_length = (windows['first_position'] + windows['length'] - 1).groupby(windows['protein_id']).max()
at_start = (site_win['first_position'] == 1).to_numpy()
at_end = (site_win['first_position'] + site_win['window_length'] - 1
          == site_win['protein_id'].map(protein_length)).to_numpy()
K = max(HALF_WIDTHS)
assert ((offset - K >= 0) | at_start).all() and ((offset + K < window_length) | at_end).all()
print(f'Sites: {len(sites):,}   whose ±{K} window reaches a protein end: '
      f'{((offset - K < 0) | (offset + K >= window_length)).sum():,}')

sites_of_window = site_win.groupby('window_row').indices
length_of_window = dict(zip(windows['row'], windows['length']))
spot = np.random.default_rng(0).choice(len(sites), 2000, replace=False)

def copy_with_progress(src, dst, chunk=64 * 2**20):
    """Copy a file in chunks, showing progress."""
    with open(src, 'rb') as fi, open(dst, 'wb') as fo, \
         tqdm(total=os.path.getsize(src), unit='B', unit_scale=True, desc=f'copy {os.path.basename(src)}') as bar:
        while block := fi.read(chunk):
            fo.write(block)
            bar.update(len(block))

for name in PLM_NAMES:
    emb_meta = json.load(open(f'{EMB_DIR}/{name}/residues.json'))
    out_dir, local_dir = f'{FEAT_DIR}/{name}', f'{LOCAL_DIR}/{name}'
    meta_file = f'{out_dir}/features.json'
    if os.path.exists(meta_file):
        meta = json.load(open(meta_file))
        if meta['windows_checksum'] == emb_meta['windows_checksum'] and meta['sites_checksum'] == sites_checksum:
            print(f'{name}: features already computed, skipped')
            continue

    n_rows, dim = emb_meta['shape']
    out_bytes = len(sites) * dim * 4 * (1 + len(HALF_WIDTHS))
    os.makedirs(local_dir, exist_ok=True)
    assert shutil.disk_usage('/content').free > 1.1 * (n_rows * dim * 4 + out_bytes), 'not enough local disk'

    local_residues = f'{local_dir}/residues.npy'
    copy_with_progress(f'{EMB_DIR}/{name}/residues.npy', local_residues)
    M = np.load(local_residues, mmap_mode='r')
    assert M.shape == (n_rows, dim)

    files = ['target'] + [f'mean_pm{k:02d}' for k in HALF_WIDTHS]
    outs = {f: np.lib.format.open_memmap(f'{local_dir}/{f}.npy', mode='w+', dtype=np.float32,
                                         shape=(len(sites), dim)) for f in files}
    start = time.time()
    for row0 in tqdm(sorted(sites_of_window), desc=f'{name} features'):
        idx = sites_of_window[row0]
        block = np.asarray(M[row0:row0 + length_of_window[row0]])
        L = len(block)
        csum = np.zeros((L + 1, dim))
        np.cumsum(block, axis=0, dtype=np.float64, out=csum[1:])
        i = offset[idx]
        outs['target'][idx] = block[i]
        for k in HALF_WIDTHS:
            lo, hi = np.maximum(i - k, 0), np.minimum(i + k + 1, L)
            outs[f'mean_pm{k:02d}'][idx] = (csum[hi] - csum[lo]) / (hi - lo)[:, None]
    minutes = (time.time() - start) / 60

    # Checks: target = residue matrix row; means recomputed directly for 2,000 random sites
    rows = sites['row'].to_numpy()
    assert np.array_equal(outs['target'][spot], M[rows[spot]])
    max_diff = 0.0
    for s in spot:
        r0, L, i = site_win.at[s, 'window_row'], window_length[s], offset[s]
        for k in HALF_WIDTHS:
            direct = M[r0 + max(i - k, 0):r0 + min(i + k + 1, L)].astype(np.float64).mean(axis=0)
            max_diff = max(max_diff, np.abs(outs[f'mean_pm{k:02d}'][s] - direct).max())
    assert max_diff < 1e-5, max_diff
    for f in files:
        assert np.isfinite(outs[f]).all()
        outs[f].flush()
    del outs, M

    os.makedirs(out_dir, exist_ok=True)
    for f in files:
        copy_with_progress(f'{local_dir}/{f}.npy', f'{out_dir}/{f}.npy')
        assert os.path.getsize(f'{out_dir}/{f}.npy') == os.path.getsize(f'{local_dir}/{f}.npy')
    with open(meta_file, 'w') as fh:
        json.dump({'plm': name, 'files': files, 'half_widths': HALF_WIDTHS, 'shape': [len(sites), dim],
                   'dtype': 'float32', 'windows_checksum': emb_meta['windows_checksum'],
                   'sites_checksum': sites_checksum, 'spot_check_max_abs_diff': float(max_diff),
                   'minutes': round(minutes, 1), 'date': str(date.today())}, fh, indent=1)
    shutil.rmtree(local_dir)
    print(f'{name}: {len(sites):,} sites x {dim}   {minutes:.1f} min   spot check max |diff| {max_diff:.1e}   saved\n')
