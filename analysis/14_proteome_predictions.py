# %%
import os
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))

import gzip, json, re, urllib.parse, urllib.request
from datetime import date
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Proteomes: UniProt reference proteomes of five species, canonical sequences (one per gene)
#   Saccharomyces cerevisiae is not included: "yeasts lack these enzymes" (OGT and OGA; Li et al.
#   2021, J Gen Appl Microbiol 67:33-41, abstract).
#   Source: UniProt reference proteomes, current release (knowledgebase/reference_proteomes/Eukaryota/
#   <proteome>/<proteome>_<taxon>.fasta.gz; per the release README, these files hold the canonical
#   sequences, one per gene, and *_additional.fasta the isoforms). The same source for every species;
#   a column records whether each protein is a Swiss-Prot entry (header 'sp|') or unreviewed ('tr|').
#   Each proteome is downloaded once and the UniProt release (README) is recorded with the file.
#   Each protein is then matched to our dataset (accession, isoform suffix ignored, or identical
#   sequence) to record its split and its labelled positive sites (kept only when the sequence is
#   identical to ours).
#   Outputs in OUT_DIR. The earlier Swiss-Prot human and mouse run (data_processed/proteome_predictions)
#   is not overwritten.
# ---------------------------------------------------------------------------
REFERENCE = {'human': ('UP000005640', 9606), 'mouse': ('UP000000589', 10090),
             'arabidopsis': ('UP000006548', 3702), 'rice': ('UP000059680', 39947),
             'fly': ('UP000000803', 7227)}
OUT_DIR = 'data_processed/proteome_predictions/reference_proteomes'
os.makedirs(OUT_DIR, exist_ok=True)
FTP = 'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes'

def download(name, upid, taxon):
    fasta, meta_file = f'{OUT_DIR}/{name}_{upid}_{taxon}.fasta', f'{OUT_DIR}/{name}_{upid}_{taxon}.json'
    if not os.path.exists(fasta):
        with urllib.request.urlopen(f'{FTP}/README', timeout=120) as r:
            release = re.search(r'Release (\d{4}_\d{2}), (\S+)', r.read().decode(errors='replace'))
        url = f'{FTP}/Eukaryota/{upid}/{upid}_{taxon}.fasta.gz'
        with urllib.request.urlopen(url, timeout=600) as r:
            text = gzip.decompress(r.read()).decode()
        with open(fasta, 'w') as f:
            f.write(text)
        with open(meta_file, 'w') as f:
            json.dump({'url': url, 'proteome': upid, 'taxon': taxon, 'sequences': 'canonical, one per gene',
                       'uniprot_release': release.group(1) if release else None,
                       'uniprot_release_date': release.group(2) if release else None,
                       'downloaded': str(date.today())}, f, indent=1)
    return fasta, json.load(open(meta_file))

def read_fasta(path):
    """UniProt FASTA -> accession, reviewed (Swiss-Prot), gene, protein name, sequence (a record starts
    at '>' at the start of a line; protein names may contain '>')."""
    records, header, lines = [], None, []
    def add(header, lines):
        db, acc = header.split('|')[:2]
        records.append({'accession': acc, 'reviewed': db == 'sp',
                        'gene': header.split(' GN=')[1].split(' ')[0] if ' GN=' in header else '',
                        'protein': header.split(' ', 1)[1].split(' OS=')[0], 'sequence': ''.join(lines)})
    for line in open(path):
        line = line.rstrip('\n')
        if line.startswith('>'):
            if header is not None:
                add(header, lines)
            header, lines = line[1:], []
        else:
            lines.append(line.strip())
    add(header, lines)
    return pd.DataFrame(records)

atlas = pd.read_parquet('data_processed/atlas5_proteins.parquet')
by_acc = {pid.split('-')[0]: r for pid, r in zip(atlas['protein_id'], atlas.itertuples())}
by_seq = {s: r for s, r in zip(atlas['sequence'], atlas.itertuples())}

proteomes, proteome_meta = {}, {}
for name, (upid, taxon) in REFERENCE.items():
    fasta, meta = download(name, upid, taxon)
    p = read_fasta(fasta)
    p['length'] = p['sequence'].str.len()
    assert p['accession'].is_unique
    splits, positives = [], []
    for acc, seq in zip(p['accession'], p['sequence']):
        r = by_acc.get(acc) or by_seq.get(seq)
        splits.append(r.split if r is not None else 'absent')
        positives.append(sorted(int(x) for x in r.positive) if r is not None and r.sequence == seq else [])
    p['our_split'], p['our_positives'] = splits, positives
    p.drop(columns='sequence').to_parquet(f'{OUT_DIR}/{name}_proteins.parquet', index=False)
    proteomes[name], proteome_meta[name] = p, meta

    n_st = p['sequence'].str.count('[ST]')
    other = pd.Series(list(''.join(p['sequence']))).value_counts().drop(list('ACDEFGHIKLMNPQRSTVWY'), errors='ignore')
    print(f'===== {name} ({upid}): UniProt release {meta["uniprot_release"]} ({meta["uniprot_release_date"]}), '
          f'downloaded {meta["downloaded"]}')
    print(f'Proteins: {len(p):,} (Swiss-Prot {p["reviewed"].sum():,}, unreviewed {(~p["reviewed"]).sum():,})   '
          f'residues: {p["length"].sum():,}   S/T: {n_st.sum():,}   longer than 1,022 residues: {(p["length"] > 1022).sum():,}')
    print(f'Letters outside the 20 standard amino acids: {other.to_dict()}')
    print('In our dataset (by split): ' + ', '.join(f'{k} {v:,}' for k, v in p['our_split'].value_counts().items()))
    print(f'Labelled positive sites carried over (identical sequence only): {p["our_positives"].str.len().sum():,}\n')

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
from transformers import AutoTokenizer, EsmModel
from tqdm.auto import tqdm
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Scores of every S/T of the five reference proteomes with the frozen final model
#   The scoring code is the one used for the external validation (NB05, last cell) and the case
#   studies (NB13), which repeats the feature code of NB02 unchanged; final model, Platt
#   parameters and cut-off are read from final_model.pt.
#   Proteins are scored in chunks of CHUNK proteins; each chunk is written to Drive when
#   complete and skipped on later runs, so the cell resumes after a disconnection. Proteins
#   without S/T have nothing to score and are skipped.
#   Check first: proteins of our test split whose sequence is identical to ours are compared with
#   the stored test scores (NB05).
#   Timing: wall-clock time of each chunk (windows, ESM2-150M embeddings, features, final model),
#   saved next to the chunk as chunk_NNN.json with the GPU name; the sum over the chunks of a
#   proteome is saved as results/compute/NB14_proteome_{species}.json (compute-records cell) once
#   every chunk is timed. Chunks are kept in OUT_DIR/chunks/{species}. If predictions from an earlier
#   run exist, the new ones are compared with them before they are replaced.
# ---------------------------------------------------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FINAL_DIR = 'data_processed/final_model'
LOCAL_FILE_DIR = '/content'
CHUNK = 2000

MAX_RES, OVERLAP, K = 1022, 256, 20
ESM_PATH, MAX_TOKENS = 'facebook/esm2_t30_150M_UR50D', 65536   # as in NB02 for ESM2-150M
LOCAL_FILE = f'{LOCAL_FILE_DIR}/proteome_residues.npy'
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

final = torch.load(f'{FINAL_DIR}/final_model.pt', map_location='cpu')
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


# Check: test proteins with identical sequences vs the stored test scores (human)
test_scores = pd.read_parquet(f'{FINAL_DIR}/test_scores.parquet')
test_seqs = set(atlas.loc[atlas['split'] == 'test', 'sequence'])
human = proteomes['human']
check = human[(human['our_split'] == 'test') & human['accession'].isin(test_scores['protein_id'])
              & human['sequence'].isin(test_seqs)].head(5)
got, _ = score_proteins(check['accession'].tolist(), check['sequence'].tolist())
cmp = got.merge(test_scores[['protein_id', 'position', 'logit']], on=['protein_id', 'position'], suffixes=('', '_test'))
print(f'Check on {len(check)} test proteins, {len(cmp):,} unmasked S/T: logits max |diff| '
      f'{np.abs(cmp["logit"] - cmp["logit_test"]).max():.1e}')
assert len(cmp) > 0 and np.abs(cmp['logit'] - cmp['logit_test']).max() < 1e-3

# Each proteome, in chunks
for name, p in proteomes.items():
    chunk_dir = f'{OUT_DIR}/chunks/{name}'
    os.makedirs(chunk_dir, exist_ok=True)
    n_chunks = -(-len(p) // CHUNK)
    chunk_path = lambda c: f'{chunk_dir}/chunk_{c:03d}.parquet'
    timing_path = lambda c: f'{chunk_dir}/chunk_{c:03d}.json'
    for c in tqdm(range(n_chunks), desc=f'{name}: protein chunks'):
        if os.path.exists(chunk_path(c)):
            continue
        part = p.iloc[c * CHUNK:(c + 1) * CHUNK]
        part = part[part['sequence'].str.contains('[ST]')]
        t0 = time.perf_counter()
        s, _ = score_proteins(part['accession'].tolist(), part['sequence'].tolist())
        seconds = time.perf_counter() - t0
        s.rename(columns={'protein_id': 'accession'}).to_parquet(chunk_path(c) + '.part', index=False)
        with open(timing_path(c), 'w') as f:   # written before the chunk is complete: a chunk on Drive has its time
            json.dump({'seconds': round(seconds, 2), 'proteins': len(part), 'st_sites': len(s), 'gpu': gpu_name()}, f)
        os.replace(chunk_path(c) + '.part', chunk_path(c))

    timings = [json.load(open(timing_path(c))) for c in range(n_chunks) if os.path.exists(timing_path(c))]
    if len(timings) == n_chunks:
        save_compute('NB14', f'proteome_{name}', sum(t['seconds'] for t in timings),
                     proteins=sum(t['proteins'] for t in timings), st_sites=sum(t['st_sites'] for t in timings))
    else:
        print(f'{name}: {n_chunks - len(timings)} of {n_chunks} chunks have no timing: no compute record written')

    pred = pd.concat([pd.read_parquet(chunk_path(c)) for c in range(n_chunks)], ignore_index=True)
    assert len(pred) == p['sequence'].str.count('[ST]').sum()
    info = p.set_index('accession')
    pred['reviewed'] = pred['accession'].map(info['reviewed'])
    pred['gene'] = pred['accession'].map(info['gene'])
    pred['our_split'] = pred['accession'].map(info['our_split'])
    positives = {(a, x) for a, xs in zip(p['accession'], p['our_positives']) for x in xs}
    pred['labelled_positive'] = [(a, x) in positives for a, x in zip(pred['accession'], pred['position'])]
    pred = pred[['accession', 'reviewed', 'gene', 'position', 'residue', 'logit', 'probability', 'called', 'our_split',
                 'labelled_positive']]
    if os.path.exists(f'{OUT_DIR}/{name}_proteome_predictions.parquet'):
        previous = pd.read_parquet(f'{OUT_DIR}/{name}_proteome_predictions.parquet')
        assert previous[['accession', 'position']].equals(pred[['accession', 'position']])
        print(f'{name}, previous run vs this run: logit max |diff| {(previous["logit"] - pred["logit"]).abs().max():.1e}   '
              f'calls changed {int((previous["called"] != pred["called"]).sum())}')
    pred.to_parquet(f'{OUT_DIR}/{name}_proteome_predictions.parquet', index=False)

    print(f'\n===== {name}: S/T scored {len(pred):,} in {pred["accession"].nunique():,} proteins   '
          f'called positive at the validation cut-off: {pred["called"].sum():,} ({pred["called"].mean():.1%})')
    print(pred.groupby('our_split').agg(sites=('called', 'size'), called=('called', 'sum'),
                                        labelled_positive=('labelled_positive', 'sum')).to_string())
    print(pred.groupby('reviewed').agg(sites=('called', 'size'), called=('called', 'sum'))
          .rename(index={True: 'Swiss-Prot', False: 'unreviewed'}).to_string())

# %%
import io, json
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Fraction of S/T labelled positive in Atlas 5.0, by bin of calibrated probability, over all
# proteins of our dataset (all organisms)
#   Every site of our dataset is scored with the final model from the features precomputed in
#   NB02 (ESM2-150M, target residue + mean of the +-20 window), with the code of NB05; masked
#   S/T excluded (decision D1). Check: validation and test logits are compared with the scores
#   stored by NB05. All scores are saved for later use.
#   One colour per split: training (sites seen in training), validation (used for Platt scaling,
#   so close to the diagonal by construction) and test (independent). Equal-count bins: within
#   each split, S/T are divided into N_BINS bins by probability quantiles, so every bin of a split
#   holds the same number of S/T; x = mean probability in the bin; Wilson 95% interval saved in
#   the table, not plotted. The diagonal is perfect calibration. Negatives are unverified
#   (decision 10), so the fractions are lower bounds.
#   Left: full range; right: 0-ZOOM on both axes.
#   Figure saved as PDF and RGB TIFF at 300 dpi (Frontiers rules: 180 mm wide, text >= 8 pt).
# ---------------------------------------------------------------------------
FINAL_DIR = 'data_processed/final_model'
FEAT_DIR = 'data_processed/features'
N_BINS = 20
ZOOM = 0.25
ROWS = 50000   # sites scored per step
FIG_DIR = 'results/figures/proteome_predictions'
os.makedirs(FIG_DIR, exist_ok=True)
SPLITS = {'train': ('training', '#a9a8a2'), 'val': ('validation', '#d9730d'), 'test': ('test', '#2a78d6')}

final = torch.load(f'{FINAL_DIR}/final_model.pt', map_location='cpu')
assert final['plm'] == 'ESM2-150M' and final['window'] == 'pm20'
clf = nn.Sequential(nn.Linear(len(final['mean']), final['config']['hidden']), nn.ReLU(),
                    nn.Dropout(final['config']['dropout']), nn.Linear(final['config']['hidden'], 1))
clf.load_state_dict(final['state'])
clf = clf.eval()
PLATT_A, PLATT_B = final['platt']['a'], final['platt']['b']

sites = pd.read_parquet('data_processed/embeddings/sites.parquet')
assert json.load(open(f'{FEAT_DIR}/{final["plm"]}/features.json'))['shape'][0] == len(sites)
target = np.load(f'{FEAT_DIR}/{final["plm"]}/target.npy', mmap_mode='r')
window_mean = np.load(f'{FEAT_DIR}/{final["plm"]}/mean_{final["window"]}.npy', mmap_mode='r')

logit = np.empty(len(sites), dtype=np.float32)
with torch.no_grad():
    for i in tqdm(range(0, len(sites), ROWS), desc='scoring the sites of our dataset'):
        X = torch.from_numpy(np.concatenate([np.asarray(target[i:i + ROWS]), np.asarray(window_mean[i:i + ROWS])], axis=1))
        logit[i:i + ROWS] = clf((X - final['mean']) / final['std']).squeeze(1).numpy()

scores = sites[['protein_id', 'position', 'split', 'label']].copy()
scores['logit'] = logit
scores = scores[scores['label'] != 'masked'].reset_index(drop=True)
scores['probability'] = 1 / (1 + np.exp(-(PLATT_A * scores['logit'].astype(np.float64) + PLATT_B)))   # float64, as in NB05
scores['positive'] = scores['label'] == 'positive'

for split, file in [('val', 'validation_scores'), ('test', 'test_scores')]:
    stored = pd.read_parquet(f'{FINAL_DIR}/{file}.parquet')
    cmp = scores[scores['split'] == split].merge(stored[['protein_id', 'position', 'logit']],
                                                 on=['protein_id', 'position'], suffixes=('', '_stored'))
    diff = np.abs(cmp['logit'] - cmp['logit_stored']).max()
    print(f'Check {split}: {len(cmp):,} of {len(stored):,} stored sites matched; logits max |diff| {diff:.1e}')
    assert len(cmp) == len(stored) and diff < 1e-4
scores.to_parquet(f'{FINAL_DIR}/atlas_site_scores.parquet', index=False)

def save_figure(fig, name, width_mm):
    """PDF (vector, for the LaTeX manuscript) and RGB TIFF at 300 dpi (production format)."""
    assert abs(fig.get_size_inches()[0] * 25.4 - width_mm) < 1e-6
    fig.savefig(f'{FIG_DIR}/{name}.pdf')
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=300)
    Image.open(buffer).convert('RGB').save(f'{FIG_DIR}/{name}.tif', dpi=(300, 300))

def wilson(k, n, z=1.96):
    """Wilson score interval for k successes in n trials."""
    f = k / n
    centre = (f + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * np.sqrt(f * (1 - f) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return centre - half, centre + half

scores['bin'] = scores.groupby('split')['probability'].transform(lambda x: pd.qcut(x, N_BINS, labels=False))
table = (scores.groupby(['split', 'bin'])
         .agg(sites=('positive', 'size'), positives=('positive', 'sum'), mean_probability=('probability', 'mean'),
              min_probability=('probability', 'min'), max_probability=('probability', 'max')).reset_index())
table['fraction_positive'] = table['positives'] / table['sites']
table['ci_low'], table['ci_high'] = wilson(table['positives'], table['sites'])
table.to_csv(f'{FINAL_DIR}/atlas_equal_count_bins_vs_labels.csv', index=False)

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 85 / 25.4), layout='constrained')
for ax, limit in zip(axes, [1, ZOOM]):
    ax.plot([0, limit], [0, limit], color='#cfcec8', linewidth=1, zorder=0)
    for split, (label, colour) in SPLITS.items():
        d = table[table['split'] == split]
        ax.plot(d['mean_probability'], d['fraction_positive'], linestyle='none', marker='o', markersize=4,
                color=colour, label=f'{label} ({d["sites"].sum():,} S/T, {d["positives"].sum():,} positive)')
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel('Calibrated probability (bin mean)')
axes[0].set_ylabel('Fraction labelled positive in Atlas 5.0')
axes[0].set_title(f'All {scores["protein_id"].nunique():,} proteins of our dataset', loc='left')
axes[1].set_title(f'0-{ZOOM} on both axes', loc='left')
axes[0].legend(frameon=False, loc='upper left')
save_figure(fig, 'figure_reliability_our_dataset', 180)
plt.show()

# %%
import matplotlib.pyplot as plt
import torch

# ---------------------------------------------------------------------------
# Upper tail of the calibrated probability over all S/T of each proteome
#   One panel per species: probability against the fraction of S/T at or above it (log scale); the
#   species of the panel in blue, the other species in light grey for comparison. Orange line:
#   validation cut-off.
#   Table: proteins, Swiss-Prot share, probability at given top fractions of S/T, and the fraction
#   called at the cut-off (all proteins and Swiss-Prot proteins only).
#   Figure saved as PDF and RGB TIFF (save_figure and FIG_DIR of the previous cell).
# ---------------------------------------------------------------------------
BLUE, LIGHT, ORANGE = '#2a78d6', '#cfcec8', '#d9730d'
TOP_FRACTIONS = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5]
CUTOFF = torch.load('data_processed/final_model/final_model.pt', map_location='cpu')['cutoff']['probability']

top = np.logspace(-5, 0, 501)
tails, rows = {}, {}
for name in proteomes:
    pred = pd.read_parquet(f'{OUT_DIR}/{name}_proteome_predictions.parquet', columns=['accession', 'reviewed', 'probability'])
    prob = pred['probability'].to_numpy()
    tails[name] = np.quantile(prob, 1 - top)
    rows[name] = {'proteins': f'{pred["accession"].nunique():,}',
                  'Swiss-Prot proteins': f'{pred.drop_duplicates("accession")["reviewed"].mean():.1%}',
                  'S/T': f'{len(prob):,}', 'called at cut-off': f'{(prob >= CUTOFF).mean():.2%}',
                  'called, Swiss-Prot only': f'{(pred.loc[pred["reviewed"], "probability"] >= CUTOFF).mean():.2%}',
                  **{f'p at top {f:.2%}': f'{np.quantile(prob, 1 - f):.3f}' for f in TOP_FRACTIONS}}

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False})
n_cols = 3
n_rows = -(-len(proteomes) // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(180 / 25.4, 55 * n_rows / 25.4), sharex=True, sharey=True,
                         layout='constrained', squeeze=False)
for ax, name in zip(axes.flat, proteomes):
    for other in proteomes:
        if other != name:
            ax.plot(top, tails[other], color=LIGHT, linewidth=1)
    ax.plot(top, tails[name], color=BLUE, linewidth=1.5)
    ax.axhline(CUTOFF, color=ORANGE, linewidth=1)
    ax.set_xscale('log')
    ax.set_xlim(1e-5, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f'{name.capitalize()}\n{rows[name]["S/T"]} S/T, {rows[name]["called at cut-off"]} called', loc='left')
axes[0, 0].text(1.2e-5, CUTOFF, f'validation cut-off {CUTOFF:.3f}', color=ORANGE, va='bottom')
for ax in list(axes.flat)[len(proteomes):]:
    ax.axis('off')
for k, ax in enumerate(list(axes.flat)[:len(proteomes)]):
    if k % n_cols == 0:
        ax.set_ylabel('Calibrated probability')
    if k + n_cols >= len(proteomes):   # lowest panel of its column: x-axis numbers and label
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.set_xlabel('Fraction of S/T at or above')
save_figure(fig, 'figure_probability_tail_reference_proteomes', 180)
plt.show()

print(pd.DataFrame(rows).to_string())

# %%
import re, subprocess, sys
try:
    from Bio import Align
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'biopython'], check=True)
    from Bio import Align
from Bio.Align import substitution_matrices
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Candidate new O-GlcNAc sites: high-probability S/T not reported in Atlas 5.0 or UniProt
# (each of the five reference proteomes)
#   Site kept if: (1) calibrated probability >= HIGH_P; (2) not a positive or masked S/T of
#   Atlas 5.0 (our dataset); (3) not annotated as O-GlcNAc in UniProt (glycosylation features
#   'O-linked (GlcNAc...'). Protein kept if (4) UniProt protein existence is 'Evidence at
#   protein level'. Recorded, not filtered: our split and the known sites of the protein.
#   Atlas sites are transferred to the current UniProt sequence: directly when the sequences
#   are identical; otherwise by global alignment (BLOSUM62, gap open -10, extend -0.5, free end
#   gaps), keeping positions whose aligned residues are identical. Atlas entries are those with
#   the same accession (isoform suffix ignored) or the identical sequence.
#   UniProt fields are downloaded once per species for every entry of its reference proteome
#   (query proteome:<proteome>; checked to cover every protein of cell 0) and the release is recorded.
# ---------------------------------------------------------------------------
HIGH_P = 0.160
MAX_ALIGN = 10000   # longest sequence aligned (memory); longer pairs are counted, not aligned

aligner = Align.PairwiseAligner()
aligner.mode = 'global'
aligner.substitution_matrix = substitution_matrices.load('BLOSUM62')
aligner.open_gap_score, aligner.extend_gap_score = -10, -0.5
try:
    aligner.end_insertion_score = aligner.end_deletion_score = 0     # free end gaps (Biopython >= 1.85)
except AttributeError:
    aligner.target_end_gap_score = aligner.query_end_gap_score = 0  # earlier Biopython names

def position_map(a, b):
    """1-based positions of sequence a mapped to b where the aligned residues are identical."""
    aln = aligner.align(a, b)[0]
    mapping = {}
    for (s1, e1), (s2, _) in zip(*aln.aligned):
        for k in range(e1 - s1):
            if a[s1 + k] == b[s2 + k]:
                mapping[s1 + k + 1] = s2 + k + 1
    return mapping

def uniprot_fields(name, upid):
    """Protein existence and O-GlcNAc glycosylation positions of every entry of the reference proteome."""
    path, meta_file = f'{OUT_DIR}/{name}_uniprot_fields.tsv', f'{OUT_DIR}/{name}_uniprot_fields.json'
    if not os.path.exists(path):
        query = f'proteome:{upid}'
        url = ('https://rest.uniprot.org/uniprotkb/stream?format=tsv&fields=accession,protein_existence,ft_carbohyd'
               '&query=' + urllib.parse.quote(query))
        with urllib.request.urlopen(url, timeout=600) as r:
            release = r.headers.get('X-UniProt-Release')
            text = r.read().decode()
        with open(path, 'w') as f:
            f.write(text)
        with open(meta_file, 'w') as f:
            json.dump({'query': query, 'fields': 'accession,protein_existence,ft_carbohyd', 'uniprot_release': release,
                       'downloaded': str(date.today())}, f, indent=1)
    t = pd.read_csv(path, sep='\t').fillna('')
    t['oglcnac'] = [sorted(int(m.group(1)) for m in re.finditer(r'CARBOHYD (\d+); /note="O-linked \(GlcNAc', s))
                    for s in t['Glycosylation']]
    return t.set_index('Entry'), json.load(open(meta_file))

atlas_by_acc, atlas_by_seq = {}, {}
for r in atlas.itertuples():
    atlas_by_acc.setdefault(r.protein_id.split('-')[0], []).append(r)
    atlas_by_seq.setdefault(r.sequence, []).append(r)

for name, p in proteomes.items():
    fields, meta = uniprot_fields(name, REFERENCE[name][0])
    assert p['accession'].isin(fields.index).all()
    print(f'===== {name}: UniProt fields release {meta["uniprot_release"]} (proteome FASTA: '
          f'{proteome_meta[name]["uniprot_release"]})')

    # Atlas positive and masked S/T on the current sequence
    atlas_pos, atlas_masked = set(), set()
    n_identical = n_aligned = n_too_long = 0
    for acc, seq in tqdm(list(zip(p['accession'], p['sequence'])), desc=f'{name}: Atlas sites'):
        hits = {r.protein_id: r for r in atlas_by_acc.get(acc, []) + atlas_by_seq.get(seq, [])}.values()
        for r in hits:
            if r.sequence == seq:
                mapping, n_identical = None, n_identical + 1
            elif max(len(r.sequence), len(seq)) > MAX_ALIGN:
                n_too_long += 1
                continue
            else:
                mapping, n_aligned = position_map(r.sequence, seq), n_aligned + 1
            for target, positions in [(atlas_pos, r.positive), (atlas_masked, r.masked)]:
                for x in positions:
                    x = int(x) if mapping is None else mapping.get(int(x))
                    if x is not None:
                        target.add((acc, x))
    atlas_masked -= atlas_pos
    print(f'Atlas entries matched: {n_identical:,} identical sequence, {n_aligned:,} aligned, '
          f'{n_too_long:,} too long to align')

    pred = pd.read_parquet(f'{OUT_DIR}/{name}_proteome_predictions.parquet')
    keys = list(zip(pred['accession'], pred['position']))
    uniprot_sites = {(a, x) for a, xs in fields['oglcnac'].items() for x in xs}
    pred['atlas_positive'] = [k in atlas_pos for k in keys]
    pred['atlas_masked'] = [k in atlas_masked for k in keys]
    pred['uniprot_oglcnac'] = [k in uniprot_sites for k in keys]
    pred['protein_existence'] = pred['accession'].map(fields['Protein existence'])
    pred['high'] = pred['probability'] >= HIGH_P
    pred['new'] = pred['high'] & ~pred['atlas_positive'] & ~pred['atlas_masked'] & ~pred['uniprot_oglcnac']
    pred['candidate'] = pred['new'] & (pred['protein_existence'] == 'Evidence at protein level')

    per = pred.groupby('accession').agg(st=('position', 'size'), high=('high', 'sum'),
                                        known_atlas=('atlas_positive', 'sum'), masked_atlas=('atlas_masked', 'sum'),
                                        known_uniprot=('uniprot_oglcnac', 'sum'), new_candidates=('candidate', 'sum'))
    cand = pred[pred['candidate']].drop(columns=['high', 'new', 'candidate', 'atlas_positive', 'atlas_masked',
                                                 'uniprot_oglcnac', 'labelled_positive'])
    cand = cand.join(per, on='accession')
    cand['protein'] = cand['accession'].map(p.set_index('accession')['protein'])
    cand.to_parquet(f'{OUT_DIR}/{name}_new_site_candidates.parquet', index=False)

    funnel = [('S/T scored', len(pred)),
              (f'(1) probability >= {HIGH_P}', pred['high'].sum()),
              ('(2) not Atlas 5.0 positive or masked', (pred['high'] & ~pred['atlas_positive'] & ~pred['atlas_masked']).sum()),
              ('(3) not UniProt O-GlcNAc', pred['new'].sum()),
              ('(4) protein existence: protein level', pred['candidate'].sum())]
    print(pd.DataFrame(funnel, columns=['filter', 'sites']).to_string(index=False, formatters={'sites': '{:,}'.format}))
    print(f'Candidate sites: {len(cand):,} in {cand["accession"].nunique():,} proteins; by our split: '
          + ', '.join(f'{k} {v:,}' for k, v in cand.drop_duplicates('accession')['our_split'].value_counts().items())
          + ' (proteins)')
    print(f'Labelled positives gained by alignment (vs identical sequence only): '
          f'{pred["atlas_positive"].sum() - pred["labelled_positive"].sum():,}\n')

# %%
# ---------------------------------------------------------------------------
# Candidate new sites at a stricter threshold, for the search of case studies
#   From the candidates of the previous cell (new in Atlas 5.0 and UniProt, protein existence at
#   protein level), sites with calibrated probability >= STRICT_P. One row per protein, ranked by
#   the number of such sites (ties by maximum probability); sites listed as residue+position
#   (probability). Known sites of the protein (Atlas positive and masked, transferred to the
#   current sequence; UniProt O-GlcNAc) shown for context.
# ---------------------------------------------------------------------------
STRICT_P = 0.5

for name in proteomes:
    cand = pd.read_parquet(f'{OUT_DIR}/{name}_new_site_candidates.parquet')
    strict = cand[cand['probability'] >= STRICT_P].sort_values('position')
    strict['site'] = [f'{r}{x} ({q:.2f})' for r, x, q in zip(strict['residue'], strict['position'], strict['probability'])]
    per = (strict.groupby('accession')
           .agg(gene=('gene', 'first'), protein=('protein', 'first'), split=('our_split', 'first'), st=('st', 'first'),
                new_sites=('site', 'size'), max_p=('probability', 'max'), known_atlas=('known_atlas', 'first'),
                masked_atlas=('masked_atlas', 'first'), known_uniprot=('known_uniprot', 'first'),
                sites=('site', ', '.join))
           .sort_values(['new_sites', 'max_p'], ascending=False))
    per['protein'] = per['protein'].str.slice(0, 40)
    per.to_csv(f'{OUT_DIR}/{name}_new_site_candidates_p{STRICT_P}.csv')

    print(f'===== {name}: new sites with probability >= {STRICT_P}: {len(strict):,} in {len(per):,} proteins; '
          'by our split: ' + ', '.join(f'{k} {v:,}' for k, v in per['split'].value_counts().items()) + ' (proteins)')
    with pd.option_context('display.max_colwidth', 120, 'display.width', 250):
        print(per.to_string(float_format='{:.2f}'.format))
    print()

# %%
import time
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Annotation of the candidate new sites (probability >= STRICT_P) for the choice of case studies
#   Per protein (UniProt entry, current release; Europe PMC):
#     location     subcellular location of the canonical protein (UniProt comment)
#     disease      diseases linked to the protein (UniProt 'Involvement in disease')
#     epmc_oglcnac papers whose title or abstract mention the gene symbol or the recommended
#                  protein name together with O-GlcNAc terms (Europe PMC; ambiguous symbols
#                  inflate the count; the papers themselves are read later)
#   Per site (UniProt features):
#     topology     topological domain, transmembrane segment, signal or transit peptide at the
#                  site ('-' if none annotated)
#     regions      domains, zinc fingers, repeats, motifs, compositional bias and regions
#                  (e.g. 'Disordered') containing the site
#     nearby       active sites, binding sites, sites, modified residues, glycosylation and
#                  cross-links within +-NEAR residues (type, description, position)
#   UniProt entries and Europe PMC counts are saved on first download and reused.
# ---------------------------------------------------------------------------
NEAR = 5
ANNOT_DIR = f'{OUT_DIR}/case_study_annotation'
os.makedirs(f'{ANNOT_DIR}/uniprot', exist_ok=True)
TOPOLOGY = ('Topological domain', 'Transmembrane', 'Intramembrane', 'Signal', 'Transit peptide')
REGIONS = ('Domain', 'Zinc finger', 'Repeat', 'Motif', 'Compositional bias', 'Region', 'Coiled coil', 'DNA binding')
POINTS = ('Active site', 'Binding site', 'Site', 'Modified residue', 'Glycosylation', 'Cross-link', 'Disulfide bond')
OGLCNAC_TERMS = ['O-GlcNAc', 'O-GlcNAcylation', 'O-GlcNAcylated', 'O-linked N-acetylglucosamine']

def get_json(url, tries=5):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)

def uniprot_entry(acc):
    path = f'{ANNOT_DIR}/uniprot/{acc}.json'
    if not os.path.exists(path):
        json.dump(get_json(f'https://rest.uniprot.org/uniprotkb/{acc}.json'), open(path, 'w'))
    return json.load(open(path))

def field_or(field, terms):
    return ' OR '.join(f'{field}:"{t}"' for t in terms)

epmc_file = f'{ANNOT_DIR}/europepmc_counts.json'
epmc = json.load(open(epmc_file)) if os.path.exists(epmc_file) else {}

def epmc_count(gene, protein):
    names = [n for n in (gene, protein) if n]
    if not names:
        return 0
    query = (f'({field_or("TITLE", names)} OR {field_or("ABSTRACT", names)}) AND '
             f'({field_or("TITLE", OGLCNAC_TERMS)} OR {field_or("ABSTRACT", OGLCNAC_TERMS)})')
    if query not in epmc:
        d = get_json('https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json&pageSize=1&query='
                     + urllib.parse.quote(query))
        epmc[query] = {'hits': d['hitCount'], 'date': str(date.today())}
        json.dump(epmc, open(epmc_file, 'w'), indent=1)
        time.sleep(0.2)
    return epmc[query]['hits']

def span(f):
    return int(f['location']['start']['value'] or 0), int(f['location']['end']['value'] or 0)

for name in proteomes:
    strict = pd.read_parquet(f'{OUT_DIR}/{name}_new_site_candidates.parquet')
    strict = strict[strict['probability'] >= STRICT_P].sort_values(['accession', 'position'])
    rows, proteins = [], {}
    for acc in tqdm(strict['accession'].unique(), desc=f'{name}: annotation'):
        e = uniprot_entry(acc)
        location = sorted({l['location']['value'] for c in e.get('comments', [])
                           if c['commentType'] == 'SUBCELLULAR LOCATION' and 'molecule' not in c
                           for l in c.get('subcellularLocations', [])})
        disease = [c['disease']['diseaseId'] for c in e.get('comments', []) if c['commentType'] == 'DISEASE' and 'disease' in c]
        gene = e['genes'][0]['geneName']['value'] if e.get('genes') and 'geneName' in e['genes'][0] else ''
        d = e['proteinDescription']   # unreviewed entries have submission names instead of a recommended name
        protein = d.get('recommendedName', (d.get('submissionNames') or [{}])[0]).get('fullName', {}).get('value', '')
        proteins[acc] = {'location': '; '.join(location) or '-', 'disease': '; '.join(disease) or '-',
                         'epmc_oglcnac': epmc_count(gene, protein)}
        feats = e.get('features', [])
        for r in strict[strict['accession'] == acc].itertuples():
            x = r.position
            topology = [f"{f['type']}" + (f": {f['description']}" if f.get('description') else '')
                        for f in feats if f['type'] in TOPOLOGY and span(f)[0] <= x <= span(f)[1]]
            regions = [f"{f['type']}: {f.get('description', '')}" for f in feats
                       if f['type'] in REGIONS and span(f)[0] <= x <= span(f)[1]]
            nearby = [f"{f['type']}: {f.get('description', '')} @{span(f)[0]}" for f in feats
                      if f['type'] in POINTS and (abs(span(f)[0] - x) <= NEAR or abs(span(f)[1] - x) <= NEAR)]
            rows.append({'accession': acc, 'gene': r.gene, 'split': r.our_split, 'site': f'{r.residue}{x}',
                         'probability': r.probability, 'topology': '; '.join(topology) or '-',
                         'regions': '; '.join(regions) or '-', 'nearby': '; '.join(nearby) or '-',
                         **proteins[acc]})
    annot = pd.DataFrame(rows)
    annot.to_csv(f'{ANNOT_DIR}/{name}_candidates_p{STRICT_P}_annotated.csv', index=False)

    print(f'===== {name}: {len(annot)} sites in {annot["accession"].nunique()} proteins (probability >= {STRICT_P})')
    order = (annot.groupby('accession').agg(n=('site', 'size'), p=('probability', 'max'))
             .sort_values(['n', 'p'], ascending=False).index)
    for acc in order:
        a = annot[annot['accession'] == acc]
        info = proteins[acc]
        print(f'\n{acc} {a["gene"].iloc[0]} ({a["split"].iloc[0]}) | location: {info["location"]} | '
              f'O-GlcNAc papers (Europe PMC): {info["epmc_oglcnac"]} | disease: {info["disease"]}')
        for r in a.itertuples():
            print(f'   {r.site} {r.probability:.2f} | topology: {r.topology} | regions: {r.regions} | nearby: {r.nearby}')
    print()

# %%
# ---------------------------------------------------------------------------
# Human-mouse comparison of one candidate protein (case-study screening): compare_pair(human, mouse)
#   The human and mouse reference-proteome sequences are aligned (global, BLOSUM62, gap open -10,
#   extend -0.5, free end gaps; same aligner as the Atlas transfer above). Table: every S/T with
#   probability >= show_p in either species, or known in either (Atlas 5.0 positive or masked,
#   transferred to the current sequence; UniProt O-GlcNAc), with the aligned position and residue
#   of the other species and the UniProt regions of the human protein containing it.
#   Alignment blocks around the human sites with probability >= STRICT_P (+- flank residues);
#   marks under each sequence: K known site, # probability >= STRICT_P, + probability >= show_p.
#   The table is saved as case_study_annotation/pair_{human}_{mouse}.csv.
# ---------------------------------------------------------------------------
_predictions, _fields = {}, {}

def predictions(name):
    if name not in _predictions:
        _predictions[name] = pd.read_parquet(f'{OUT_DIR}/{name}_proteome_predictions.parquet',
                                             columns=['accession', 'position', 'probability'])
    return _predictions[name]

def glycosylation_field(name, acc):
    if name not in _fields:
        _fields[name] = pd.read_csv(f'{OUT_DIR}/{name}_uniprot_fields.tsv', sep='\t').fillna('').set_index('Entry')
    return _fields[name].at[acc, 'Glycosylation']

def gapped(aln, a, b):
    """Gapped strings of a global alignment and, per column, the 1-based positions (None at gaps)."""
    ga, gb, pa, pb = [], [], [], []
    c = aln.coordinates
    for k in range(c.shape[1] - 1):
        (i0, i1), (j0, j1) = (c[0, k], c[0, k + 1]), (c[1, k], c[1, k + 1])
        for t in range(max(i1 - i0, j1 - j0)):
            ia = i0 + t if i1 > i0 else None
            jb = j0 + t if j1 > j0 else None
            ga.append(a[ia] if ia is not None else '-')
            gb.append(b[jb] if jb is not None else '-')
            pa.append(ia + 1 if ia is not None else None)
            pb.append(jb + 1 if jb is not None else None)
    return ''.join(ga), ''.join(gb), pa, pb

def known_sites(name, acc, seq):
    """Atlas positive / masked (transferred to seq) and UniProt O-GlcNAc positions of one protein."""
    status = {}
    hits = {r.protein_id: r for r in atlas_by_acc.get(acc, []) + atlas_by_seq.get(seq, [])}.values()
    for r in hits:
        mapping = None if r.sequence == seq else position_map(r.sequence, seq)
        for label, positions in [('Atlas masked', r.masked), ('Atlas positive', r.positive)]:
            for x in positions:
                x = int(x) if mapping is None else mapping.get(int(x))
                if x is not None:
                    status[x] = label
    for m in re.finditer(r'CARBOHYD (\d+); /note="O-linked \(GlcNAc', glycosylation_field(name, acc)):
        status.setdefault(int(m.group(1)), 'UniProt')
    return status

def compare_pair(human, mouse, show_p=0.160, flank=15):
    pair = {'human': human, 'mouse': mouse}
    seq, prob, status = {}, {}, {}
    for name, acc in pair.items():
        seq[name] = proteomes[name].set_index('accession').at[acc, 'sequence']
        pred = predictions(name)
        prob[name] = pred[pred['accession'] == acc].set_index('position')['probability'].to_dict()
        status[name] = known_sites(name, acc, seq[name])

    aln = aligner.align(seq['human'], seq['mouse'])[0]
    gh, gm, ph, pm = gapped(aln, seq['human'], seq['mouse'])
    both = [(a, b) for a, b in zip(gh, gm) if a != '-' and b != '-']
    genes = {n: proteomes[n].set_index('accession').at[a, 'gene'] for n, a in pair.items()}
    print(f'Human {genes["human"]} {human} ({len(seq["human"])} aa) vs mouse {genes["mouse"]} {mouse} '
          f'({len(seq["mouse"])} aa): identity {sum(a == b for a, b in both) / len(both):.1%} '
          f'over {len(both)} aligned residues')

    regions = [(f.get('description', ''), *span(f)) for f in uniprot_entry(human).get('features', [])
               if f['type'] in REGIONS + ('Domain',)]
    rows = []
    for a, b, x, y in zip(gh, gm, ph, pm):
        if not any(z is not None and (z in status[n] or prob[n].get(z, 0) >= show_p) for n, z in [('human', x), ('mouse', y)]):
            continue
        rows.append({'human': f'{a}{x}' if x else '-', 'p_human': prob['human'].get(x),
                     'known_human': status['human'].get(x, '-'),
                     'mouse': f'{b}{y}' if y else '-', 'p_mouse': prob['mouse'].get(y),
                     'known_mouse': status['mouse'].get(y, '-'), 'same_residue': a == b,
                     'human_regions': '; '.join(d for d, s, e in regions if x and s <= x <= e) or '-'})
    table = pd.DataFrame(rows)
    table.to_csv(f'{ANNOT_DIR}/pair_{human}_{mouse}.csv', index=False)
    with pd.option_context('display.max_colwidth', 80, 'display.width', 250):
        print(table.to_string(index=False, float_format='{:.2f}'.format, na_rep='-'))

    def marks(name, positions, lo, hi):
        out = []
        for x in positions[lo:hi]:
            q = prob[name].get(x) if x is not None else None
            out.append('K' if x in status[name] else ' ' if q is None else
                       '#' if q >= STRICT_P else '+' if q >= show_p else ' ')
        return ''.join(out)

    blocks = []
    for k in [k for k, x in enumerate(ph) if x is not None and prob['human'].get(x, 0) >= STRICT_P]:
        lo, hi = max(k - flank, 0), min(k + flank + 1, len(ph))
        if blocks and lo <= blocks[-1][1]:
            blocks[-1][1] = hi
        else:
            blocks.append([lo, hi])
    for lo, hi in blocks:
        first = lambda positions: next((x for x in positions[lo:hi] if x is not None), '-')
        print(f'\nhuman {first(ph):>5}  {gh[lo:hi]}\n'
              f'             {marks("human", ph, lo, hi)}\n'
              f'             {"".join("|" if a == b else " " for a, b in zip(gh[lo:hi], gm[lo:hi]))}\n'
              f'mouse {first(pm):>5}  {gm[lo:hi]}\n'
              f'             {marks("mouse", pm, lo, hi)}')
    print('\nK known site (Atlas 5.0 or UniProt)   # probability >= STRICT_P   + probability >= show_p   | identical residue')
    return table

# %%
t = compare_pair('Q92750', 'G5E8Z2')   # TAF4B

# %%
t = compare_pair('O75112', 'Q9JKS4')   # LDB3

# %%
t = compare_pair('Q96RK0', 'Q924A2')   # CIC

# %%
t = compare_pair('Q96BD5', 'Q6ZPK0')   # PHF21A

# %%
t = compare_pair('O43670', 'Q9JMD0')   # ZNF207

# %%
t = compare_pair('P15408', 'P47930')   # FOSL2

# %%
# ---------------------------------------------------------------------------
# Literature check of the shortlisted proteins (probability >= STRICT_P): Europe PMC search
# with every UniProt name of the protein
#   Names (UniProt entry, current release): gene name and gene synonyms; recommended full and
#   short names; alternative full and short names; submission names (unreviewed entries). A paper is counted when its title or abstract
#   contains any of the names and any O-GlcNAc term. Names of <= SHORT_NAME characters are kept
#   but listed, as they can match unrelated text. The first TITLES hits (most cited first) are
#   listed for reading. Results saved per query and reused.
# ---------------------------------------------------------------------------
SHORT_NAME = 3
TITLES = 5

def uniprot_names(e):
    names = []
    for g in e.get('genes', []):
        names += [g['geneName']['value']] if 'geneName' in g else []
        names += [s['value'] for s in g.get('synonyms', [])]
    d = e.get('proteinDescription', {})
    for block in [d.get('recommendedName', {})] + d.get('alternativeNames', []) + d.get('submissionNames', []):
        names += [block['fullName']['value']] if 'fullName' in block else []
        names += [s['value'] for s in block.get('shortNames', [])]
    return list(dict.fromkeys(n.replace('"', '') for n in names))

epmc_names_file = f'{ANNOT_DIR}/europepmc_all_names.json'
epmc_names = json.load(open(epmc_names_file)) if os.path.exists(epmc_names_file) else {}

def epmc_search(names):
    query = (f'({field_or("TITLE", names)} OR {field_or("ABSTRACT", names)}) AND '
             f'({field_or("TITLE", OGLCNAC_TERMS)} OR {field_or("ABSTRACT", OGLCNAC_TERMS)})')
    if not names:
        return {'hits': 0, 'first': []}
    if query not in epmc_names:
        d = get_json('https://www.ebi.ac.uk/europepmc/webservices/rest/search?format=json&resultType=lite'
                     f'&sort=CITED%20desc&pageSize={TITLES}&query=' + urllib.parse.quote(query))
        hits = [{'id': r.get('pmid') or r.get('id'), 'year': r.get('pubYear'), 'title': r.get('title')}
                for r in d.get('resultList', {}).get('result', [])]
        epmc_names[query] = {'hits': d['hitCount'], 'first': hits, 'date': str(date.today())}
        json.dump(epmc_names, open(epmc_names_file, 'w'), indent=1)
        time.sleep(0.2)
    return epmc_names[query]

for name in proteomes:
    annot = pd.read_csv(f'{ANNOT_DIR}/{name}_candidates_p{STRICT_P}_annotated.csv')
    rows = []
    for acc in tqdm(annot['accession'].unique(), desc=f'{name}: Europe PMC'):
        names = uniprot_names(uniprot_entry(acc))
        r = epmc_search(names)
        rows.append({'accession': acc, 'gene': annot.loc[annot['accession'] == acc, 'gene'].iloc[0],
                     'epmc_symbol_and_name': annot.loc[annot['accession'] == acc, 'epmc_oglcnac'].iloc[0],
                     'epmc_all_names': r['hits'], 'names': '; '.join(names),
                     'short_names': '; '.join(n for n in names if len(n) <= SHORT_NAME) or '-', 'first': r['first']})
    lit = pd.DataFrame(rows).sort_values('epmc_all_names')
    lit.drop(columns='first').to_csv(f'{ANNOT_DIR}/{name}_candidates_p{STRICT_P}_europepmc.csv', index=False)

    print(f'===== {name}: {len(lit)} proteins; with no O-GlcNAc paper (all names): {(lit["epmc_all_names"] == 0).sum()}')
    for r in lit.itertuples():
        print(f'\n{r.accession} {r.gene}: {r.epmc_all_names} papers (symbol + recommended name only: '
              f'{r.epmc_symbol_and_name}) | short names: {r.short_names}\n   names: {r.names}')
        for h in r.first:
            print(f'   - {h["year"]} {h["id"]}: {h["title"]}')
    print()

# %%
# ---------------------------------------------------------------------------
# Case study 2 (LDB3): labels of the T399/T400 motif in our dataset
#   Every protein of our dataset containing MOTIF (human LDB3 395-405, identical in mouse Ldb3;
#   contains T399/T400) is listed with its split and organism; for each S/T within +- FLANK
#   residues of the motif: the label used in training/evaluation (sites.parquet: positive,
#   negative or masked) and the final model's calibrated probability (atlas_site_scores.parquet,
#   cell 3; masked sites are not scored). Also: whether our mouse Ldb3 entry has the current
#   Swiss-Prot sequence (Q9JKS4).
# ---------------------------------------------------------------------------
MOTIF = 'PRVVTTASIRP'
FLANK = 25

site_labels = pd.read_parquet('data_processed/embeddings/sites.parquet', columns=['protein_id', 'position', 'label', 'split'])
site_scores = pd.read_parquet('data_processed/final_model/atlas_site_scores.parquet', columns=['protein_id', 'position', 'probability'])

mouse_seq = proteomes['mouse'].set_index('accession').at['Q9JKS4', 'sequence']
for r in atlas_by_acc.get('Q9JKS4', []):
    print(f'Our entry {r.protein_id} (split {r.split}): sequence identical to Swiss-Prot Q9JKS4: {r.sequence == mouse_seq}')

hits = atlas[atlas['sequence'].str.contains(MOTIF, regex=False)]
print(f'\nProteins of our dataset containing {MOTIF}: {len(hits)}')
for r in hits.itertuples():
    starts = [k for k in range(len(r.sequence)) if r.sequence.startswith(MOTIF, k)]
    print(f'\n{r.protein_id} | split {r.split} | organism {getattr(r, "organism", "-")} | motif at '
          + ', '.join(str(s + 1) for s in starts))
    lab = site_labels[site_labels['protein_id'] == r.protein_id].set_index('position')
    prob = site_scores[site_scores['protein_id'] == r.protein_id].set_index('position')['probability']
    for s in starts:
        lo, hi = max(s + 1 - FLANK, 1), min(s + len(MOTIF) + FLANK, len(r.sequence))
        rows = [{'position': x, 'residue': r.sequence[x - 1], 'in_motif': s + 1 <= x < s + 1 + len(MOTIF),
                 'label': lab['label'].get(x, '-'), 'probability': prob.get(x)}
                for x in range(lo, hi + 1) if r.sequence[x - 1] in 'ST']
        print(f'   {r.sequence[lo - 1:hi]}  ({lo}-{hi})')
        print(pd.DataFrame(rows).to_string(index=False, float_format='{:.2f}'.format, na_rep='-'))

# %%
import io
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------
# Figure: case study 2, LDB3 (ZASP) - predicted O-GlcNAc sites in human, with mouse Ldb3 aligned
#   (A) UniProt annotation of human LDB3 (O75112): domains (dark) and disordered regions (light).
#   (B) Human: calibrated probability of every S/T (stems); the TOP_N highest ranked S/T in blue,
#       labelled with their rank; the next one (rank TOP_N + 1) labelled in grey.
#   (C) Mouse Ldb3 (Q9JKS4, canonical) on human coordinates (global alignment as above): calibrated
#       probability of every aligned S/T, residues aligned to the blue human S/T in blue;
#       O-GlcNAcAtlas 5.0 positives of mouse Ldb3 (transferred from our training isoform Q9JKS4-4
#       to the canonical sequence) as dark diamonds.
#   (D) Human and mouse sequences around the predicted sites (human ZOOM): blue, the residues of (B)
#       and (C); bold with a diamond below, O-GlcNAcAtlas 5.0 positive (mouse); '|' identical residue.
#   Orange line: validation cut-off.
#   Frontiers figure rules: 180 mm wide, text >= 8 pt, panels labelled (A)-(D), solid lines only,
#   no red or green; PDF and RGB TIFF at 300 dpi.
# ---------------------------------------------------------------------------
HUMAN, MOUSE = 'O75112', 'Q9JKS4'
ZOOM = (378, 414)
TOP_N = 2   # highest ranked human S/T shown in blue (T399 and T400 in the text)
FIG_DIR = 'results/figures/case_studies'
os.makedirs(FIG_DIR, exist_ok=True)
BLUE, GREY, LIGHT, DARK, ORANGE, INK = '#2a78d6', '#a9a8a2', '#cfcec8', '#8a8984', '#d9730d', '#4a4945'
EXT_CUTOFF = torch.load('data_processed/final_model/final_model.pt', map_location='cpu')['cutoff']['probability']

def save_figure(fig, name, width_mm):
    """PDF (vector, for the LaTeX manuscript) and RGB TIFF at 300 dpi (production format)."""
    assert abs(fig.get_size_inches()[0] * 25.4 - width_mm) < 1e-6
    fig.savefig(f'{FIG_DIR}/{name}.pdf')
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=300)
    Image.open(buffer).convert('RGB').save(f'{FIG_DIR}/{name}.tif', dpi=(300, 300))

def ordinal(n):
    return f'{n}{"th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")}'

seq = {'human': proteomes['human'].set_index('accession').at[HUMAN, 'sequence'],
       'mouse': proteomes['mouse'].set_index('accession').at[MOUSE, 'sequence']}
prob = {n: predictions(n)[predictions(n)['accession'] == a].set_index('position')['probability']
        for n, a in [('human', HUMAN), ('mouse', MOUSE)]}
known_mouse = {x for x, s in known_sites('mouse', MOUSE, seq['mouse']).items() if s == 'Atlas positive'}
assert not any(s == 'Atlas positive' for s in known_sites('human', HUMAN, seq['human']).values())

aln = aligner.align(seq['human'], seq['mouse'])[0]
gh, gm, ph, pm = gapped(aln, seq['human'], seq['mouse'])
to_human = {y: x for x, y in zip(ph, pm) if x is not None and y is not None}
L = len(seq['human'])

# Ranks of the human S/T (1 = highest probability)
rank_h = prob['human'].rank(ascending=False, method='min').astype(int)
top = sorted(rank_h[rank_h <= TOP_N].index)
next_ranked = sorted(rank_h[rank_h == TOP_N + 1].index)

features = uniprot_entry(HUMAN)['features']
domains = [(f['description'], *span(f)) for f in features if f['type'] == 'Domain']
disordered = [span(f) for f in features if f['type'] == 'Region' and f.get('description') == 'Disordered']

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False})
fig = plt.figure(figsize=(180 / 25.4, 125 / 25.4), layout='constrained')
grid = fig.add_gridspec(4, 1, height_ratios=[0.45, 2, 1.6, 1.1])
ax_dom = fig.add_subplot(grid[0])
ax_h = fig.add_subplot(grid[1], sharex=ax_dom)
ax_m = fig.add_subplot(grid[2], sharex=ax_dom)
ax_seq = fig.add_subplot(grid[3])

# (A) annotation
ax_dom.plot([1, L], [0, 0], color=GREY, linewidth=1.5)
for s, e in disordered:
    ax_dom.fill_between([s, e], -0.25, 0.25, color=LIGHT, linewidth=0)
for name, s, e in domains:
    ax_dom.fill_between([s, e], -0.4, 0.4, color=DARK, linewidth=0)
    ax_dom.text((s + e) / 2, 0, name.replace(' zinc-binding', ''), ha='center', va='center', color='white')
ax_dom.set_ylim(-0.6, 0.6)
ax_dom.axis('off')
ax_dom.set_title('(A) Human LDB3 (O75112): domains (dark) and disordered regions (light), UniProt', loc='left')

# (B) human, (C) mouse on human coordinates
def stems(ax, x, p, highlight):
    for mask, colour, z in [(~highlight, GREY, 1), (highlight, BLUE, 3)]:
        ax.vlines(x[mask], 0, p[mask], color=colour, linewidth=1, zorder=z)
        ax.scatter(x[mask], p[mask], s=8, color=colour, zorder=z + 1, edgecolors='none')
    ax.axhline(EXT_CUTOFF, color=ORANGE, linewidth=1)
    ax.annotate(f'Cut-off ({EXT_CUTOFF:.3f})', (1, EXT_CUTOFF), xycoords=('axes fraction', 'data'), xytext=(5, 0),
                textcoords='offset points', color=ORANGE, va='center', ha='left', annotation_clip=False)
    ax.set_ylim(0, 1.2 * p.max())
    ax.set_ylabel('Predicted probability')

xh, phv = prob['human'].index.to_numpy(), prob['human'].to_numpy()
stems(ax_h, xh, phv, np.isin(xh, top))
ax_h.annotate('/'.join(f'{seq["human"][x - 1]}{x}' for x in top)
              + f' ({" and ".join(ordinal(rank_h[x]) for x in top)} of {len(xh)})',
              (np.mean(top), prob['human'][top].max()), xytext=(0, 3), textcoords='offset points',
              ha='center', va='bottom', color=BLUE)
for x in next_ranked:
    ax_h.annotate(f'{seq["human"][x - 1]}{x} ({ordinal(rank_h[x])})', (x, prob['human'][x]), xytext=(0, 3),
                  textcoords='offset points', ha='right', va='bottom', color=INK)
ax_h.set_title(f'(B) Human LDB3: {len(xh)} serines and threonines, no site in O-GlcNAcAtlas 5.0', loc='left')

aligned = [(to_human[y], p, y) for y, p in prob['mouse'].items() if y in to_human]
xm = np.array([a for a, _, _ in aligned]); pmv = np.array([p for _, p, _ in aligned])
stems(ax_m, xm, pmv, np.isin(xm, top))
known = [(to_human[y], prob['mouse'][y]) for y in sorted(known_mouse) if y in to_human]
ax_m.scatter([a for a, _ in known], [p for _, p in known], marker='D', s=18, color=INK, zorder=6,
             label='In O-GlcNAcAtlas 5.0 (mouse)')
ax_m.legend(frameon=False, loc='lower right', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.1)
ax_m.set_title(f'(C) Mouse Ldb3 (Q9JKS4) aligned to human (identity '
               f'{sum(a == b for a, b in zip(gh, gm) if a != "-" and b != "-") / sum(a != "-" and b != "-" for a, b in zip(gh, gm)):.1%})',
               loc='left')
ax_m.set_xlabel('Position in human LDB3 (O75112)')
ax_m.set_xlim(0, 1.01 * L)
plt.setp(ax_h.get_xticklabels(), visible=False)

# (D) sequences around the predicted sites
cols = [k for k, x in enumerate(ph) if x is not None and ZOOM[0] <= x <= ZOOM[1]]
for i, k in enumerate(cols):
    for row, (g, pos, name) in enumerate([(gh, ph, 'human'), (gm, pm, 'mouse')]):
        x = pos[k]
        colour, weight = INK, 'normal'
        if x is not None and (x if name == 'human' else to_human.get(x)) in top:
            colour, weight = BLUE, 'bold'
        if name == 'mouse' and x in known_mouse:
            colour, weight = 'black', 'bold'
        ax_seq.text(i, 1 - row, g[k], ha='center', va='center', family='monospace', color=colour, weight=weight)
        if name == 'mouse' and x in known_mouse:
            ax_seq.scatter(i, -0.45, marker='D', s=14, color=INK)
    if gh[k] == gm[k]:
        ax_seq.text(i, 0.5, '|', ha='center', va='center', family='monospace', color=GREY)
first = lambda pos: next(pos[k] for k in cols if pos[k] is not None)
ax_seq.text(-1, 1, f'human {first(ph)}', ha='right', va='center')
ax_seq.text(-1, 0, f'mouse {first(pm)}', ha='right', va='center')
ax_seq.set_xlim(-8, len(cols))
ax_seq.set_ylim(-0.7, 1.5)
ax_seq.axis('off')
ax_seq.set_title(f'(D) Human {ZOOM[0]}-{ZOOM[1]} and aligned mouse sequence', loc='left')

save_figure(fig, 'figure_case_LDB3', 180)
plt.show()
print('Highest ranked human S/T: ' + ', '.join(f'{seq["human"][x - 1]}{x} (rank {rank_h[x]}, {prob["human"][x]:.3f})'
                                              for x in rank_h[rank_h <= TOP_N + 1].sort_values().index))
print('Mouse O-GlcNAcAtlas 5.0 positives (canonical numbering -> human): '
      + ', '.join(f'{seq["mouse"][y - 1]}{y}->{to_human.get(y, "-")}' for y in sorted(known_mouse)))

# %%
# Rank of every S/T of human LDB3 (O75112) by calibrated probability
pred = pd.read_parquet(f'{OUT_DIR}/human_proteome_predictions.parquet')
ldb3 = pred[pred['accession'] == 'O75112'].copy()
ldb3['rank'] = ldb3['probability'].rank(ascending=False, method='min').astype(int)
print(f'Human LDB3: {len(ldb3)} S/T')
print(ldb3.nsmallest(5, 'rank')[['position', 'residue', 'probability', 'rank']].to_string(index=False, float_format='{:.3f}'.format))

# %%
# ---------------------------------------------------------------------------
# LPP (human Q93052): sites reported by Gong et al. 2026 (BBA Mol Cell Res 1873:120117; IP-MS and
# mutagenesis; main sites S33/S35) and our new candidate S155
#   Check: the reported peptide METTHSFGNPSISVSTQQPPK lies at 23-43 of the current sequence.
#   Per site: residue, calibrated probability, rank among the protein's S/T, Atlas 5.0 status
#   (transferred to the current sequence) and our split.
# ---------------------------------------------------------------------------
LPP = 'Q93052'
REPORTED = {12: 'IP-MS candidate', 25: 'IP-MS candidate', 28: 'IP-MS candidate',
            33: 'main site (IP-MS + S33A/S35A)', 35: 'main site (IP-MS + S33A/S35A)', 155: 'our new candidate'}
PEPTIDE, PEPTIDE_START = 'METTHSFGNPSISVSTQQPPK', 23

lpp_seq = proteomes['human'].set_index('accession').at[LPP, 'sequence']
assert lpp_seq[PEPTIDE_START - 1:PEPTIDE_START - 1 + len(PEPTIDE)] == PEPTIDE, 'peptide not at the reported position'
lpp = predictions('human')[predictions('human')['accession'] == LPP].copy()
lpp['rank'] = lpp['probability'].rank(ascending=False, method='min').astype(int)
lpp_status = known_sites('human', LPP, lpp_seq)
split = proteomes['human'].set_index('accession').at[LPP, 'our_split']
print(f'LPP {LPP}: {len(lpp_seq)} aa, {len(lpp)} S/T, our split: {split}')

rows = [{'site': f'{lpp_seq[x - 1]}{x}', 'source': src,
         'probability': lpp.set_index('position')['probability'].get(x),
         'rank': lpp.set_index('position')['rank'].get(x), 'atlas_5.0': lpp_status.get(x, '-')}
        for x, src in REPORTED.items()]
print(pd.DataFrame(rows).to_string(index=False, float_format='{:.3f}'.format))
print('\nTop 10 S/T of LPP:')
top = lpp.nsmallest(10, 'rank')[['position', 'probability', 'rank']]
top.insert(1, 'residue', top['position'].map(lambda x: lpp_seq[x - 1]))
top['atlas_5.0'] = top['position'].map(lambda x: lpp_status.get(x, '-'))
print(top.to_string(index=False, float_format='{:.3f}'.format))

# %%
# ---------------------------------------------------------------------------
# Candidate CIC (capicua, human Q96RK0): new high-probability sites vs every known source
#   Sources, all transferred to human Q96RK0 positions (alignment as above, identical residues):
#     Atlas 5.0 human (Q96RK0) and mouse (Q924A2) entries: positive / masked;
#     the 2025-26 external studies (NB01 external section): every reported protein whose sequence
#     contains MOTIF (a stretch of the candidate region identical in human and mouse), with its
#     localized and unlocalized sites.
#   Table: S/T of human CIC with probability >= SHOW_P or reported by any source, with rank in the
#   protein; then the top 15 S/T.
# ---------------------------------------------------------------------------
CIC_H, CIC_M = 'Q96RK0', 'Q924A2'
MOTIF = 'TNVVRPVSSTPVPIASKPF'
SHOW_P = 0.160

h_seq = proteomes['human'].set_index('accession').at[CIC_H, 'sequence']
m_seq = proteomes['mouse'].set_index('accession').at[CIC_M, 'sequence']
cic = predictions('human')[predictions('human')['accession'] == CIC_H].set_index('position')['probability']
rank = cic.rank(ascending=False, method='min').astype(int)
print(f'Human CIC {CIC_H}: {len(h_seq)} aa, {len(cic)} S/T; our split: '
      f'{proteomes["human"].set_index("accession").at[CIC_H, "our_split"]}')

sources = {}   # human position -> list of source labels
def add(pos, label):
    if pos is not None:
        sources.setdefault(pos, []).append(label)

for x, s in known_sites('human', CIC_H, h_seq).items():
    add(x, f'Atlas human ({s.split()[-1]})')
m_to_h = position_map(m_seq, h_seq)
for y, s in known_sites('mouse', CIC_M, m_seq).items():
    add(m_to_h.get(y), f'Atlas mouse ({s.split()[-1]})')

ext_sites = pd.read_parquet('data_processed/external_sites_mapped.parquet')
ext_proteins = pd.read_parquet('data_processed/external_proteins.parquet').set_index('seq_id')
matched = ext_proteins[ext_proteins['sequence'].str.contains(MOTIF, regex=False)]
print(f'External-study proteins containing {MOTIF}: '
      + '; '.join(f'{i} {list(r.accessions)} ({len(r.sequence)} aa)' for i, r in matched.iterrows()))
for seq_id, r in matched.iterrows():
    to_h = {x: x for x in range(1, len(h_seq) + 1)} if r.sequence == h_seq else position_map(r.sequence, h_seq)
    for s in ext_sites[ext_sites['seq_id'] == seq_id].itertuples():
        if pd.notna(s.position):
            positions = [int(s.position)]
        else:   # NOTISE region "(1 of a,b,c)": every candidate position, as in NB12
            positions = [int(p) for p in re.findall(r'\d+', s.detail.split('of', 1)[1])]
        for p in positions:
            add(to_h.get(p), f'{s.study} ({"localized" if s.localized else "unlocalized"})')

rows = [{'site': f'{h_seq[x - 1]}{x}', 'probability': p, 'rank': rank[x],
         'sources': '; '.join(sorted(set(sources.get(x, [])))) or '-'}
        for x, p in cic.items() if p >= SHOW_P or x in sources]
table = pd.DataFrame(rows)
with pd.option_context('display.max_colwidth', 120, 'display.width', 250):
    print(table.to_string(index=False, float_format='{:.3f}'.format))
    print('\nTop 15 S/T of human CIC:')
    print(table.sort_values('rank').head(15).to_string(index=False, float_format='{:.3f}'.format))
