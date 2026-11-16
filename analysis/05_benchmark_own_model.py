# %%
import os
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Final model (locked by validation AUROC): ESM2-150M, target residue concatenated with
# the mean of the ±20 window, one hidden layer of 256 units, dropout 0.25, lr 1e-4.
#   Weights and standardization (training mean and SD) come from the training grid.
#   Check: the saved weights reproduce the validation AUROC recorded during training.
#   The test set is not touched here.
# ---------------------------------------------------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'   # the final model is small; CPU is enough
PLM, WINDOW = 'ESM2-150M', 'pm20'
FINAL = {'hidden': 256, 'dropout': 0.25, 'lr': 1e-4}
FEAT_DIR = 'data_processed/features'
RUN_DIR = f'data_processed/training/grid/{PLM}/{WINDOW}'
OUT_DIR = 'data_processed/final_model'
os.makedirs(OUT_DIR, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False   # float32 throughout, as in training

saved = torch.load(f'{RUN_DIR}/models.pt', map_location='cpu')
k = saved['configs'].index(FINAL)
results = pd.read_csv(f'{RUN_DIR}/results.csv')
recorded = results.loc[(results[list(FINAL)] == pd.Series(FINAL)).all(axis=1)]
assert len(recorded) == 1

mean, std = saved['mean'].to(DEVICE), saved['std'].to(DEVICE)
model = nn.Sequential(nn.Linear(len(mean), FINAL['hidden']), nn.ReLU(), nn.Dropout(FINAL['dropout']),
                      nn.Linear(FINAL['hidden'], 1))
model.load_state_dict(saved['states'][k])
model = model.to(DEVICE).eval()

sites = pd.read_parquet('data_processed/embeddings/sites.parquet')
meta = json.load(open(f'{FEAT_DIR}/{PLM}/features.json'))
assert meta['shape'][0] == len(sites)
target = np.load(f'{FEAT_DIR}/{PLM}/target.npy', mmap_mode='r')
window_mean = np.load(f'{FEAT_DIR}/{PLM}/mean_{WINDOW}.npy', mmap_mode='r')

def logits(mask):
    """Model logits for the selected sites (standardized with the training mean and SD)."""
    X = torch.from_numpy(np.concatenate([np.asarray(target[mask]), np.asarray(window_mean[mask])], axis=1)).to(DEVICE)
    X = (X - mean) / std
    with torch.no_grad():
        return torch.cat([model(X[i:i + 65536]).squeeze(1) for i in range(0, len(X), 65536)]).cpu().numpy()

# Validation sites (masked residues excluded), as in training
is_val = ((sites['split'] == 'val') & (sites['label'] != 'masked')).to_numpy()
val = sites.loc[is_val, ['protein_id', 'position', 'label']].reset_index(drop=True)
val['y'] = (val['label'] == 'positive').astype(int)
val['logit'] = logits(is_val)

auroc = roc_auc_score(val['y'], val['logit'])
print(f'Validation sites: {len(val):,} ({val["y"].sum():,} positive)')
print(f'Validation AUROC: recorded in training {recorded["best_val_auroc"].iloc[0]:.6f}   '
      f'recomputed {auroc:.6f}   (best epoch {recorded["best_epoch"].iloc[0]})')
assert abs(auroc - recorded['best_val_auroc'].iloc[0]) < 1e-6

val.to_parquet(f'{OUT_DIR}/validation_scores.parquet', index=False)
torch.save({'plm': PLM, 'window': WINDOW, 'config': FINAL, 'state': saved['states'][k],
            'mean': saved['mean'], 'std': saved['std']}, f'{OUT_DIR}/final_model.pt')

# %%
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

# ---------------------------------------------------------------------------
# Platt scaling: calibrated probability = sigmoid(a * logit + b)
#   a and b are fitted on the validation sites at their natural imbalance, by logistic
#   regression of the label on the model logit (no penalty, 0/1 targets). The model was
#   trained on balanced batches, so its raw sigmoid output assumes ~50% positives.
#   The validation set is also the early-stopping and model-selection set (stated in
#   Methods); calibration is checked on the test set.
# ---------------------------------------------------------------------------
platt = LogisticRegression(penalty=None, max_iter=1000, tol=1e-10)
platt.fit(val[['logit']], val['y'])
a, b = float(platt.coef_[0, 0]), float(platt.intercept_[0])

def calibrate(logit):
    z = a * np.asarray(logit, dtype=np.float64) + b   # float64: float32 rounding creates ties
    return 1 / (1 + np.exp(-z))

val['raw_probability'] = 1 / (1 + np.exp(-val['logit'].astype(np.float64)))
val['probability'] = calibrate(val['logit'])

val['raw_probability'] = 1 / (1 + np.exp(-val['logit']))
val['probability'] = calibrate(val['logit'])

print(f'Platt parameters: a = {a:.4f}   b = {b:.4f}   (scikit-learn {sklearn.__version__})')
print(f'Validation positive rate: {val["y"].mean():.4f}   mean raw probability: {val["raw_probability"].mean():.4f}   '
      f'mean calibrated probability: {val["probability"].mean():.4f}')
print('Fit quality on validation (same data as the fit, so not a test of calibration):')
for name, p in [('raw', val['raw_probability']), ('calibrated', val['probability'])]:
    print(f'  {name:>10}: Brier {brier_score_loss(val["y"], p):.5f}   log loss {log_loss(val["y"], p):.5f}')
assert abs(roc_auc_score(val['y'], val['probability']) - auroc) < 1e-9   # monotone map: ranking unchanged

val.to_parquet(f'{OUT_DIR}/validation_scores.parquet', index=False)
final = torch.load(f'{OUT_DIR}/final_model.pt', map_location='cpu')
final['platt'] = {'a': a, 'b': b, 'fitted_on': 'validation sites (masked excluded)', 'sklearn': sklearn.__version__}
torch.save(final, f'{OUT_DIR}/final_model.pt')

# %%
from sklearn.metrics import matthews_corrcoef

# ---------------------------------------------------------------------------
# Cut-off for yes/no calls (MCC): the calibrated probability that maximises MCC on the
#   validation sites; it is applied unchanged to the test set. A site is called positive
#   when its calibrated probability is >= the cut-off.
# ---------------------------------------------------------------------------
def mcc_by_cutoff(y, score):
    """MCC at every distinct score used as the cut-off (score >= cut-off is positive)."""
    order = np.argsort(-score, kind='stable')
    s, yy = score[order], y[order].astype(np.float64)
    tp, fp = np.cumsum(yy), np.cumsum(1 - yy)
    last = np.r_[s[1:] != s[:-1], True]          # last site of each run of tied scores
    tp, fp, cutoffs = tp[last], fp[last], s[last]
    fn, tn = yy.sum() - tp, (len(yy) - yy.sum()) - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros_like(den), where=den > 0)
    return cutoffs, mcc, tp, fp

y_val, p_val = val['y'].to_numpy(), val['probability'].to_numpy()
cutoffs, mcc, tp, fp = mcc_by_cutoff(y_val, p_val)
best = int(np.argmax(mcc))
CUTOFF = float(cutoffs[best])
assert abs(matthews_corrcoef(y_val, p_val >= CUTOFF) - mcc[best]) < 1e-9

logit_cutoff = (np.log(CUTOFF / (1 - CUTOFF)) - b) / a
print(f'MCC-maximising cut-off on validation: calibrated probability {CUTOFF:.4f}   (model logit {logit_cutoff:.3f})')
print(f'Validation at this cut-off: MCC {mcc[best]:.4f}   precision {tp[best] / (tp[best] + fp[best]):.4f}   '
      f'recall {tp[best] / y_val.sum():.4f}   sites called positive {int(tp[best] + fp[best]):,} of {len(y_val):,}')

final = torch.load(f'{OUT_DIR}/final_model.pt', map_location='cpu')
final['cutoff'] = {'probability': CUTOFF, 'rule': 'calibrated probability >= cutoff',
                   'chosen_by': 'maximum MCC on validation sites'}
torch.save(final, f'{OUT_DIR}/final_model.pt')

# %%
from sklearn.metrics import average_precision_score, precision_recall_curve

# ---------------------------------------------------------------------------
# Test set, scored once with the frozen final model (weights, Platt parameters and MCC
# cut-off all fixed on training/validation data). Masked residues excluded.
#   AUROC (primary), AUPRC, MCC/precision/recall at the validation cut-off, precision at
#   fixed recall (best precision reached at recall >= level, from the test PR curve),
#   Brier score, and mean calibrated probability vs observed positive rate.
# ---------------------------------------------------------------------------
RECALL_LEVELS = [0.10, 0.25, 0.50, 0.75]

is_test = ((sites['split'] == 'test') & (sites['label'] != 'masked')).to_numpy()
test = sites.loc[is_test, ['protein_id', 'position', 'label']].reset_index(drop=True)
test['y'] = (test['label'] == 'positive').astype(int)
test['logit'] = logits(is_test)
test['probability'] = calibrate(test['logit'])
test.to_parquet(f'{OUT_DIR}/test_scores.parquet', index=False)

y_test, p_test = test['y'].to_numpy(), test['probability'].to_numpy()
called = p_test >= CUTOFF
tp = int((called & (y_test == 1)).sum())
precision, recall, _ = precision_recall_curve(y_test, p_test)
at_recall = {r: precision[recall >= r].max() for r in RECALL_LEVELS}

metrics = {'sites': len(test), 'positive': int(y_test.sum()), 'positive rate': y_test.mean(),
           'AUROC': roc_auc_score(y_test, p_test), 'AUPRC': average_precision_score(y_test, p_test),
           'MCC': matthews_corrcoef(y_test, called), 'precision at cut-off': tp / called.sum(),
           'recall at cut-off': tp / y_test.sum(), 'sites called positive': int(called.sum()),
           **{f'precision at recall {r:.2f}': v for r, v in at_recall.items()},
           'Brier': brier_score_loss(y_test, p_test), 'mean calibrated probability': p_test.mean()}
pd.Series(metrics).to_csv(f'{OUT_DIR}/test_metrics.csv', header=['value'])

print(f'Test sites: {len(test):,} ({y_test.sum():,} positive, rate {y_test.mean():.4f}) '
      f'in {test["protein_id"].nunique()} proteins')
print(f'AUROC {metrics["AUROC"]:.4f}   AUPRC {metrics["AUPRC"]:.4f}')
print(f'At the validation cut-off (calibrated probability >= {CUTOFF:.4f}): MCC {metrics["MCC"]:.4f}   '
      f'precision {metrics["precision at cut-off"]:.4f}   recall {metrics["recall at cut-off"]:.4f}   '
      f'sites called positive {called.sum():,}')
print('Precision at fixed recall: ' + '   '.join(f'recall {r:.2f}: {v:.4f}' for r, v in at_recall.items()))
print(f'Calibration: Brier {metrics["Brier"]:.5f}   mean calibrated probability {p_test.mean():.4f} '
      f'vs observed positive rate {y_test.mean():.4f}')

# %%
import io
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from sklearn.metrics import roc_curve
from statsmodels.stats.proportion import proportion_confint

# ---------------------------------------------------------------------------
# Test-set performance of the final model (figure number set with the manuscript)
#   (A) ROC curve and (B) precision-recall curve on the test sites; the diamond marks the
#   validation MCC cut-off. Grey line: a classifier with no skill (diagonal in A; the test
#   positive rate in B).
#   (C) Reliability diagram: test sites in 10 equal-count bins of calibrated probability;
#   mean calibrated probability vs observed fraction of positives, with 95% Wilson
#   intervals, on log scales (most sites have low probabilities). Grey line: perfect
#   calibration.
#   Frontiers figure rules: 180 mm wide, text >= 8 pt and lines >= 2 pt at final size,
#   solid lines only, panels labelled (A)-(C) outside the plot area, no red or green;
#   PDF and RGB TIFF at 300 dpi.
# ---------------------------------------------------------------------------
FIG_DIR = 'results/figures/final_model'
os.makedirs(FIG_DIR, exist_ok=True)
BLUE, GREY, BLACK = '#1c5cab', '#8a8984', '#0b0b0b'
N_BINS = 10

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

# Operating point of the validation cut-off on the test set
fpr_cut = (called & (y_test == 0)).sum() / (y_test == 0).sum()
rec_cut, prec_cut = metrics['recall at cut-off'], metrics['precision at cut-off']

# Reliability bins: equal numbers of test sites, by calibrated probability
order = np.argsort(p_test, kind='stable')
bins = []
for idx in np.array_split(order, N_BINS):
    k, n = int(y_test[idx].sum()), len(idx)
    low, high = proportion_confint(k, n, alpha=0.05, method='wilson')
    bins.append({'sites': n, 'positive': k, 'min_probability': p_test[idx].min(), 'max_probability': p_test[idx].max(),
                 'mean_probability': p_test[idx].mean(), 'observed_fraction': k / n, 'ci_low': low, 'ci_high': high})
bins = pd.DataFrame(bins)
bins.to_csv(f'{FIG_DIR}/reliability_bins_test.csv', index=False)

fig, axes = plt.subplots(1, 3, figsize=(180 / 25.4, 66 / 25.4), layout='constrained')

ax = axes[0]
fpr, tpr, _ = roc_curve(y_test, p_test)
ax.plot([0, 1], [0, 1], color=GREY)
ax.plot(fpr, tpr, color=BLUE)
ax.plot(fpr_cut, rec_cut, marker='D', color=BLACK, markersize=6, linestyle='')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate (recall)')
ax.text(0.97, 0.05, f'AUROC {metrics["AUROC"]:.3f}', ha='right', va='bottom')

ax = axes[1]
ax.axhline(y_test.mean(), color=GREY)
ax.plot(recall, precision, color=BLUE)
ax.plot(rec_cut, prec_cut, marker='D', color=BLACK, markersize=6, linestyle='')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_ylim(0, 1)
ax.text(0.97, 0.95, f'AUPRC {metrics["AUPRC"]:.3f}', ha='right', va='top')

ax = axes[2]
low_lim = 0.5 * min(bins['ci_low'].min(), bins['mean_probability'].min())
high_lim = 2 * max(bins['ci_high'].max(), bins['mean_probability'].max())
ax.plot([low_lim, high_lim], [low_lim, high_lim], color=GREY)
ax.errorbar(bins['mean_probability'], bins['observed_fraction'],
            yerr=[bins['observed_fraction'] - bins['ci_low'], bins['ci_high'] - bins['observed_fraction']],
            color=BLUE, marker='o', markersize=5, elinewidth=2, capsize=0)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(low_lim, high_lim)
ax.set_ylim(low_lim, high_lim)
ax.set_xlabel('Mean calibrated probability\n(log scale)')
ax.set_ylabel('Observed fraction positive\n(log scale)')
ax.text(0.97, 0.05, f'Brier {metrics["Brier"]:.4f}', ha='right', va='bottom', transform=ax.transAxes)

for k, ax in enumerate(axes):
    ax.set_title(f'({chr(65 + k)})', loc='left', fontweight='bold')
    ax.set_aspect('equal')
    if k < 2:
        ax.set_xlim(0, 1)

fig.legend([Line2D([], [], color=BLUE), Line2D([], [], color=GREY),
            Line2D([], [], marker='D', color=BLACK, linestyle='', markersize=6)],
           ['Final model, test set', 'No skill (A, B); perfect calibration (C)', 'Validation MCC cut-off'],
           loc='outside lower center', ncol=3, frameon=False)

save_figure(fig, 'figure_test_performance', 180)
plt.show()
print(bins.to_string(index=False, float_format='{:.4f}'.format))

# %%
import subprocess, time
from datetime import date
from transformers import AutoTokenizer, EsmModel
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# External validation: every S/T of every external protein, scored with the frozen final model
#   Proteins: data_processed/external_proteins.parquet (NB01, external section). No labels are
#   used here; sites, labels and filters are applied in the final comparison notebook.
#   The features are built exactly as for the training data (NB02, cells 0, 1 and 4; their
#   functions are repeated below unchanged):
#     - proteins of up to 1,022 residues are embedded whole; longer proteins are cut into the
#       fewest windows of 1,022 residues overlapping by >= 256, spaced evenly, and each S/T
#       takes its embedding and neighbours from the window where it is most central;
#     - ESM2-150M last layer, float32 (TF32 off), windows batched by length under a token budget;
#     - classifier input: the target embedding concatenated with the mean of the +-20 window
#       (target included; truncated at protein ends), means from float64 cumulative sums.
#   Then the final model: training mean/SD standardization, MLP logit, Platt calibration, and
#   the validation MCC cut-off (all read from final_model.pt).
#   Check first: five test proteins (four random, plus the longest) go through the same code,
#   and their features and logits are compared with the stored ones.
#   Timing (compute statement): wall-clock time from sequences to calibrated probabilities
#   (windows, ESM2-150M embeddings, features, final model; model loading excluded) for all test
#   proteins and for the external proteins; each saved as results/compute/NB05_{step}.json.
#   The test proteins scored here are also compared with the stored test logits (cell 3), and
#   the external scores with those of an earlier run, if any, before they are replaced.
# ---------------------------------------------------------------------------
MAX_RES, OVERLAP, K = 1022, 256, 20
ESM_PATH, MAX_TOKENS = 'facebook/esm2_t30_150M_UR50D', 65536   # as in NB02 for ESM2-150M
LOCAL_FILE = '/content/external_residues.npy'
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

# Check: five test proteins through this code vs the stored features (NB02) and test logits (NB05)
atlas = pd.read_parquet('data_processed/atlas5_proteins.parquet')
test_prot = atlas[atlas['split'] == 'test']
check_ids = list(test_prot.sample(4, random_state=0)['protein_id']) + [test_prot.loc[test_prot['length'].idxmax(), 'protein_id']]
check_seqs = [test_prot.set_index('protein_id').at[p, 'sequence'] for p in check_ids]
check, check_feats = score_proteins(check_ids, check_seqs, keep_features=True)

all_sites = pd.read_parquet('data_processed/embeddings/sites.parquet')
stored_rows = (check[['protein_id', 'position']]   # rows of sites.parquet, in the order of `check`
               .merge(all_sites.reset_index()[['index', 'protein_id', 'position']], on=['protein_id', 'position'], how='left')
               ['index'].to_numpy())
assert len(stored_rows) == len(check)
stored = np.concatenate([np.load(f'{FEAT_DIR}/{PLM}/target.npy', mmap_mode='r')[stored_rows],
                         np.load(f'{FEAT_DIR}/{PLM}/mean_pm20.npy', mmap_mode='r')[stored_rows]], axis=1)
test_logits = pd.read_parquet(f'{OUT_DIR}/test_scores.parquet')
cmp = check.merge(test_logits[['protein_id', 'position', 'logit']], on=['protein_id', 'position'], suffixes=('', '_test'))
print(f'Check on {len(check_ids)} test proteins (lengths {", ".join(str(len(s)) for s in check_seqs)}), {len(check):,} S/T: '
      f'features max |diff| {np.abs(check_feats - stored).max():.1e}   '
      f'logits max |diff| {np.abs(cmp["logit"] - cmp["logit_test"]).max():.1e} ({len(cmp):,} unmasked sites)')
assert np.abs(check_feats - stored).max() < 1e-3 and np.abs(cmp['logit'] - cmp['logit_test']).max() < 1e-3

# All test proteins, timed; logits compared with the stored test logits
t0 = time.perf_counter()
test_all, _ = score_proteins(test_prot['protein_id'].tolist(), test_prot['sequence'].tolist())
seconds = time.perf_counter() - t0
cmp_all = test_all.merge(test_logits[['protein_id', 'position', 'logit']], on=['protein_id', 'position'],
                         suffixes=('', '_test'))
assert len(cmp_all) == len(test_logits)
print(f'All {len(test_prot)} test proteins, {len(test_all):,} S/T: logits max |diff| vs stored '
      f'{np.abs(cmp_all["logit"] - cmp_all["logit_test"]).max():.1e} ({len(cmp_all):,} unmasked sites)')
assert np.abs(cmp_all['logit'] - cmp_all['logit_test']).max() < 1e-3
save_compute('NB05', 'test_proteins', seconds, proteins=len(test_prot), st_sites=len(test_all))

# External proteins, timed
ext_proteins = pd.read_parquet('data_processed/external_proteins.parquet')
t0 = time.perf_counter()
ext_scores, _ = score_proteins(ext_proteins['seq_id'].tolist(), ext_proteins['sequence'].tolist())
seconds = time.perf_counter() - t0
ext_scores = ext_scores.rename(columns={'protein_id': 'seq_id'})
assert len(ext_scores) == ext_proteins['sequence'].str.count('[ST]').sum()
save_compute('NB05', 'external_proteins', seconds, proteins=len(ext_proteins), st_sites=len(ext_scores))
if os.path.exists(f'{OUT_DIR}/external_scores.parquet'):
    previous = pd.read_parquet(f'{OUT_DIR}/external_scores.parquet')
    assert previous[['seq_id', 'position']].equals(ext_scores[['seq_id', 'position']])
    print(f'External scores, previous run vs this run: logit max |diff| '
          f'{np.abs(previous["logit"] - ext_scores["logit"]).max():.1e}   '
          f'calls changed {int((previous["called"] != ext_scores["called"]).sum())}')
ext_scores.to_parquet(f'{OUT_DIR}/external_scores.parquet', index=False)
print(f'External proteins: {len(ext_proteins):,}   S/T scored: {len(ext_scores):,}   '
      f'called positive at the validation cut-off: {ext_scores["called"].sum():,} ({ext_scores["called"].mean():.2%})')

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
# Test-set performance per species
#   Species: the organism of each test protein (for sequences reported under several species and
#   merged in NB01, the organism of the kept ID). Per species: AUROC, AUPRC, and MCC, precision and
#   recall at the validation cut-off, with the number of training proteins and positive sites of the
#   same species. Species with fewer than MIN_POSITIVES positive test sites are listed but not drawn.
#   Outputs: test_metrics_by_species.csv and a figure.
# ---------------------------------------------------------------------------
OUT_DIR = 'data_processed/final_model'
FIG_DIR = 'results/figures/final_model'
os.makedirs(FIG_DIR, exist_ok=True)
MIN_POSITIVES = 20

test = pd.read_parquet(f'{OUT_DIR}/test_scores.parquet')
CUTOFF = torch.load(f'{OUT_DIR}/final_model.pt', map_location='cpu')['cutoff']['probability']
proteins = pd.read_parquet('data_processed/atlas5_proteins.parquet')
organism_of = proteins.set_index('protein_id')['organism']
test['organism'] = test['protein_id'].map(organism_of)
assert test['organism'].notna().all()
merged = proteins[(proteins['split'] == 'test') & proteins['organisms'].str.contains(';')]
print(f'Test proteins whose sequence is reported under several species: {len(merged)} '
      f'(species = organism of the kept ID)')

train = proteins[proteins['split'] == 'train']
train_counts = (train.assign(n_pos=train['positive'].str.len()).groupby('organism')
                .agg(**{'training proteins': ('protein_id', 'size'), 'training positive sites': ('n_pos', 'sum')}))

rows = []
for organism, d in test.groupby('organism'):
    y, p = d['y'].to_numpy(), d['probability'].to_numpy()
    called = p >= CUTOFF
    tp = int((called & (y == 1)).sum())
    both = 0 < y.sum() < len(y)
    rows.append({'organism': organism, 'proteins': d['protein_id'].nunique(), 'sites': len(d), 'positive': int(y.sum()),
                 'positive rate': y.mean(), 'AUROC': roc_auc_score(y, p) if both else np.nan,
                 'AUPRC': average_precision_score(y, p) if both else np.nan,
                 'MCC': matthews_corrcoef(y, called) if both else np.nan,
                 'precision': tp / max(called.sum(), 1), 'recall': tp / max(y.sum(), 1),
                 'drawn': bool(y.sum() >= MIN_POSITIVES)})
by_species = (pd.DataFrame(rows).join(train_counts, on='organism').fillna({'training proteins': 0, 'training positive sites': 0})
              .astype({'training proteins': int, 'training positive sites': int})
              .sort_values('positive', ascending=False).reset_index(drop=True))
by_species.to_csv(f'{OUT_DIR}/test_metrics_by_species.csv', index=False)
overall = roc_auc_score(test['y'], test['probability'])
print(f'\nAll test sites: AUROC {overall:.4f}; species with >= {MIN_POSITIVES} positive test sites are drawn')
print(by_species.drop(columns='drawn').to_string(index=False, float_format='{:.3f}'.format))

# Figure: AUROC per species (>= MIN_POSITIVES positive test sites), most positive sites at the top
# (Frontiers: 180 mm wide, text >= 8 pt, no red or green; PDF and RGB TIFF at 300 dpi)
OUR_COLOUR, REF_COLOUR = '#2a78d6', '#8a8984'
d = by_species[by_species['drawn']].iloc[::-1]
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(180 / 25.4, (len(d) * 7 + 24) / 25.4))
fig.subplots_adjust(left=0.24, right=0.62, top=1 - 9 / (len(d) * 7 + 24), bottom=13 / (len(d) * 7 + 24))
ypos = np.arange(len(d))
ax.axvline(overall, color=REF_COLOUR, linewidth=0.8, linestyle=(0, (2, 2)))
ax.hlines(ypos, 0.5, d['AUROC'], color=OUR_COLOUR, linewidth=1.5)
ax.scatter(d['AUROC'], ypos, color=OUR_COLOUR, s=22, zorder=3)
for yp, (_, r) in zip(ypos, d.iterrows()):
    ax.annotate(f'{r["AUROC"]:.3f}', (r['AUROC'], yp), xytext=(5, 0), textcoords='offset points', va='center', fontsize=8)
    for x, text in [(1.30, f'{r["positive"]:,} ({r["proteins"]:,})'), (1.86, f'{r["training positive sites"]:,} ({r["training proteins"]:,})')]:
        ax.annotate(text, (x, yp), xycoords=('axes fraction', 'data'), ha='right', va='center', fontsize=8)
for x, text in [(1.30, 'Test positive sites\n(proteins)'), (1.86, 'Training positive sites\n(proteins)')]:
    ax.annotate(text, (x, 1), xycoords='axes fraction', xytext=(0, 4), textcoords='offset points',
                ha='right', va='bottom', fontsize=8, fontweight='bold')
ax.set_yticks(ypos, [f'$\\it{{{o.replace(" ", "~")}}}$' for o in d['organism']])   # species names in italics
ax.tick_params(axis='y', length=0)
ax.set_xlim(0.5, 1.0)
ax.set_xlabel(f'AUROC (0.5 = no skill; dashed: all test sites, {overall:.3f})')

fig.savefig(f'{FIG_DIR}/figure_test_by_species_AUROC.pdf')
buffer = io.BytesIO()
fig.savefig(buffer, format='png', dpi=300)
Image.open(buffer).convert('RGB').save(f'{FIG_DIR}/figure_test_by_species_AUROC.tif', dpi=(300, 300))
plt.show()
