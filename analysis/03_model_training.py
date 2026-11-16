# %%
import os
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))

import copy, json, time
from datetime import date

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Training data: one feature table per pLM and window option
#   Window options: the target residue alone ('residue'), or the target residue
#   concatenated with the mean of the window ±K ('pm05' ... 'pm35', target included).
#   Training uses train sites and model selection uses validation sites; masked sites are
#   excluded from both. Features are standardized with the mean and SD of the training sites.
# ---------------------------------------------------------------------------
DEVICE = 'cuda'
FEAT_DIR, TRAIN_DIR = 'data_processed/features', 'data_processed/training'
WINDOWS = ['residue', 'pm05', 'pm10', 'pm15', 'pm20', 'pm25', 'pm35']

sites = pd.read_parquet('data_processed/embeddings/sites.parquet')
is_train = ((sites['split'] == 'train') & (sites['label'] != 'masked')).to_numpy()
is_val = ((sites['split'] == 'val') & (sites['label'] != 'masked')).to_numpy()
y_all = (sites['label'] == 'positive').to_numpy(np.float32)
print(f'Train sites: {is_train.sum():,} ({int(y_all[is_train].sum()):,} positive)   '
      f'validation sites: {is_val.sum():,} ({int(y_all[is_val].sum()):,} positive)')

def load_features(plm, window):
    """Standardized train and validation features on the GPU, and the training mean and SD."""
    meta = json.load(open(f'{FEAT_DIR}/{plm}/features.json'))
    assert meta['shape'][0] == len(sites)
    parts = [np.load(f'{FEAT_DIR}/{plm}/target.npy', mmap_mode='r')]
    if window != 'residue':
        parts.append(np.load(f'{FEAT_DIR}/{plm}/mean_{window}.npy', mmap_mode='r'))
    def subset(mask):
        return torch.from_numpy(np.concatenate([np.asarray(p[mask]) for p in parts], axis=1)).to(DEVICE)
    X_train, X_val = subset(is_train), subset(is_val)
    std, mean = torch.std_mean(X_train.double(), dim=0)
    std[std == 0] = 1.0
    mean, std = mean.float(), std.float()
    return (X_train - mean) / std, (X_val - mean) / std, mean.cpu(), std.cpu()

# %%
# ---------------------------------------------------------------------------
# Hyperparameter grid, trained side by side
#   18 configurations: one hidden layer of 16, 64 or 256 units, dropout 0, 0.25 or 0.5,
#   learning rate 1e-4 or 1e-5. All configurations see the same batches.
#   Epoch: every training positive once plus the same number of training negatives drawn
#   without replacement, fresh each epoch (random undersampling redrawn every epoch);
#   shuffled, batches of 1,024. Adam, no weight decay; loss: binary cross-entropy on the
#   logit. Validation AUROC after every epoch; each configuration stops after 300 epochs
#   without improvement (at most 6,000 epochs) and keeps the weights of its best epoch.
#   A line is printed as each configuration stops; results are uploaded to Drive at the end.
# ---------------------------------------------------------------------------
SEED, BATCH, MAX_EPOCHS, PATIENCE = 0, 1024, 6000, 300
GRID_DIR = f'{TRAIN_DIR}/grid'
CONFIGS = [{'hidden': h, 'dropout': p, 'lr': lr} for h in (16, 64, 256) for p in (0.0, 0.25, 0.5) for lr in (1e-4, 1e-5)]

def make_model(n_in, hidden, dropout):
    """Logistic regression (hidden = 0) or a one-hidden-layer MLP; outputs a logit."""
    if hidden == 0:
        return nn.Linear(n_in, 1)
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

def describe(c):
    if c['hidden'] == 0:
        return f"logistic regression           lr {c['lr']:.0e}"
    return f"hidden {c['hidden']:3d}  dropout {c['dropout']:.2f}  lr {c['lr']:.0e}"

def val_auroc(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        scores = torch.cat([model(X_val[i:i + 65536]).squeeze(1) for i in range(0, len(X_val), 65536)])
    model.train()
    return roc_auc_score(y_val, scores.cpu().numpy())

def train_grid(plm, window, configs=CONFIGS, out_dir=None):
    """Train all configurations for one pLM and window option; save results, history and best weights."""
    out_dir = out_dir or f'{GRID_DIR}/{plm}/{window}'
    if os.path.exists(f'{out_dir}/done.json'):
        print(f'{out_dir}: already trained, skipped')
        return
    load_start = time.time()
    X_train, X_val, mean, std = load_features(plm, window)
    load_minutes = (time.time() - load_start) / 60
    y_train = torch.from_numpy(y_all[is_train]).to(DEVICE)
    y_val = y_all[is_val]
    pos = torch.nonzero(y_train == 1).squeeze(1)
    neg = torch.nonzero(y_train == 0).squeeze(1)

    torch.manual_seed(SEED)
    gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    runs = []
    for c in configs:
        model = make_model(X_train.shape[1], c['hidden'], c['dropout']).to(DEVICE)
        runs.append({'config': c, 'model': model, 'opt': torch.optim.Adam(model.parameters(), lr=c['lr'], fused=True),
                     'best_auroc': -1.0, 'best_epoch': 0, 'best_state': None, 'active': True})

    def report(r, epochs, reason):
        tqdm.write(f"{describe(r['config'])}   best val AUROC {r['best_auroc']:.4f} at epoch {r['best_epoch']:4d}"
                   f"   ({epochs} epochs, {reason})")

    history, epoch_log = [], []
    start = time.time()
    bar = tqdm(range(1, MAX_EPOCHS + 1), desc=f'{plm} {window}')
    for epoch in bar:
        active = [r for r in runs if r['active']]
        if not active:
            break
        t0 = time.time()
        fresh_neg = neg[torch.randperm(len(neg), device=DEVICE, generator=gen)[:len(pos)]]
        order = torch.cat([pos, fresh_neg])
        order = order[torch.randperm(len(order), device=DEVICE, generator=gen)]
        losses = torch.zeros(len(active), device=DEVICE)
        for b in range(0, len(order), BATCH):
            idx = order[b:b + BATCH]
            xb, yb = X_train[idx], y_train[idx]
            for j, r in enumerate(active):
                loss = F.binary_cross_entropy_with_logits(r['model'](xb).squeeze(1), yb)
                r['opt'].zero_grad(set_to_none=True)
                loss.backward()
                r['opt'].step()
                losses[j] += loss.detach() * len(idx)
        losses = (losses / len(order)).cpu().numpy()
        train_seconds = time.time() - t0
        for j, r in enumerate(active):
            auroc = val_auroc(r['model'], X_val, y_val)
            history.append({**r['config'], 'epoch': epoch, 'train_loss': float(losses[j]), 'val_auroc': auroc})
            if auroc > r['best_auroc']:
                r.update(best_auroc=auroc, best_epoch=epoch, best_state=copy.deepcopy(r['model'].state_dict()))
            elif epoch - r['best_epoch'] >= PATIENCE:
                r['active'] = False
                report(r, epoch, 'early stop')
        epoch_log.append({'epoch': epoch, 'active_configs': len(active), 'train_seconds': train_seconds,
                          'total_seconds': time.time() - t0})
        bar.set_postfix(active=sum(r['active'] for r in runs), best=f"{max(r['best_auroc'] for r in runs):.4f}")
    for r in runs:
        if r['active']:
            report(r, MAX_EPOCHS, 'epoch limit reached')
    minutes = (time.time() - start) / 60

    history = pd.DataFrame(history)
    results = pd.DataFrame([{**r['config'], 'best_epoch': r['best_epoch'], 'best_val_auroc': r['best_auroc'],
                             'epochs_run': int((history[['hidden', 'dropout', 'lr']] == pd.Series(r['config'])).all(axis=1).sum())}
                            for r in runs])
    os.makedirs(out_dir, exist_ok=True)
    history.to_csv(f'{out_dir}/history.csv', index=False)
    results.to_csv(f'{out_dir}/results.csv', index=False)
    pd.DataFrame(epoch_log).to_csv(f'{out_dir}/epoch_times.csv', index=False)
    torch.save({'plm': plm, 'window': window, 'mean': mean, 'std': std, 'configs': configs,
                'states': [r['best_state'] for r in runs]}, f'{out_dir}/models.pt')
    with open(f'{out_dir}/done.json', 'w') as f:
        json.dump({'plm': plm, 'window': window, 'seed': SEED, 'training_minutes': round(minutes, 1),
                   'loading_minutes': round(load_minutes, 1), 'epochs': len(epoch_log),
                   'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                   'sklearn': sklearn.__version__, 'date': str(date.today())}, f, indent=1)

    log = pd.DataFrame(epoch_log)
    print(f'\n{plm} {window}: {minutes:.1f} min training, {load_minutes:.1f} min loading, {len(log)} epochs   '
          f'seconds per epoch with all {len(configs)} configurations active: '
          f'{log.loc[log["active_configs"] == len(configs), "total_seconds"].mean():.1f}\n')

# %%
# ---------------------------------------------------------------------------
# Pilot runs (ESM2-8M, window ±15), which preceded the grid and set its learning rates and
# sampling scheme
#   Training function used by the pilot: as train_grid above, except for the sampling and
#   the stopping rule. Epoch: every training negative once (shuffled) plus as many positives
#   drawn with replacement (random oversampling to balance; each positive appears about 43
#   times per epoch). Each configuration stops after 10 epochs without improvement (at most
#   100 epochs). The pilot configurations also include logistic regression (hidden = 0).
#   Findings (recorded in the decisions file): learning rate 1e-3 peaked at the first epoch
#   and 1e-6 was still improving at epoch 100, so the grid uses 1e-4 and 1e-5; the grid
#   switched to undersampling redrawn every epoch.
# ---------------------------------------------------------------------------
PILOT_MAX_EPOCHS, PILOT_PATIENCE = 100, 10
PILOT_CONFIGS = ([{'hidden': 0, 'dropout': 0.0, 'lr': lr} for lr in (1e-3, 1e-4)] +
                 [{'hidden': h, 'dropout': p, 'lr': lr} for h in (16, 64, 256) for p in (0.0, 0.25, 0.5)
                  for lr in (1e-3, 1e-4)])

def train_grid_pilot(plm, window, configs, out_dir):
    """Pilot version of train_grid: random oversampling, at most PILOT_MAX_EPOCHS epochs,
    patience PILOT_PATIENCE epochs. Saves results, history and best weights to out_dir."""
    if os.path.exists(f'{out_dir}/done.json'):
        print(f'{out_dir}: already trained, skipped')
        return
    load_start = time.time()
    X_train, X_val, mean, std = load_features(plm, window)
    load_minutes = (time.time() - load_start) / 60
    y_train = torch.from_numpy(y_all[is_train]).to(DEVICE)
    y_val = y_all[is_val]
    pos = torch.nonzero(y_train == 1).squeeze(1)
    neg = torch.nonzero(y_train == 0).squeeze(1)

    torch.manual_seed(SEED)
    gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    runs = []
    for c in configs:
        model = make_model(X_train.shape[1], c['hidden'], c['dropout']).to(DEVICE)
        runs.append({'config': c, 'model': model, 'opt': torch.optim.Adam(model.parameters(), lr=c['lr'], fused=True),
                     'best_auroc': -1.0, 'best_epoch': 0, 'best_state': None, 'active': True})

    def report(r, epochs, reason):
        tqdm.write(f"{describe(r['config'])}   best val AUROC {r['best_auroc']:.4f} at epoch {r['best_epoch']:3d}"
                   f"   ({epochs} epochs, {reason})")

    history, epoch_log = [], []
    start = time.time()
    bar = tqdm(range(1, PILOT_MAX_EPOCHS + 1), desc=f'{plm} {window}')
    for epoch in bar:
        active = [r for r in runs if r['active']]
        if not active:
            break
        t0 = time.time()
        order = torch.cat([neg, pos[torch.randint(len(pos), (len(neg),), device=DEVICE, generator=gen)]])
        order = order[torch.randperm(len(order), device=DEVICE, generator=gen)]
        losses = torch.zeros(len(active), device=DEVICE)
        for b in range(0, len(order), BATCH):
            idx = order[b:b + BATCH]
            xb, yb = X_train[idx], y_train[idx]
            for j, r in enumerate(active):
                loss = F.binary_cross_entropy_with_logits(r['model'](xb).squeeze(1), yb)
                r['opt'].zero_grad(set_to_none=True)
                loss.backward()
                r['opt'].step()
                losses[j] += loss.detach() * len(idx)
        losses = (losses / len(order)).cpu().numpy()
        train_seconds = time.time() - t0
        for j, r in enumerate(active):
            auroc = val_auroc(r['model'], X_val, y_val)
            history.append({**r['config'], 'epoch': epoch, 'train_loss': float(losses[j]), 'val_auroc': auroc})
            if auroc > r['best_auroc']:
                r.update(best_auroc=auroc, best_epoch=epoch, best_state=copy.deepcopy(r['model'].state_dict()))
            elif epoch - r['best_epoch'] >= PILOT_PATIENCE:
                r['active'] = False
                report(r, epoch, 'early stop')
        epoch_log.append({'epoch': epoch, 'active_configs': len(active), 'train_seconds': train_seconds,
                          'total_seconds': time.time() - t0})
        bar.set_postfix(active=sum(r['active'] for r in runs), best=f"{max(r['best_auroc'] for r in runs):.4f}")
    for r in runs:
        if r['active']:
            report(r, PILOT_MAX_EPOCHS, 'epoch limit reached')
    minutes = (time.time() - start) / 60

    history = pd.DataFrame(history)
    results = pd.DataFrame([{**r['config'], 'best_epoch': r['best_epoch'], 'best_val_auroc': r['best_auroc'],
                             'epochs_run': int((history[['hidden', 'dropout', 'lr']] == pd.Series(r['config'])).all(axis=1).sum())}
                            for r in runs])
    os.makedirs(out_dir, exist_ok=True)
    history.to_csv(f'{out_dir}/history.csv', index=False)
    results.to_csv(f'{out_dir}/results.csv', index=False)
    pd.DataFrame(epoch_log).to_csv(f'{out_dir}/epoch_times.csv', index=False)
    torch.save({'plm': plm, 'window': window, 'mean': mean, 'std': std, 'configs': configs,
                'states': [r['best_state'] for r in runs]}, f'{out_dir}/models.pt')
    with open(f'{out_dir}/done.json', 'w') as f:
        json.dump({'plm': plm, 'window': window, 'seed': SEED, 'training_minutes': round(minutes, 1),
                   'loading_minutes': round(load_minutes, 1), 'epochs': len(epoch_log),
                   'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                   'sklearn': sklearn.__version__, 'date': str(date.today())}, f, indent=1)

    log = pd.DataFrame(epoch_log)
    print(f'\n{plm} {window}: {minutes:.1f} min training, {load_minutes:.1f} min loading, {len(log)} epochs   '
          f'seconds per epoch with all {len(configs)} configurations active: '
          f'{log.loc[log["active_configs"] == len(configs), "total_seconds"].mean():.1f}')

# %%
# Pilot run 1: learning rates 1e-3 and 1e-4
train_grid_pilot('ESM2-8M', 'pm15', PILOT_CONFIGS, f'{TRAIN_DIR}/ESM2-8M/pm15')

# %%
# ---------------------------------------------------------------------------
# Pilot run 2, learning-rate test on ESM2-8M, window ±15: the same 20 configurations with
# learning rates 1e-5 and 1e-6 (pilot run 1 used 1e-3 and 1e-4), then all four rates side by side
# ---------------------------------------------------------------------------
LR_TEST = ([{'hidden': 0, 'dropout': 0.0, 'lr': lr} for lr in (1e-5, 1e-6)] +
           [{'hidden': h, 'dropout': p, 'lr': lr} for h in (16, 64, 256) for p in (0.0, 0.25, 0.5) for lr in (1e-5, 1e-6)])
LR_TEST_DIR = f'{TRAIN_DIR}/lr_test/ESM2-8M/pm15'
train_grid_pilot('ESM2-8M', 'pm15', LR_TEST, LR_TEST_DIR)

both = pd.concat([pd.read_csv(f'{TRAIN_DIR}/ESM2-8M/pm15/results.csv'), pd.read_csv(f'{LR_TEST_DIR}/results.csv')])
print('\nBest validation AUROC (and its epoch) by configuration and learning rate:')
print(both.pivot_table(index=['hidden', 'dropout'], columns='lr', values='best_val_auroc').round(4).to_string())
print(both.pivot_table(index=['hidden', 'dropout'], columns='lr', values='best_epoch').to_string())

# %%
# Full grid: every pLM and window option (pLMs whose features are not on Drive yet are skipped)
for plm in ['ESM2-8M', 'ESM2-35M', 'ESM2-150M', 'ESM2-650M']:
    if not os.path.exists(f'{FEAT_DIR}/{plm}/features.json'):
        print(f'{plm}: features not available yet, skipped')
        continue
    for window in WINDOWS:
        train_grid(plm, window)

# %%
# ---------------------------------------------------------------------------
# Grid extension: hidden 512
#   The best configurations of the grid used the largest hidden size (256), so the grid is
#   extended with 512 units (dropout 0, 0.25, 0.5; learning rate 1e-4, 1e-5), with the same
#   features, sampling, seed and stopping rule. Saved separately and merged with the main
#   grid in notebook 04.
# ---------------------------------------------------------------------------
CONFIGS_H512 = [{'hidden': 512, 'dropout': p, 'lr': lr} for p in (0.0, 0.25, 0.5) for lr in (1e-4, 1e-5)]
for plm in ['ESM2-8M', 'ESM2-35M', 'ESM2-150M', 'ESM2-650M']:
    for window in WINDOWS:
        train_grid(plm, window, configs=CONFIGS_H512, out_dir=f'{TRAIN_DIR}/grid_h512/{plm}/{window}')
