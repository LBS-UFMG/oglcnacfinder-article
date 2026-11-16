# %%
import os
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Grid results: one row per pLM, window option and configuration (best validation AUROC)
# ---------------------------------------------------------------------------
GRID_DIR, FIG_DIR = 'data_processed/training/grid', 'results/figures/grid_validation'
PLMS = ['ESM2-8M', 'ESM2-35M', 'ESM2-150M', 'ESM2-650M']
WINDOWS = ['residue', 'pm05', 'pm10', 'pm15', 'pm20', 'pm25', 'pm35']
os.makedirs(FIG_DIR, exist_ok=True)

# Main grid (18 configurations per pair) and its hidden-512 extension (6 per pair)
GRIDS = {'data_processed/training/grid': 18, 'data_processed/training/grid_h512': 6}
grid = pd.concat([pd.read_csv(f'{d}/{p}/{w}/results.csv').assign(plm=p, window=w)
                  for d in GRIDS for p in PLMS for w in WINDOWS], ignore_index=True)
assert len(grid) == len(PLMS) * len(WINDOWS) * sum(GRIDS.values())
assert not grid.duplicated(['plm', 'window', 'hidden', 'dropout', 'lr']).any()
grid.to_csv('data_processed/training/grid_results.csv', index=False)
print(f'{len(grid)} runs   best validation AUROC from {grid["best_val_auroc"].min():.4f} '
      f'to {grid["best_val_auroc"].max():.4f}')

# %%
import io
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

# ---------------------------------------------------------------------------
# Parameter-effects figure: how much each parameter matters (figure number set with the manuscript)
#   One panel per parameter. Grey dots: every run of the grid at each value of the parameter
#   (672 runs per panel, spread horizontally so they do not overlap). Blue line: the median
#   of all runs at each value. A parameter matters when the line moves; the spread of the
#   grey dots shows how sensitive the result is to the other parameters.
#   Frontiers figure rules: 180 mm wide, text >= 8 pt and lines >= 2 pt at final size,
#   panels labelled (A)-(E) outside the plot area, axis labels with units; PDF and RGB TIFF
#   at 300 dpi.
# ---------------------------------------------------------------------------
BLUE, GREY = '#2a78d6', '#6b6a66'
EFFECT_PANELS = [('window', WINDOWS, ['residue\nonly', '±5', '±10', '±15', '±20', '±25', '±35'], 'Window (± residues)'),
                 ('plm', PLMS, ['8M', '35M', '150M', '650M'], 'Protein language model (ESM-2)'),
                 ('hidden', [16, 64, 256, 512], ['16', '64', '256', '512'], 'Hidden units'),
                 ('dropout', [0.0, 0.25, 0.5], ['0', '0.25', '0.5'], 'Dropout rate'),
                 ('lr', [1e-4, 1e-5], ['$10^{-4}$', '$10^{-5}$'], 'Learning rate')]
YLIM = (grid['best_val_auroc'].min() - 0.002, grid['best_val_auroc'].max() + 0.002)

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

fig = plt.figure(figsize=(180 / 25.4, 125 / 25.4), layout='constrained')
gs = fig.add_gridspec(2, 6)
axes = [fig.add_subplot(gs[0, :4]), fig.add_subplot(gs[0, 4:]),
        fig.add_subplot(gs[1, :2]), fig.add_subplot(gs[1, 2:4]), fig.add_subplot(gs[1, 4:])]
rng = np.random.default_rng(0)
for k, (ax, (param, levels, ticks, label)) in enumerate(zip(axes, EFFECT_PANELS)):
    median_per_level = []
    for i, level in enumerate(levels):
        values = grid.loc[grid[param] == level, 'best_val_auroc'].to_numpy()
        ax.scatter(i + rng.uniform(-0.3, 0.3, len(values)), values, s=5, color=GREY, alpha=0.35, linewidths=0)
        median_per_level.append(np.median(values))
    ax.plot(range(len(levels)), median_per_level, color=BLUE, marker='o', markersize=5, zorder=3)
    ax.set_xticks(range(len(levels)), ticks)
    ax.set_xlim(-0.6, len(levels) - 0.4)
    ax.set_ylim(YLIM)
    ax.set_title(f'({chr(65 + k)}) {label}', loc='left')
    if k in (0, 2):
        ax.set_ylabel('Validation AUROC')
    else:
        ax.tick_params(labelleft=False)
fig.legend([Line2D([], [], color=GREY, marker='o', linestyle='', markersize=4, alpha=0.6),
            Line2D([], [], color=BLUE, marker='o', markersize=5)],
           ['Every run (one classifier configuration)', 'Median of all runs at each value'],
           loc='outside lower center', ncol=2, frameon=False)
save_figure(fig, 'figure_supplementary_parameter_effects', 180)
plt.show()

# %%
import os
from itertools import combinations

import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Paired comparisons of hyperparameter values (validation AUROC)
#   For each parameter and each pair of its values (A, B), runs that differ only in that
#   parameter are paired: one pair per combination of the other parameters. The paired
#   differences (A - B) are tested with the two-sided Wilcoxon signed-rank test, a test whose
#   result depends only on the two values compared (Benavoli, Corani & Mangili 2016, JMLR
#   17(5):1-10, sec. 5). Family-wise error is controlled with the Holm correction within each
#   parameter (each parameter is one question). Reported with each test: number of pairs,
#   median and mean difference, and the share of pairs where A wins, ties or loses.
#   The pairs share the same training and validation data, so they are not independent:
#   the tests measure how consistent a difference is across our settings, not how it
#   would generalize to new data.
#   Hidden sizes compared: 16, 64, 256, 512.
# ---------------------------------------------------------------------------
HIDDEN_SIZES = [16, 64, 256, 512]
data = grid[grid['hidden'].isin(HIDDEN_SIZES)]
PARAMS = {'window': WINDOWS, 'plm': PLMS, 'hidden': HIDDEN_SIZES, 'dropout': [0.0, 0.25, 0.5], 'lr': [1e-4, 1e-5]}

def value_label(param, v):
    if param == 'plm':
        return v.replace('ESM2-', '')
    if param == 'window':
        return 'residue' if v == 'residue' else f'±{int(v[2:])}'
    if param == 'lr':
        return f'{v:.0e}'.replace('e-0', 'e-')
    return f'{v:g}'

rows = []
for param, levels in PARAMS.items():
    others = [p for p in PARAMS if p != param]
    n_blocks = int(np.prod([len(PARAMS[p]) for p in others]))
    family = []
    for a, b in combinations(levels, 2):
        pa = data[data[param] == a].set_index(others)['best_val_auroc']
        pb = data[data[param] == b].set_index(others)['best_val_auroc']
        diff = (pa - pb.reindex(pa.index)).to_numpy()
        assert len(diff) == n_blocks and not np.isnan(diff).any(), (param, a, b)
        statistic, p = wilcoxon(diff, alternative='two-sided')
        family.append({'parameter': param, 'A': value_label(param, a), 'B': value_label(param, b),
                       'pairs': len(diff), 'median_diff': np.median(diff), 'mean_diff': diff.mean(),
                       'A_wins': (diff > 0).mean(), 'ties': (diff == 0).mean(), 'B_wins': (diff < 0).mean(),
                       'W': statistic, 'p': p})
    p_holm = multipletests([r['p'] for r in family], method='holm')[1]
    for r, q in zip(family, p_holm):
        r['p_holm'] = q
    rows += family

tests = pd.DataFrame(rows)
os.makedirs('results/tables', exist_ok=True)
tests.to_csv('results/tables/pairwise_wilcoxon_validation.csv', index=False)

# ---------------------------------------------------------------------------
# Report: one section per parameter. Comparisons are listed in both directions, (A, B) and
# (B, A), grouped by A; each test is still counted once in the Holm correction and in the CSV.
# Result column: which value is significantly better (Holm p < 0.05; * < 0.05, ** < 0.01,
# *** < 0.001), or "n.s.". Each section ends with the significant wins and losses of every value.
# ---------------------------------------------------------------------------
NAMES = {'window': 'Window (± residues)', 'plm': 'Protein language model (ESM2)', 'hidden': 'Hidden units',
         'dropout': 'Dropout rate', 'lr': 'Learning rate'}

def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

def report(param, levels):
    t = tests[tests['parameter'] == param]
    forward = t.assign(sign=1)
    backward = t.rename(columns={'A': 'B', 'B': 'A', 'A_wins': 'B_wins', 'B_wins': 'A_wins'}).assign(sign=-1)
    both = pd.concat([forward, backward])
    order = {value_label(param, v): i for i, v in enumerate(levels)}
    both = both.sort_values(['A', 'B'], key=lambda c: c.map(order))

    header = (f"{'A':>8} {'B':>8} {'pairs':>6} {'median diff':>12} {'A wins':>7} {'ties':>5} "
              f"{'B wins':>7} {'p':>9} {'p (Holm)':>9}   result")
    rule = '-' * len(header)
    print(f"\n{NAMES[param]}: {len(t)} comparisons, {t['pairs'].iloc[0]} paired runs each, "
          f"Holm correction within this parameter")
    print(rule)
    print(header)
    print(rule)
    previous = None
    for _, r in both.iterrows():
        if previous is not None and r['A'] != previous:
            print()
        previous = r['A']
        diff = r['sign'] * r['median_diff']
        if r['p_holm'] < 0.05:
            result = f"{r['A'] if diff > 0 else r['B']} better {stars(r['p_holm'])}"
        else:
            result = 'n.s.'
        print(f"{r['A']:>8} {r['B']:>8} {r['pairs']:>6} {diff:>+12.4f} {r['A_wins']:>7.0%} {r['ties']:>5.0%} "
              f"{r['B_wins']:>7.0%} {r['p']:>9.1e} {r['p_holm']:>9.1e}   {result}")
    print(rule)

    significant = t[t['p_holm'] < 0.05]
    winners = np.where(significant['median_diff'] > 0, significant['A'], significant['B'])
    losers = np.where(significant['median_diff'] > 0, significant['B'], significant['A'])
    record = {lab: (int((winners == lab).sum()), int((losers == lab).sum())) for lab in order}
    ranked = sorted(record, key=lambda lab: (record[lab][1] - record[lab][0], order[lab]))
    print('Significant wins-losses:  ' + '   '.join(f'{lab} {record[lab][0]}-{record[lab][1]}' for lab in ranked))

for param, levels in PARAMS.items():
    report(param, levels)

# %%
import io
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle, FancyArrowPatch
from PIL import Image

# ---------------------------------------------------------------------------
# Pairwise-comparison figure: hyperparameter values as tournament graphs (figure number set
# with the manuscript)
#   One panel per parameter; its values are placed on a circle. Each comparison is an arrow
#   from the worse value to the better one, so the best value is where the arrowheads meet.
#   Blue arrow, solid head: the better value is significantly better (paired Wilcoxon
#   signed-rank test, Holm-adjusted p < 0.05 within the parameter). Grey arrow, hollow head:
#   better but not significantly (the two kinds differ by shape as well as colour).
#   Better = positive median paired difference (a grey line without a head would mark a
#   median difference of exactly zero). Results from the paired-comparison cell (`tests`).
#   Value labels as written in the manuscript (DISPLAY_LABELS; display only).
#   Frontiers figure rules: 180 mm wide, text >= 8 pt and lines >= 2 pt at final size,
#   panels labelled (A)-(E) outside the drawing, solid lines only, no red or green;
#   PDF and RGB TIFF at 300 dpi.
# ---------------------------------------------------------------------------
WIN_COLOUR, DRAW_COLOUR, NODE_EDGE = '#1c5cab', '#8a8984', '#0b0b0b'
NODE_RADIUS, LABEL_RADIUS, ARROW_SCALE, ALPHA = 0.07, 1.15, 13, 0.05
PANEL_TITLES = {'window': 'Window (± residues)', 'plm': 'Protein language model (ESM-2)',
                'hidden': 'Hidden units', 'dropout': 'Dropout rate', 'lr': 'Learning rate'}
DISPLAY_LABELS = {'residue': 'residue only', '1e-4': '$10^{-4}$', '1e-5': '$10^{-5}$'}

def draw_tournament(ax, param, levels):
    """Values as dots on a circle (first at the top, clockwise), labelled outside; one arrow per
    comparison, pointing to the better value."""
    labels = [value_label(param, v) for v in levels]
    angles = np.pi / 2 - 2 * np.pi * np.arange(len(labels)) / len(labels)
    if len(labels) == 2:
        angles = np.array([np.pi, 0.0])
    pos = {lab: np.array([np.cos(a), np.sin(a)]) for lab, a in zip(labels, angles)}
    for _, r in tests[tests['parameter'] == param].iterrows():
        loser, winner = (r['B'], r['A']) if r['median_diff'] > 0 else (r['A'], r['B'])
        u = (pos[winner] - pos[loser]) / np.linalg.norm(pos[winner] - pos[loser])
        start, end = pos[loser] + (NODE_RADIUS + 0.05) * u, pos[winner] - (NODE_RADIUS + 0.05) * u
        significant = r['p_holm'] < ALPHA
        colour = WIN_COLOUR if significant else DRAW_COLOUR
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>' if r['median_diff'] != 0 else '-',
                                     mutation_scale=ARROW_SCALE, linewidth=2, zorder=2 if significant else 1,
                                     edgecolor=colour, facecolor=colour if significant else 'white'))
    for lab in labels:
        ax.add_patch(Circle(pos[lab], NODE_RADIUS, color=NODE_EDGE, zorder=3))
        x, y = pos[lab]
        ax.text(*(LABEL_RADIUS * pos[lab]), DISPLAY_LABELS.get(lab, lab), fontsize=8,
                ha='left' if x > 0.3 else 'right' if x < -0.3 else 'center',
                va='bottom' if y > 0.3 else 'top' if y < -0.3 else 'center')
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal')
    ax.axis('off')

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'legend.fontsize': 8})
fig = plt.figure(figsize=(180 / 25.4, 150 / 25.4), layout='constrained')
gs = fig.add_gridspec(2, 6, height_ratios=[1.4, 1])
axes = [fig.add_subplot(gs[0, 0:4]), fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4]), fig.add_subplot(gs[1, 4:6])]
for k, (ax, (param, levels)) in enumerate(zip(axes, PARAMS.items())):
    draw_tournament(ax, param, levels)
    ax.set_title(f'({chr(65 + k)}) {PANEL_TITLES[param]}', loc='left')

class ArrowHandler(HandlerPatch):
    """Legend entry drawn as the same arrow used in the panels."""
    def create_artists(self, legend, orig, xdescent, ydescent, width, height, fontsize, trans):
        arrow = FancyArrowPatch((0, height / 2), (width, height / 2), arrowstyle='-|>', mutation_scale=ARROW_SCALE,
                                edgecolor=orig.get_edgecolor(), facecolor=orig.get_facecolor(), linewidth=2,
                                transform=trans)
        return [arrow]

legend_arrows = [FancyArrowPatch((0, 0), (1, 0), edgecolor=WIN_COLOUR, facecolor=WIN_COLOUR),
                 FancyArrowPatch((0, 0), (1, 0), edgecolor=DRAW_COLOUR, facecolor='white')]
fig.legend(legend_arrows,
           ['Arrow points to the significantly better value (Wilcoxon, Holm p < 0.05)',
            'Arrow points to the better value, not significant'],
           handler_map={FancyArrowPatch: ArrowHandler()}, loc='outside lower center', ncol=1, frameon=False)

assert abs(fig.get_size_inches()[0] * 25.4 - 180) < 1e-6
fig.savefig(f'{FIG_DIR}/figure4_pairwise_tournaments.pdf')
buffer = io.BytesIO()
fig.savefig(buffer, format='png', dpi=300)
Image.open(buffer).convert('RGB').save(f'{FIG_DIR}/figure4_pairwise_tournaments.tif', dpi=(300, 300))
plt.show()

# %%
# ---------------------------------------------------------------------------
# Full ranking of all runs (every pLM, window option and classifier configuration) by
# validation AUROC. The selected configuration is marked in the printed table and in the CSV.
# ---------------------------------------------------------------------------
CHOSEN = {'window': 'pm20', 'plm': 'ESM2-150M', 'hidden': 256, 'dropout': 0.25, 'lr': 1e-4}

ranking = grid.sort_values('best_val_auroc', ascending=False).reset_index(drop=True)
ranking.insert(0, 'rank', np.arange(1, len(ranking) + 1))
is_chosen = (ranking[list(CHOSEN)] == pd.Series(CHOSEN)).all(axis=1).to_numpy()
assert is_chosen.sum() == 1, 'the chosen configuration must match exactly one run'

table = ranking[['rank', 'plm', 'window', 'hidden', 'dropout', 'lr', 'best_epoch', 'best_val_auroc']].copy()
for param in ['plm', 'window', 'dropout', 'lr']:
    table[param] = [value_label(param, v) for v in table[param]]
table['chosen'] = np.where(is_chosen, 'yes', '')
table.to_csv('results/tables/ranking_all_runs.csv', index=False)

best, chosen = ranking.iloc[0], ranking[is_chosen].iloc[0]
print(f"Chosen configuration: rank {chosen['rank']} of {len(ranking)}   validation AUROC {chosen['best_val_auroc']:.4f}   "
      f"best run {best['best_val_auroc']:.4f} (difference {best['best_val_auroc'] - chosen['best_val_auroc']:.4f})\n")

lines = (table.drop(columns='chosen')
         .rename(columns={'plm': 'ESM2', 'best_epoch': 'best epoch', 'best_val_auroc': 'val AUROC'})
         .to_string(index=False, formatters={'val AUROC': '{:.4f}'.format}).splitlines())
rule = '-' * len(lines[0])
print(rule, lines[0], rule, sep='\n')
for line, mark in zip(lines[1:], is_chosen):
    print(f'{line}   <== chosen' if mark else line)
print(rule)

# %%
import json

# ---------------------------------------------------------------------------
# Training compute of the grid, for the compute statement
#   Wall-clock minutes from each (pLM, window) pair's done.json, written by train_grid (NB03): the
#   training time of all configurations of the pair, trained side by side, and the time to load its
#   features. Grids and pairs as in the first cell (GRIDS, PLMS, WINDOWS).
#   Output: results/tables/compute_grid_training.csv (one row per grid, pLM and window).
# ---------------------------------------------------------------------------
TABLE_DIR = 'results/tables'
os.makedirs(TABLE_DIR, exist_ok=True)

rows = []
for d, n_configs in GRIDS.items():
    for p in PLMS:
        for w in WINDOWS:
            with open(f'{d}/{p}/{w}/done.json') as f:
                done = json.load(f)
            rows.append({'grid': os.path.basename(d), 'plm': p, 'window': w, 'runs': n_configs,
                         'training_minutes': done['training_minutes'], 'loading_minutes': done['loading_minutes'],
                         'epochs': done['epochs']})
compute = pd.DataFrame(rows)
compute.to_csv(f'{TABLE_DIR}/compute_grid_training.csv', index=False)

summary = compute.groupby(['grid', 'plm'], sort=False)[['runs', 'training_minutes', 'loading_minutes']].sum()
per_grid = compute.groupby('grid', sort=False)[['runs', 'training_minutes', 'loading_minutes']].sum()
with pd.option_context('display.float_format', '{:.1f}'.format):
    print(summary.to_string())
    print('\nPer grid:')
    print(per_grid.to_string())
total = per_grid.sum()
print(f'\nAll grids: {int(total["runs"])} runs, {total["training_minutes"]:.1f} min training '
      f'+ {total["loading_minutes"]:.1f} min loading features')
