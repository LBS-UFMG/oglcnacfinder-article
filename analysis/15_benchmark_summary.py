# %%
import os
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))

import io
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# Benchmark summary: fair comparison with existing O-GlcNAc site predictors
#   For each tool, the subset of our test set without homologs (>= 30% identity, MMseqs2) of the
#   proteins in that tool's training data (filter a; LM-OGlcNAc-Site: all proteins of the data
#   snapshot in its repository history, NB06; O-GlcNAcPRED-DL: proteins traced from the windows
#   on its web server, NB07). Our model is scored on the same sites, so its values differ between
#   rows. Metrics as saved by each benchmark notebook (NB06-NB11); calls at each tool's own
#   cut-off and at our validation cut-off.
#   Outputs: CSV, a LaTeX table (better value of each pair in bold) and a figure of the table
#   (higher AUROC of each comparison in bold).
# ---------------------------------------------------------------------------
TOOL_DIR = 'data_processed/other_tools'
OUT_DIR = 'data_processed/benchmark_summary'
os.makedirs(OUT_DIR, exist_ok=True)
METRICS = ['AUROC', 'AUPRC', 'MCC']

# (name in the paper, metric file, text identifying the tool's row in the 'model' column, seen data used for the filter)
TOOLS = [('LM-OGlcNAc-Site', 'LM-OGlcNAc-Site/test_metrics_filtered_snapshot.csv', '33', 'their training data (repository snapshot)'),
         ('O-GlcNAcPRED-DL', 'O-GlcNAcPRED-DL/test_metrics_filtered_server.csv', 'matched', 'their training data (web server)'),
         ('RNN-[600,75] (Seber)', 'Seber/test_metrics_filtered.csv', 'RNN-[600,75]', 'their training data'),
         ('RNN-225 (Seber and Braatz)', 'Seber/test_metrics_filtered.csv', 'RNN-225', 'their training data'),
         ('Stack-OglyPred-PLM', 'Stack-OglyPred-PLM/test_metrics_filtered.csv', 'Stack', 'their training data'),
         ('BSB 2025 (Arcanjo et al.)', 'BSB2025/test_metrics_filtered.csv', 'BSB', 'their training and validation data')]

def is_ours(model):
    return model.str.strip().str.lower().isin(['our model', 'ours'])

rows = []
for name, file, key, seen in TOOLS:
    df = pd.read_csv(f'{TOOL_DIR}/{file}')
    if 'organisms' in df.columns:
        df = df[df['organisms'] == 'all organisms']
    fair = df[df['subset'].str.contains('homologs')]
    theirs = fair[fair['model'].str.contains(key, regex=False) & ~is_ours(fair['model'])]
    ours = fair[is_ours(fair['model'])]
    assert len(theirs) == 1 and len(ours) == 1, f'{name}: models in file {sorted(df["model"].unique())}'
    t, o = theirs.iloc[0], ours.iloc[0]
    rows.append({'tool': name, 'proteins': int(t['proteins']), 'sites': int(t['sites']), 'positive': int(t['positive']),
                 **{f'{m} {who}': r[m] for m in METRICS for who, r in [('theirs', t), ('ours', o)]},
                 'seen data (filter)': seen})
table = pd.DataFrame(rows)
table.to_csv(f'{OUT_DIR}/benchmark_fair_subset.csv', index=False)
print(table.drop(columns='seen data (filter)').to_string(index=False, float_format='{:.3f}'.format))
print('\nOurs higher: ' + ', '.join(f'{m} {(table[f"{m} ours"] > table[f"{m} theirs"]).sum()}/{len(table)}' for m in METRICS))

# LaTeX (booktabs): better value of each pair in bold
def pair(t, o):
    f = lambda v, best: r'\textbf{' + f'{v:.3f}' + '}' if best else f'{v:.3f}'
    return f(t, t > o) + ' & ' + f(o, o > t)

lines = [r'\begin{tabular}{lrrcccccc}', r'\toprule',
         r' & & & \multicolumn{2}{c}{AUROC} & \multicolumn{2}{c}{AUPRC} & \multicolumn{2}{c}{MCC} \\',
         r'\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}',
         r'Tool & Proteins & Positives & Tool & Ours & Tool & Ours & Tool & Ours \\', r'\midrule']
for r in table.to_dict('records'):
    lines.append(r['tool'].replace('<=', r'$\leq$') + f' & {r["proteins"]:,} & {r["positive"]:,} & '
                 + ' & '.join(pair(r[f'{m} theirs'], r[f'{m} ours']) for m in METRICS) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
with open(f'{OUT_DIR}/benchmark_fair_subset.tex', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('\n' + '\n'.join(lines))

# Figure of the table (PDF + RGB TIFF at 300 dpi; 180 mm wide, 8 pt): higher AUROC of each row in bold
FIG_DIR = 'results/figures/benchmark'
os.makedirs(FIG_DIR, exist_ok=True)
header = ['Tool', 'Proteins', 'Positives'] + [f'{m}\n{who}' for m in METRICS for who in ('tool', 'ours')]
body = [[r['tool'].replace('<=', '≤'), f'{r["proteins"]:,}', f'{r["positive"]:,}']
        + [f'{r[f"{m} {who}"]:.3f}' for m in METRICS for who in ('theirs', 'ours')] for r in table.to_dict('records')]

plt.rcParams.update({'font.size': 8})
fig, ax = plt.subplots(figsize=(180 / 25.4, (len(body) + 2) * 0.27), layout='constrained')
ax.axis('off')
tab = ax.table(cellText=body, colLabels=header, loc='center', cellLoc='center',
               colWidths=[0.30, 0.09, 0.09] + [0.087] * 6)
tab.auto_set_font_size(False)
tab.set_fontsize(8)
tab.scale(1, 1.45)
for (i, j), cell in tab.get_celld().items():
    cell.visible_edges = 'TB' if i == 0 else ('B' if i == len(body) else '')   # rules as in booktabs
    if j == 0:
        cell.set_text_props(ha='left')
        cell.PAD = 0.02
    if i == 0:
        cell.set_text_props(weight='bold')
for i, r in enumerate(table.to_dict('records'), start=1):
    for k, m in enumerate(METRICS):
        col = 3 + 2 * k + (r[f'{m} ours'] > r[f'{m} theirs'])
        tab[i, col].set_text_props(weight='bold')
fig.savefig(f'{FIG_DIR}/table_benchmark_fair_subset.pdf')
buffer = io.BytesIO()
fig.savefig(buffer, format='png', dpi=300)
Image.open(buffer).convert('RGB').save(f'{FIG_DIR}/table_benchmark_fair_subset.tif', dpi=(300, 300))
plt.show()

# %%
# ---------------------------------------------------------------------------
# Benchmark summary, second table: proteins identical to the tool's training proteins removed
#   (filter b). Same layout as the previous table; reuses TOOLS, is_ours, pair, METRICS and the
#   folders of the previous cell.
#   Outputs: CSV, a LaTeX table and a figure of the table (better value of each pair in bold).
# ---------------------------------------------------------------------------
rows = []
for name, file, key, seen in TOOLS:
    df = pd.read_csv(f'{TOOL_DIR}/{file}')
    if 'organisms' in df.columns:
        df = df[df['organisms'] == 'all organisms']
    sub = df[df['subset'].str.contains('same protein', regex=False)]
    theirs = sub[sub['model'].str.contains(key, regex=False) & ~is_ours(sub['model'])]
    ours = sub[is_ours(sub['model'])]
    assert len(theirs) == 1 and len(ours) == 1, f'{name}: subsets {sorted(df["subset"].unique())}, models {sorted(df["model"].unique())}'
    t, o = theirs.iloc[0], ours.iloc[0]
    rows.append({'tool': name, 'proteins': int(t['proteins']), 'sites': int(t['sites']), 'positive': int(t['positive']),
                 **{f'{m} {who}': r[m] for m in METRICS for who, r in [('theirs', t), ('ours', o)]},
                 'seen data (filter)': seen})
table_b = pd.DataFrame(rows)
table_b.to_csv(f'{OUT_DIR}/benchmark_same_protein_removed.csv', index=False)
print(table_b.drop(columns='seen data (filter)').to_string(index=False, float_format='{:.3f}'.format))
print('\nOurs higher: ' + ', '.join(f'{m} {(table_b[f"{m} ours"] > table_b[f"{m} theirs"]).sum()}/{len(table_b)}' for m in METRICS))

# LaTeX (booktabs): better value of each pair in bold
lines = [r'\begin{tabular}{lrrcccccc}', r'\toprule',
         r' & & & \multicolumn{2}{c}{AUROC} & \multicolumn{2}{c}{AUPRC} & \multicolumn{2}{c}{MCC} \\',
         r'\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}',
         r'Tool & Proteins & Positives & Tool & Ours & Tool & Ours & Tool & Ours \\', r'\midrule']
for r in table_b.to_dict('records'):
    lines.append(r['tool'] + f' & {r["proteins"]:,} & {r["positive"]:,} & '
                 + ' & '.join(pair(r[f'{m} theirs'], r[f'{m} ours']) for m in METRICS) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
with open(f'{OUT_DIR}/benchmark_same_protein_removed.tex', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('\n' + '\n'.join(lines))

# Figure of the table (PDF + RGB TIFF at 300 dpi; 180 mm wide, 8 pt): better value of each pair in bold
header = ['Tool', 'Proteins', 'Positives'] + [f'{m}\n{who}' for m in METRICS for who in ('tool', 'ours')]
body = [[r['tool'], f'{r["proteins"]:,}', f'{r["positive"]:,}']
        + [f'{r[f"{m} {who}"]:.3f}' for m in METRICS for who in ('theirs', 'ours')] for r in table_b.to_dict('records')]
plt.rcParams.update({'font.size': 8})
fig, ax = plt.subplots(figsize=(180 / 25.4, (len(body) + 2) * 0.27), layout='constrained')
ax.axis('off')
tab = ax.table(cellText=body, colLabels=header, loc='center', cellLoc='center',
               colWidths=[0.30, 0.09, 0.09] + [0.087] * 6)
tab.auto_set_font_size(False)
tab.set_fontsize(8)
tab.scale(1, 1.45)
for (i, j), cell in tab.get_celld().items():
    cell.visible_edges = 'TB' if i == 0 else ('B' if i == len(body) else '')   # rules as in booktabs
    if j == 0:
        cell.set_text_props(ha='left')
        cell.PAD = 0.02
    if i == 0:
        cell.set_text_props(weight='bold')
for i, r in enumerate(table_b.to_dict('records'), start=1):
    for k, m in enumerate(METRICS):
        col = 3 + 2 * k + (r[f'{m} ours'] > r[f'{m} theirs'])
        tab[i, col].set_text_props(weight='bold')
fig.savefig(f'{FIG_DIR}/table_benchmark_same_protein_removed.pdf')
buffer = io.BytesIO()
fig.savefig(buffer, format='png', dpi=300)
Image.open(buffer).convert('RGB').save(f'{FIG_DIR}/table_benchmark_same_protein_removed.tif', dpi=(300, 300))
plt.show()

# %%
# ---------------------------------------------------------------------------
# Benchmark summary, third table: all test proteins (no leakage filter). Same layout as the previous
#   tables; reuses TOOLS, is_ours, pair, METRICS and the folders of the first cell. The BSB 2025
#   model takes proteins of <= 1,022 residues only, so its row covers 517 of the 736 test proteins.
#   Outputs: CSV, a LaTeX table and a figure of the table (better value of each pair in bold).
# ---------------------------------------------------------------------------
rows = []
for name, file, key, seen in TOOLS:
    df = pd.read_csv(f'{TOOL_DIR}/{file}')
    if 'organisms' in df.columns:
        df = df[df['organisms'] == 'all organisms']
    sub = df[df['subset'].str.contains('all test proteins', regex=False)]
    theirs = sub[sub['model'].str.contains(key, regex=False) & ~is_ours(sub['model'])]
    ours = sub[is_ours(sub['model'])]
    assert len(theirs) == 1 and len(ours) == 1, f'{name}: subsets {sorted(df["subset"].unique())}, models {sorted(df["model"].unique())}'
    t, o = theirs.iloc[0], ours.iloc[0]
    rows.append({'tool': name, 'proteins': int(t['proteins']), 'sites': int(t['sites']), 'positive': int(t['positive']),
                 **{f'{m} {who}': r[m] for m in METRICS for who, r in [('theirs', t), ('ours', o)]},
                 'seen data (filter)': seen})
table_all = pd.DataFrame(rows)
table_all.to_csv(f'{OUT_DIR}/benchmark_all_test_proteins.csv', index=False)
print(table_all.drop(columns='seen data (filter)').to_string(index=False, float_format='{:.3f}'.format))
print('\nOurs higher: ' + ', '.join(f'{m} {(table_all[f"{m} ours"] > table_all[f"{m} theirs"]).sum()}/{len(table_all)}' for m in METRICS))

# LaTeX (booktabs): better value of each pair in bold
lines = [r'\begin{tabular}{lrrcccccc}', r'\toprule',
         r' & & & \multicolumn{2}{c}{AUROC} & \multicolumn{2}{c}{AUPRC} & \multicolumn{2}{c}{MCC} \\',
         r'\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}',
         r'Tool & Proteins & Positives & Tool & Ours & Tool & Ours & Tool & Ours \\', r'\midrule']
for r in table_all.to_dict('records'):
    lines.append(r['tool'] + f' & {r["proteins"]:,} & {r["positive"]:,} & '
                 + ' & '.join(pair(r[f'{m} theirs'], r[f'{m} ours']) for m in METRICS) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
with open(f'{OUT_DIR}/benchmark_all_test_proteins.tex', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('\n' + '\n'.join(lines))

# Figure of the table (PDF + RGB TIFF at 300 dpi; 180 mm wide, 8 pt): better value of each pair in bold
header = ['Tool', 'Proteins', 'Positives'] + [f'{m}\n{who}' for m in METRICS for who in ('tool', 'ours')]
body = [[r['tool'], f'{r["proteins"]:,}', f'{r["positive"]:,}']
        + [f'{r[f"{m} {who}"]:.3f}' for m in METRICS for who in ('theirs', 'ours')] for r in table_all.to_dict('records')]
plt.rcParams.update({'font.size': 8})
fig, ax = plt.subplots(figsize=(180 / 25.4, (len(body) + 2) * 0.27), layout='constrained')
ax.axis('off')
tab = ax.table(cellText=body, colLabels=header, loc='center', cellLoc='center',
               colWidths=[0.30, 0.09, 0.09] + [0.087] * 6)
tab.auto_set_font_size(False)
tab.set_fontsize(8)
tab.scale(1, 1.45)
for (i, j), cell in tab.get_celld().items():
    cell.visible_edges = 'TB' if i == 0 else ('B' if i == len(body) else '')   # rules as in booktabs
    if j == 0:
        cell.set_text_props(ha='left')
        cell.PAD = 0.02
    if i == 0:
        cell.set_text_props(weight='bold')
for i, r in enumerate(table_all.to_dict('records'), start=1):
    for k, m in enumerate(METRICS):
        col = 3 + 2 * k + (r[f'{m} ours'] > r[f'{m} theirs'])
        tab[i, col].set_text_props(weight='bold')
fig.savefig(f'{FIG_DIR}/table_benchmark_all_test_proteins.pdf')
buffer = io.BytesIO()
fig.savefig(buffer, format='png', dpi=300)
Image.open(buffer).convert('RGB').save(f'{FIG_DIR}/table_benchmark_all_test_proteins.tif', dpi=(300, 300))
plt.show()

# %%
import numpy as np

# ---------------------------------------------------------------------------
# Benchmark summary, species-specific tools: each tool (or species model) on the test proteins of the
# species it was trained for (decision: organism-specific models are compared on their own
# species), homologs >= 30% of its training proteins removed (filter a).
#   O-GlcNAcPRED-DL: human model on human proteins, mouse model on mouse proteins (NB07);
#   Seber RNNs: mammalian proteins (NB08); Stack-OglyPred-PLM: human proteins (NB10).
#   A subset with no protein left is shown as '-'. Reuses is_ours, METRICS and the folders of the
#   first cell. Outputs: CSV, a LaTeX table and a figure of the table (better value of each pair
#   in bold).
# ---------------------------------------------------------------------------
# (name in the paper, metric file, text identifying the tool's row, 'organisms' value or None, test proteins)
SPECIES_TOOLS = [('O-GlcNAcPRED-DL, human model', 'O-GlcNAcPRED-DL/test_metrics_filtered_server.csv', 'human', 'Homo sapiens', 'human'),
                 ('O-GlcNAcPRED-DL, mouse model', 'O-GlcNAcPRED-DL/test_metrics_filtered_server.csv', 'mouse', 'Mus musculus', 'mouse'),
                 ('RNN-[600,75] (Seber)', 'Seber/test_metrics_mammalian.csv', 'RNN-[600,75]', None, 'mammals'),
                 ('RNN-225 (Seber and Braatz)', 'Seber/test_metrics_mammalian.csv', 'RNN-225', None, 'mammals'),
                 ('Stack-OglyPred-PLM', 'Stack-OglyPred-PLM/test_metrics_human.csv', 'Stack', None, 'human')]
SUBSET = 'homologs'

rows = []
for name, file, key, organism, species in SPECIES_TOOLS:
    df = pd.read_csv(f'{TOOL_DIR}/{file}')
    if organism is not None:
        df = df[df['organisms'] == organism]
    sub = df[df['subset'].str.contains(SUBSET, regex=False)]
    theirs = sub[sub['model'].str.contains(key, regex=False) & ~is_ours(sub['model'])]
    ours = sub[is_ours(sub['model'])]
    assert len(theirs) == 1 and len(ours) == 1, f'{name}: subsets {sorted(df["subset"].unique())}, models {sorted(df["model"].unique())}'
    t, o = theirs.iloc[0], ours.iloc[0]
    rows.append({'tool': name, 'test proteins': species, 'proteins': int(t['proteins']), 'sites': int(t['sites']),
                 'positive': int(t['positive']),
                 **{f'{m} {who}': r[m] for m in METRICS for who, r in [('theirs', t), ('ours', o)]}})
table_sp = pd.DataFrame(rows)
table_sp.to_csv(f'{OUT_DIR}/benchmark_species_fair_subset.csv', index=False)
print(table_sp.to_string(index=False, float_format='{:.3f}'.format, na_rep='-'))
scored = table_sp[table_sp['proteins'] > 0]
print('\nOurs higher (subsets with proteins): '
      + ', '.join(f'{m} {(scored[f"{m} ours"] > scored[f"{m} theirs"]).sum()}/{len(scored)}' for m in METRICS))

def fmt(v, best):
    if pd.isna(v):
        return '-'
    return r'\textbf{' + f'{v:.3f}' + '}' if best else f'{v:.3f}'

# LaTeX (booktabs): better value of each pair in bold
lines = [r'\begin{tabular}{llrrcccccc}', r'\toprule',
         r' & & & & \multicolumn{2}{c}{AUROC} & \multicolumn{2}{c}{AUPRC} & \multicolumn{2}{c}{MCC} \\',
         r'\cmidrule(lr){5-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}',
         r'Tool & Test proteins & Proteins & Positives & Tool & Ours & Tool & Ours & Tool & Ours \\', r'\midrule']
for r in table_sp.to_dict('records'):
    cells = []
    for m in METRICS:
        t, o = r[f'{m} theirs'], r[f'{m} ours']
        cells += [fmt(t, t > o), fmt(o, o > t)]
    lines.append(f'{r["tool"]} & {r["test proteins"]} & {r["proteins"]:,} & {r["positive"]:,} & ' + ' & '.join(cells) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
with open(f'{OUT_DIR}/benchmark_species_fair_subset.tex', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('\n' + '\n'.join(lines))

# Figure of the table (PDF + RGB TIFF at 300 dpi; 180 mm wide, 8 pt): better value of each pair in bold
header = ['Tool', 'Test\nproteins', 'Proteins', 'Positives'] + [f'{m}\n{who}' for m in METRICS for who in ('tool', 'ours')]
body = [[r['tool'], r['test proteins'], f'{r["proteins"]:,}', f'{r["positive"]:,}']
        + ['-' if pd.isna(r[f'{m} {who}']) else f'{r[f"{m} {who}"]:.3f}' for m in METRICS for who in ('theirs', 'ours')]
        for r in table_sp.to_dict('records')]
plt.rcParams.update({'font.size': 8})
fig, ax = plt.subplots(figsize=(180 / 25.4, (len(body) + 2) * 0.27), layout='constrained')
ax.axis('off')
tab = ax.table(cellText=body, colLabels=header, loc='center', cellLoc='center',
               colWidths=[0.27, 0.08, 0.08, 0.08] + [0.082] * 6)
tab.auto_set_font_size(False)
tab.set_fontsize(8)
tab.scale(1, 1.45)
for (i, j), cell in tab.get_celld().items():
    cell.visible_edges = 'TB' if i == 0 else ('B' if i == len(body) else '')   # rules as in booktabs
    if j == 0:
        cell.set_text_props(ha='left')
        cell.PAD = 0.02
    if i == 0:
        cell.set_text_props(weight='bold')
for i, r in enumerate(table_sp.to_dict('records'), start=1):
    for k, m in enumerate(METRICS):
        t, o = r[f'{m} theirs'], r[f'{m} ours']
        if pd.notna(t) and pd.notna(o) and t != o:
            tab[i, 4 + 2 * k + (o > t)].set_text_props(weight='bold')
fig.savefig(f'{FIG_DIR}/table_benchmark_species_fair_subset.pdf')
buffer = io.BytesIO()
fig.savefig(buffer, format='png', dpi=300)
Image.open(buffer).convert('RGB').save(f'{FIG_DIR}/table_benchmark_species_fair_subset.tif', dpi=(300, 300))
plt.show()

# %%
# ---------------------------------------------------------------------------
# Figure: effect of training-data leakage on the comparison with each tool
#   One panel per tool: METRIC of the tool and of our model on all test proteins, with proteins
#   identical to the tool's training proteins removed (filter b), and with homologs >= 30% removed
#   (filter a). Values from the three tables above (CSV files of cells 1-3); number of positive
#   sites under each subset. The BSB 2025 model takes proteins of <= 1,022 residues only.
#   Frontiers figure rules: 180 mm wide, text >= 8 pt, solid lines only, no red or green; PDF and
#   RGB TIFF at 300 dpi.
# ---------------------------------------------------------------------------
METRIC = 'AUROC'
SUBSETS = [('benchmark_all_test_proteins', 'All'), ('benchmark_same_protein_removed', 'Identical\nremoved'),
           ('benchmark_fair_subset', 'Homologs\nremoved')]
TOOL_COLOUR, OUR_COLOUR = '#8a8984', '#2a78d6'

tables = {stem: pd.read_csv(f'{OUT_DIR}/{stem}.csv').set_index('tool') for stem, _ in SUBSETS}
tools = list(tables['benchmark_fair_subset'].index)
values = pd.DataFrame([{'tool': t, 'subset': label, 'positive': tables[stem].at[t, 'positive'],
                        'theirs': tables[stem].at[t, f'{METRIC} theirs'], 'ours': tables[stem].at[t, f'{METRIC} ours']}
                       for t in tools for stem, label in SUBSETS])
values.to_csv(f'{OUT_DIR}/leakage_effect_{METRIC}.csv', index=False)

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False})
n_cols = 3
n_rows = -(-len(tools) // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(180 / 25.4, 55 * n_rows / 25.4), sharey=True, layout='constrained')
lo = values[['theirs', 'ours']].min().min()
hi = values[['theirs', 'ours']].max().max()
for ax, tool in zip(axes.flat, tools):
    d = values[values['tool'] == tool]
    x = range(len(d))
    ax.plot(x, d['theirs'], color=TOOL_COLOUR, marker='o', markersize=4, linewidth=1.5, label='Tool')
    ax.plot(x, d['ours'], color=OUR_COLOUR, marker='o', markersize=4, linewidth=1.5, label='Our model')
    ax.set_xticks(list(x), [f'{s}\n({p:,} pos.)' for s, p in zip(d['subset'], d['positive'])])
    ax.set_xlim(-0.3, len(d) - 0.7)
    ax.set_ylim(lo - 0.03, hi + 0.03)
    ax.set_title(tool, loc='left')
for ax in axes[:, 0]:
    ax.set_ylabel(METRIC)
for ax in list(axes.flat)[len(tools):]:
    ax.axis('off')
axes.flat[0].legend(frameon=False, loc='lower left')

fig.savefig(f'{FIG_DIR}/figure_leakage_effect_{METRIC}.pdf')
buffer = io.BytesIO()
fig.savefig(buffer, format='png', dpi=300)
Image.open(buffer).convert('RGB').save(f'{FIG_DIR}/figure_leakage_effect_{METRIC}.tif', dpi=(300, 300))
plt.show()

change = values.pivot(index='tool', columns='subset', values=['theirs', 'ours'])
print(f'{METRIC} from all test proteins to homologs removed:')
for t in tools:
    a, h = 'All', 'Homologs\nremoved'
    print(f'  {t}: tool {change.at[t, ("theirs", a)]:.3f} -> {change.at[t, ("theirs", h)]:.3f} '
          f'({change.at[t, ("theirs", h)] - change.at[t, ("theirs", a)]:+.3f});   ours {change.at[t, ("ours", a)]:.3f} -> '
          f'{change.at[t, ("ours", h)]:.3f} ({change.at[t, ("ours", h)] - change.at[t, ("ours", a)]:+.3f})')

# %%
# ---------------------------------------------------------------------------
# Supplementary table: overlap between our test proteins and each tool's training data
#   From the three tables above (CSV files of cells 1-3): test proteins scored, proteins identical
#   to a protein of the tool's training data (removed by filter b) and proteins with a homolog
#   >= 30% identity in it (removed by filter a), with the proteins and positive sites left.
#   Tools sharing the same seen data and the same counts are listed in one row. The BSB 2025 model
#   takes proteins of <= 1,022 residues only (517 of the 736 test proteins).
#   Outputs: CSV, a LaTeX table and a figure of the table.
# ---------------------------------------------------------------------------
t_all = pd.read_csv(f'{OUT_DIR}/benchmark_all_test_proteins.csv').set_index('tool')
t_same = pd.read_csv(f'{OUT_DIR}/benchmark_same_protein_removed.csv').set_index('tool')
t_fair = pd.read_csv(f'{OUT_DIR}/benchmark_fair_subset.csv').set_index('tool')

rows = []
for tool in t_fair.index:
    n = int(t_all.at[tool, 'proteins'])
    rows.append({'tool': tool, 'seen data (filter)': t_fair.at[tool, 'seen data (filter)'], 'test proteins': n,
                 'identical': n - int(t_same.at[tool, 'proteins']), 'homolog >= 30%': n - int(t_fair.at[tool, 'proteins']),
                 'proteins left': int(t_fair.at[tool, 'proteins']), 'positives left': int(t_fair.at[tool, 'positive']),
                 'positives (all)': int(t_all.at[tool, 'positive'])})
overlap = pd.DataFrame(rows)
keys = ['seen data (filter)', 'test proteins', 'identical', 'homolog >= 30%', 'proteins left', 'positives left', 'positives (all)']
overlap = overlap.groupby(keys, sort=False, as_index=False).agg(tool=('tool', ' / '.join))[['tool'] + keys]
overlap['identical %'] = 100 * overlap['identical'] / overlap['test proteins']
overlap['homolog %'] = 100 * overlap['homolog >= 30%'] / overlap['test proteins']
overlap.to_csv(f'{OUT_DIR}/benchmark_training_overlap.csv', index=False)
print(overlap.to_string(index=False, float_format='{:.0f}'.format))

# LaTeX (booktabs)
lines = [r'\begin{tabular}{llrrrrr}', r'\toprule',
         r'Tool & Seen data (filter) & Test proteins & Identical & Homolog $\geq$ 30\% & Proteins left & Positives left \\',
         r'\midrule']
for r in overlap.to_dict('records'):
    lines.append(f'{r["tool"]} & {r["seen data (filter)"]} & {r["test proteins"]:,} & '
                 f'{r["identical"]:,} ({r["identical %"]:.0f}\\%) & {r["homolog >= 30%"]:,} ({r["homolog %"]:.0f}\\%) & '
                 f'{r["proteins left"]:,} & {r["positives left"]:,} of {r["positives (all)"]:,}' + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
with open(f'{OUT_DIR}/benchmark_training_overlap.tex', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('\n' + '\n'.join(lines))

# Figure of the table (PDF + RGB TIFF at 300 dpi; 180 mm wide, 8 pt)
import textwrap
header = ['Tool', 'Seen data (filter)', 'Test\nproteins', 'Identical', 'Homolog\n≥ 30%', 'Proteins\nleft', 'Positives\nleft']
body = [[r['tool'].replace(' / ', ' /\n'), textwrap.fill(r['seen data (filter)'], 24), f'{r["test proteins"]:,}',
         f'{r["identical"]:,} ({r["identical %"]:.0f}%)', f'{r["homolog >= 30%"]:,} ({r["homolog %"]:.0f}%)',
         f'{r["proteins left"]:,}', f'{r["positives left"]:,} of {r["positives (all)"]:,}'] for r in overlap.to_dict('records')]
plt.rcParams.update({'font.size': 8})
fig, ax = plt.subplots(figsize=(180 / 25.4, (len(body) + 1) * 0.5), layout='constrained')
ax.axis('off')
tab = ax.table(cellText=body, colLabels=header, loc='center', cellLoc='center',
               colWidths=[0.25, 0.21, 0.08, 0.11, 0.11, 0.08, 0.12])
tab.auto_set_font_size(False)
tab.set_fontsize(8)
tab.scale(1, 2.4)
for (i, j), cell in tab.get_celld().items():
    cell.visible_edges = 'TB' if i == 0 else ('B' if i == len(body) else '')   # rules as in booktabs
    if j in (0, 1):
        cell.set_text_props(ha='left')
        cell.PAD = 0.02
    if i == 0:
        cell.set_text_props(weight='bold')
fig.savefig(f'{FIG_DIR}/table_benchmark_training_overlap.pdf')
buffer = io.BytesIO()
fig.savefig(buffer, format='png', dpi=300)
Image.open(buffer).convert('RGB').save(f'{FIG_DIR}/table_benchmark_training_overlap.tif', dpi=(300, 300))
plt.show()

# %%
import glob, json, re

# ---------------------------------------------------------------------------
# Compute summary (compute statement): wall-clock time and hardware of every timed step
#   Read from the records each notebook saves; nothing is typed in by hand except NB01's
#   all-vs-all alignment, whose time exists only in the NB01 output.
#     NB01  MMseqs2 all-vs-all alignment: MMseqs2 progress line in the NB01 output.
#     NB02  embeddings: data_processed/embeddings/{pLM}/residues.json (minutes, GPU) and
#           compute_extraction.csv (FLOPs); classifier features: data_processed/features/{pLM}/features.json.
#     NB03  training: done.json of the pilot runs and of every (pLM, window) pair of both grids
#           (summed per grid; NB04 sums the same files in compute_grid_training.csv).
#     NB05, NB07, NB08, NB10, NB11, NB14: results/compute/*.json (compute-records cells; NB10's CPU
#           part from scripts/stack_oglypred_plm_long_proteins.py).
#     NB06  LM-OGlcNAc-Site on our test set: compute_test.json.
#     LM-OGlcNAc-Site on the external sets (scripts/lm_oglcnac_site_external.py, lab server):
#           final progress line of external_run.log and external_provenance.json.
#   Output: results/tables/compute_summary.csv (one row per step).
# ---------------------------------------------------------------------------
TABLE_DIR = 'results/tables'
os.makedirs(TABLE_DIR, exist_ok=True)
PLMS = ['ESM2-8M', 'ESM2-35M', 'ESM2-150M', 'ESM2-650M']
rows = []

def add(notebook, step, seconds, hardware, source, **extra):
    rows.append({'notebook': notebook, 'step': step, 'seconds': seconds, 'hardware': hardware or 'not recorded',
                 **extra, 'source': source})

# NB01: printed only (MMseqs2 progress line "7.36K 1h 37m 45s 46ms"; "Time for processing: 1h 37m 46s 103ms")
add('NB01', 'MMseqs2 all-vs-all alignment (7,363 proteins)', 1 * 3600 + 37 * 60 + 46.1, 'Colab, 2 CPU cores',
    'NB01 output (MMseqs2 "Time for processing")', proteins=7363)

# NB02: embeddings and features per pLM
flops = pd.read_csv('data_processed/embeddings/compute_extraction.csv').set_index('model')['FLOPs']
for plm in PLMS:
    meta = json.load(open(f'data_processed/embeddings/{plm}/residues.json'))
    add('NB02', f'{plm} per-residue embeddings', meta['extraction_minutes'] * 60, meta.get('gpu'),
        f'data_processed/embeddings/{plm}/residues.json', flops=flops.get(plm))
for plm in PLMS:
    meta = json.load(open(f'data_processed/features/{plm}/features.json'))
    add('NB02', f'{plm} classifier features', meta['minutes'] * 60, meta.get('gpu'),
        f'data_processed/features/{plm}/features.json', st_sites=meta['shape'][0])

# NB03: pilot runs and grids
for path, step in [('data_processed/training/ESM2-8M/pm15/done.json', 'pilot, ESM2-8M +-15 (20 configurations)'),
                   ('data_processed/training/lr_test/ESM2-8M/pm15/done.json', 'pilot, learning-rate test')]:
    if os.path.exists(path):
        d = json.load(open(path))
        add('NB03', step, 60 * (d['training_minutes'] + d.get('loading_minutes', 0)), d.get('gpu'), path)
for grid, n_configs in [('grid', 18), ('grid_h512', 6)]:
    done = [json.load(open(p)) for p in sorted(glob.glob(f'data_processed/training/{grid}/*/*/done.json'))]
    add('NB03', f'{grid}: {len(done)} (pLM, window) pairs x {n_configs} configurations = {len(done) * n_configs} runs',
        60 * sum(d['training_minutes'] + d['loading_minutes'] for d in done),
        ', '.join(sorted({str(d.get('gpu')) for d in done})), f'data_processed/training/{grid}/*/*/done.json',
        training_seconds=60 * sum(d['training_minutes'] for d in done))

# Compute records (NB05, NB07, NB08, NB10, NB11, NB14)
for path in sorted(glob.glob('results/compute/*.json')):
    r = json.load(open(path))
    hardware = r['gpu'] or f'CPU ({r.get("cpu_model") or "model not recorded"}, {r.get("threads", r["cpu_cores"])} threads)'
    add(r['notebook'], r['step'], r['seconds'], hardware, path, proteins=r.get('proteins'), st_sites=r.get('st_sites'),
        date=r['date'])

# NB06: LM-OGlcNAc-Site on our test set
path = 'data_processed/other_tools/LM-OGlcNAc-Site/compute_test.json'
if os.path.exists(path):
    c = json.load(open(path))
    assert c['embedding_chunks_timed'] == c['chunks'], 'LM-OGlcNAc-Site test run: not every chunk timed'
    add('NB06', 'LM-OGlcNAc-Site, test set (three pLMs + networks)', c['embedding_seconds'] + c['network_seconds'],
        ', '.join(c['embedding_gpus']), path, windows=c['windows'], st_sites=c['sites'])

# LM-OGlcNAc-Site on the external sets (lab server): final progress line "[hh:mm:ss<00:00"
log = open('data_processed/other_tools/LM-OGlcNAc-Site/external_run.log', encoding='utf-8', errors='replace').read()
assert log.count('Versions:') == 1, 'external run log: more than one run (restarted); the progress time covers one run only'
final = re.findall(r'(\d+)/(\d+) \[(\d+):(\d\d):(\d\d)<00:00', log)[-1]
assert final[0] == final[1], 'external run log: last progress line is not complete'
prov = json.load(open('data_processed/other_tools/LM-OGlcNAc-Site/external_provenance.json'))
h, m, s = map(int, final[2:])
add('LM script', 'LM-OGlcNAc-Site, external sets (three pLMs + networks)', 3600 * h + 60 * m + s, prov['gpu'],
    'external_run.log (final progress line), external_provenance.json', windows=int(final[1]),
    date=prov['finished'][:10])

summary = pd.DataFrame(rows)
summary['hours'] = summary['seconds'] / 3600
summary.to_csv(f'{TABLE_DIR}/compute_summary.csv', index=False)
with pd.option_context('display.width', 250, 'display.max_colwidth', 70):
    print(summary[['notebook', 'step', 'hours', 'hardware']].to_string(index=False, float_format='{:.2f}'.format))
missing = [nb for nb in ('NB05', 'NB06', 'NB07', 'NB08', 'NB10', 'NB11', 'NB14') if nb not in set(summary['notebook'])]
print(f'\nNotebooks without a compute record yet: {", ".join(missing) or "none"}')
print(f'Saved: {TABLE_DIR}/compute_summary.csv')

# %%
import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------
# Species figure (figure number set with the manuscript): AUROC of the final model per species
#   (A) Test set: species with >= 20 positive test sites (NB05, test_metrics_by_species.csv).
#   (B) External studies, pooled: all proteins, and proteins with the same accession or sequence as,
#   or identity >= 30% to, a training or validation protein removed (NB12, metrics_by_species.csv).
#   Columns at the right: positive sites (proteins) in the training set and in the evaluated set.
#   Plotting only; all values as saved by NB05 and NB12.
#   Frontiers figure rules: 180 mm wide, text >= 8 pt, panels labelled (A)-(B), no red or green;
#   PDF and RGB TIFF at 300 dpi.
# ---------------------------------------------------------------------------
FIG_DIR = 'results/figures/final_model'
os.makedirs(FIG_DIR, exist_ok=True)
test_sp = pd.read_csv('data_processed/final_model/test_metrics_by_species.csv')
ext_sp = pd.read_csv('data_processed/external_validation/metrics_by_species.csv')
test_sp = test_sp[test_sp['drawn']].sort_values('positive').reset_index(drop=True)   # most positive sites at the top
# subset in the CSV -> (legend label, colour, marker, vertical offset)
SUBSETS = {'all proteins': ('All proteins', '#8fb8ea', 'o', 0.2),
           'homologs (≥ 30% identity) removed': ('Proteins with ≥ 30% identity to training or validation proteins removed',
                                                 '#1f5fa8', 's', -0.2)}
assert set(ext_sp['subset']) == set(SUBSETS)
ext_order = ext_sp[ext_sp['subset'] == 'all proteins'].sort_values('positive')['organism'].tolist()
OUR_COLOUR = '#2a78d6'

def species_label(organism):
    """Genus and species in italics; strain, if any, in roman. Asterisk on S. cerevisiae: its Atlas sites
    come from yeast expressing human OGT (PMID 33229814), explained in the caption."""
    genus, species, *strain = organism.split(' ')
    return (f'$\\it{{{genus}~{species}}}$' + (' ' + ' '.join(strain) if strain else '')
            + ('*' if (genus, species) == ('Saccharomyces', 'cerevisiae') else ''))

# Layout in mm: species names, plot, two columns of counts, training then evaluated set (right-aligned at COL1 and COL2)
W, LEFT, WIDTH, COL1, COL2 = 180, 53, 60, 140, 178
ROW_A, ROW_B, TOP, XLAB, HEAD, LEG = 7, 11, 12, 13, 12, 8
h_a, h_b = len(test_sp) * ROW_A, len(ext_order) * ROW_B
H = TOP + h_a + XLAB + HEAD + h_b + XLAB + LEG
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig = plt.figure(figsize=(W / 25.4, H / 25.4))
ax_a = fig.add_axes([LEFT / W, (H - TOP - h_a) / H, WIDTH / W, h_a / H])
ax_b = fig.add_axes([LEFT / W, (LEG + XLAB) / H, WIDTH / W, h_b / H])

def header(ax, label, columns):
    """Panel label at the left edge of the figure and bold column headers, above the plot."""
    for x, text, ha, weight in [(2, label, 'left', 'normal')] + [(c, t, 'right', 'bold') for c, t in zip((COL1, COL2), columns)]:
        ax.annotate(text, (x / W, 1), xycoords=('figure fraction', 'axes fraction'), xytext=(0, 4),
                    textcoords='offset points', ha=ha, va='bottom', fontsize=8, fontweight=weight)

def cell(ax, x, y, text):
    ax.annotate(text, (x / W, y), xycoords=('figure fraction', 'data'), ha='right', va='center', fontsize=8)

# (A) Test set
ypos = np.arange(len(test_sp))
ax_a.hlines(ypos, 0.5, test_sp['AUROC'], color=OUR_COLOUR, linewidth=1.5)
ax_a.scatter(test_sp['AUROC'], ypos, color=OUR_COLOUR, s=22, zorder=3)
for yp, (_, r) in zip(ypos, test_sp.iterrows()):
    ax_a.annotate(f'{r["AUROC"]:.3f}', (r['AUROC'], yp), xytext=(5, 0), textcoords='offset points', va='center', fontsize=8)
    cell(ax_a, COL1, yp, f'{r["training positive sites"]:,} ({r["training proteins"]:,})')
    cell(ax_a, COL2, yp, f'{r["positive"]:,} ({r["proteins"]:,})')
header(ax_a, '(A) Test set', ['Training positive sites\n(proteins)', 'Test positive sites\n(proteins)'])
ax_a.set_yticks(ypos, [species_label(o) for o in test_sp['organism']])

# (B) External studies: both subsets per species, one blue hue (light to dark) and two marker shapes
ypos = {o: k for k, o in enumerate(ext_order)}
for subset, (label, colour, marker, offset) in SUBSETS.items():
    d = ext_sp[ext_sp['subset'] == subset].set_index('organism').loc[ext_order]
    y = np.array([ypos[o] for o in ext_order]) + offset
    ax_b.scatter(d['AUROC'], y, color=colour, marker=marker, s=24, zorder=3, label=label)
    for yy, (_, r) in zip(y, d.iterrows()):
        ax_b.annotate(f'{r["AUROC"]:.3f}', (r['AUROC'], yy), xytext=(5, 0), textcoords='offset points', va='center', fontsize=8)
        cell(ax_b, COL2, yy, f'{r["positive"]:,} ({r["proteins"]:,})')
for o in ext_order:
    r = ext_sp[(ext_sp['organism'] == o) & (ext_sp['subset'] == 'all proteins')].iloc[0]
    cell(ax_b, COL1, ypos[o], f'{r["training positive sites"]:,} ({r["training proteins"]:,})')
header(ax_b, '(B) External studies', ['Training positive sites\n(proteins)', 'External positive sites\n(proteins)'])
ax_b.set_yticks(list(ypos.values()), [species_label(o) for o in ext_order])
fig.legend([test_line, Line2D([], [], color=GREY, linewidth=1)],
           ['(A) O-GlcNAcFinder, test set (bars: 95% confidence interval)', 'Perfect calibration'],
           loc='outside lower center', ncols=2, frameon=False)

for ax, n in [(ax_a, len(test_sp)), (ax_b, len(ext_order))]:
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(-0.6, n - 0.4)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel('AUROC (0.5 = no skill)')

fig.savefig(f'{FIG_DIR}/figure_species_AUROC.pdf')
buffer = io.BytesIO()
fig.savefig(buffer, format='png', dpi=300)
Image.open(buffer).convert('RGB').save(f'{FIG_DIR}/figure_species_AUROC.tif', dpi=(300, 300))
plt.show()

# %%
import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------
# Top-k figure (figure number set with the manuscript): hit rate of the final model within proteins
#   Share of proteins with >= 1 positive site among their k highest ranked S/T, k = 1-20, for
#   O-GlcNAcFinder and for random choice (hypergeometric expectation); topk_hit_rate.csv (NB12).
#   (A) Test set (held-out split, < 30% identity to training and validation).
#   (B) External studies pooled, proteins with >= 30% identity to training or validation removed.
#   Values at K_MAIN written next to the points. Unlabelled S/T are not proven negatives, so hit
#   rates are lower bounds. Plotting only; all values as saved by NB12.
#   Frontiers figure rules: 180 mm wide, text >= 8 pt, panels labelled (A)-(B), no red or green;
#   PDF and RGB TIFF at 300 dpi.
# ---------------------------------------------------------------------------
FIG_DIR = 'results/figures/external_validation'
os.makedirs(FIG_DIR, exist_ok=True)
K_VALUES, K_MAIN = np.arange(1, 21), 10
OUR_COLOUR, RANDOM_COLOUR = '#2a78d6', '#8a8984'
topk = pd.read_csv('data_processed/external_validation/topk_hit_rate.csv')
# (unit, subset) in the CSV -> panel title
PANELS = {('Test set', 'held-out split (< 30% identity)'): '(A) Test set',
          ('All studies pooled', 'homologs (>= 30% identity) removed'): '(B) External studies, pooled'}

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 70 / 25.4), sharey=True, layout='constrained')
for ax, ((unit, subset), title) in zip(axes, PANELS.items()):
    r = topk[(topk['unit'] == unit) & (topk['subset'] == subset)]
    assert len(r) == 1, (unit, subset)
    r = r.iloc[0]
    # (column prefix, colour, line name, label above or below the line, k where the name is placed)
    for prefix, colour, name, above, k_label in [('hit', OUR_COLOUR, 'O-GlcNAcFinder', True, 17),
                                                 ('random', RANDOM_COLOUR, 'Random choice', False, 13)]:
        y = 100 * np.array([r[f'{prefix}@{k}'] for k in K_VALUES])
        ax.plot(K_VALUES, y, color=colour, linewidth=1.5, marker='o', markersize=3.5)
        # K_MAIN: larger point with a dark edge and its value in bold
        ax.plot(K_MAIN, y[K_MAIN - 1], marker='o', markersize=7, color=colour, markeredgecolor='#0b0b0b',
                markeredgewidth=1, zorder=4)
        dy, va = (8, 'bottom') if above else (-8, 'top')
        ax.annotate(f'{y[K_MAIN - 1]:.1f}%', (K_MAIN, y[K_MAIN - 1]), xytext=(0, dy), textcoords='offset points',
                    ha='center', va=va, fontsize=8, fontweight='bold')
        # line name on the side where the rising line moves away from the text
        ax.annotate(name, (k_label, y[k_label - 1]), xytext=(0, dy), textcoords='offset points',
                    ha='right' if above else 'left', va=va, fontsize=8)
    ax.set_title(f'{title}\n{int(r["proteins"]):,} proteins', loc='left')
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel('k (highest ranked serines and threonines)')
axes[0].set_ylabel('Proteins with a true site\namong the top k (%)')

fig.savefig(f'{FIG_DIR}/figure_topk_ours_alone.pdf')
buffer = io.BytesIO()
fig.savefig(buffer, format='png', dpi=300)
Image.open(buffer).convert('RGB').save(f'{FIG_DIR}/figure_topk_ours_alone.tif', dpi=(300, 300))
plt.show()

# %%
import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

# ---------------------------------------------------------------------------
# Top-k figure, each existing predictor vs O-GlcNAcFinder (figure number set with the manuscript)
#   Hit rate against k (k = 1-20): share of proteins with >= 1 positive site among their k highest
#   ranked S/T, for the predictor, O-GlcNAcFinder and random choice, both models ranked on the same
#   proteins and sites. One row per set, one column per predictor:
#     (A) Test set, proteins with < 30% identity to the training data of both models
#         (topk_hit_rate_tools.csv, set 'Test set'; NB12).
#     (B) External studies pooled, every protein (topk_hit_rate_tools_all_proteins.csv; NB12).
#     (C) External studies pooled, proteins with < 30% identity to the training data of both models
#         (topk_hit_rate_tools.csv, set 'All studies pooled'; NB12).
#   Lines without markers (small panels); K_MAIN: points with a dark edge; values in bold, coloured as
#   their line (darker shade), the higher one above its point.
#   Unlabelled S/T are not proven negatives, so hit rates are lower bounds. Plotting only.
#   Frontiers figure rules: 180 mm wide, text >= 8 pt, panels labelled (A)-(C), no red or green;
#   PDF and RGB TIFF at 300 dpi.
# ---------------------------------------------------------------------------
FIG_DIR = 'results/figures/external_validation'
os.makedirs(FIG_DIR, exist_ok=True)
K_VALUES, K_MAIN = np.arange(1, 21), 10
OUR_COLOUR, TOOL_COLOUR, RANDOM_COLOUR = '#2a78d6', '#8a8984', '#4a4945'
TEXT_COLOUR = {'ours': '#1c5cab', 'tool': '#5f5e5a'}   # value labels: darker shades, for contrast on white
TD = 'data_processed/external_validation'
pairwise = pd.read_csv(f'{TD}/topk_hit_rate_tools.csv')
every = pd.read_csv(f'{TD}/topk_hit_rate_tools_all_proteins.csv')
ROWS = [('(A) Test set, proteins with less than 30% identity to the training proteins of both models',
         pairwise[pairwise['set'] == 'Test set']),
        ('(B) External studies, pooled, all proteins', every[every['set'] == 'All studies pooled']),
        ('(C) External studies, pooled, proteins with less than 30% identity to the training proteins of both models',
         pairwise[pairwise['set'] == 'All studies pooled'])]
# Tool names as written in the manuscript (display only; the CSV keeps the notebook names)
TOOLS = {'LM-OGlcNAc-Site': 'LM-OGlcNAc-Site', 'O-GlcNAcPRED-DL': 'O-GlcNAcPRED-DL',
         'Stack-OglyPred-PLM': 'Stack-OglyPred-\nPLM', 'RNN-[600,75] (Seber)': 'RNN-[600,75]',
         'RNN-225 (Seber and Braatz)': 'RNN-225', 'BSB 2025 (Arcanjo et al.)': 'Arcanjo et al.\n(2025)'}

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False})
fig = plt.figure(figsize=(180 / 25.4, 150 / 25.4), layout='constrained')
subfigs = fig.subfigures(len(ROWS), 1)
for row, (subfig, (heading, data)) in enumerate(zip(subfigs, ROWS)):
    data = data.set_index('tool')
    assert set(TOOLS) <= set(data.index), heading
    subfig.suptitle(heading, x=0.01, ha='left', fontsize=8)
    axes = subfig.subplots(1, len(TOOLS), sharey=True)
    for col, (ax, (tool, name)) in enumerate(zip(axes, TOOLS.items())):
        r = data.loc[tool]
        curves = {who: 100 * np.array([r[f'{who}@{k}'] for k in K_VALUES]) for who in ('tool', 'ours', 'random')}
        ax.plot(K_VALUES, curves['random'], color=RANDOM_COLOUR, linewidth=1, linestyle=(0, (1, 1.5)))
        for who, colour in [('tool', TOOL_COLOUR), ('ours', OUR_COLOUR)]:
            ax.plot(K_VALUES, curves[who], color=colour, linewidth=1.2)
            ax.plot(K_MAIN, curves[who][K_MAIN - 1], marker='o', markersize=5, color=colour,
                    markeredgecolor='#0b0b0b', markeredgewidth=0.8, zorder=4)
        # values at K_MAIN: the higher one above its point, the lower one below; text in a darker shade of
        # its line's colour, so each value can be matched to its point when the two points overlap
        at_main = {who: curves[who][K_MAIN - 1] for who in ('tool', 'ours')}
        high = max(at_main, key=at_main.get)
        for who, v in at_main.items():
            dy, va = (5, 'bottom') if who == high else (-5, 'top')
            ax.annotate(f'{v:.1f}', (K_MAIN, v), xytext=(0, dy), textcoords='offset points', ha='center', va=va,
                        fontsize=8, fontweight='bold', color=TEXT_COLOUR[who])
        ax.text(0.03, 0.97, f'{int(r["proteins"]):,} proteins', transform=ax.transAxes, ha='left', va='top', fontsize=8)
        if row == 0:
            ax.set_title(name, fontsize=8)
        ax.set_xticks([1, 10, 20])
        ax.set_xlim(0.5, 20.5)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 50, 100])
        if row == len(ROWS) - 1:
            ax.set_xlabel('k')
    axes[0].set_ylabel('Hit rate (%)')

handles = [Line2D([], [], color=OUR_COLOUR, linewidth=1.2, label='O-GlcNAcFinder'),
           Line2D([], [], color=TOOL_COLOUR, linewidth=1.2, label='Other tool'),
           Line2D([], [], color=RANDOM_COLOUR, linewidth=1, linestyle=(0, (1, 1.5)), label='Random choice')]
fig.legend(handles=handles, loc='outside lower center', ncols=3, frameon=False)

fig.savefig(f'{FIG_DIR}/figure_topk_ours_vs_tools.pdf')
buffer = io.BytesIO()
fig.savefig(buffer, format='png', dpi=300)
Image.open(buffer).convert('RGB').save(f'{FIG_DIR}/figure_topk_ours_vs_tools.tif', dpi=(300, 300))
plt.show()

# %%
import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------------
# Top-k figure, all models on the same proteins (figure number set with the manuscript)
#   Hit rate at K_MAIN (share of proteins with >= 1 positive site among their K_MAIN highest ranked
#   S/T) of O-GlcNAcFinder, the six predictors and random choice, every model ranked on the same
#   proteins and S/T: proteins with < 30% identity to the training data of every model
#   (topk_hit_rate_all_models.csv; NB12).
#     (A) Test set (S/T scored by every predictor).  (B) External studies, pooled.
#   Random choice drawn as its own row (no reference line). Unlabelled S/T are not proven negatives,
#   so hit rates are lower bounds. Plotting only; all values as saved by NB12.
#   Frontiers figure rules: 180 mm wide, text >= 8 pt, panels labelled (A)-(B), no red or green;
#   PDF and RGB TIFF at 300 dpi.
# ---------------------------------------------------------------------------
FIG_DIR = 'results/figures/external_validation'
os.makedirs(FIG_DIR, exist_ok=True)
K_MAIN = 10
OUR_COLOUR, TOOL_COLOUR, RANDOM_COLOUR = '#2a78d6', '#8a8984', '#4a4945'
topk_all = pd.read_csv('data_processed/external_validation/topk_hit_rate_all_models.csv')
PANELS = {'Test set': '(A) Test set', 'All studies pooled': '(B) External studies, pooled'}
# Model names as written in the manuscript (display only; the CSV keeps the notebook names), in figure order
MODELS = {'Our model': 'O-GlcNAcFinder', 'LM-OGlcNAc-Site': 'LM-OGlcNAc-Site', 'O-GlcNAcPRED-DL': 'O-GlcNAcPRED-DL',
          'Stack-OglyPred-PLM': 'Stack-OglyPred-PLM', 'RNN-[600,75] (Seber)': 'RNN-[600,75]',
          'RNN-225 (Seber and Braatz)': 'RNN-225', 'BSB 2025 (Arcanjo et al.)': 'Arcanjo et al. (2025)',
          'Random choice': 'Random choice'}
COLOURS = {m: OUR_COLOUR if m == 'Our model' else RANDOM_COLOUR if m == 'Random choice' else TOOL_COLOUR for m in MODELS}

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 72 / 25.4), sharey=True, layout='constrained')
ypos = np.arange(len(MODELS))[::-1]
for ax, (set_name, title) in zip(axes, PANELS.items()):
    v = topk_all[topk_all['set'] == set_name].set_index('model')
    assert set(MODELS) <= set(v.index), set_name
    assert v['proteins'].nunique() == 1   # every model on the same proteins
    vals = 100 * v.loc[list(MODELS), f'hit@{K_MAIN}'].to_numpy()
    colours = [COLOURS[m] for m in MODELS]
    ax.hlines(ypos, 0, vals, color=colours, linewidth=1.5)
    ax.scatter(vals, ypos, color=colours, s=22, zorder=3)
    best = max(round(val, 1) for m, val in zip(MODELS, vals) if m != 'Random choice')   # best model, in bold
    for yp, val in zip(ypos, vals):
        ax.annotate(f'{val:.1f}', (val, yp), xytext=(5, 0), textcoords='offset points', va='center', fontsize=8,
                    fontweight='bold' if round(val, 1) == best else 'normal')
    ax.set_title(f'{title}\n{int(v["proteins"].iloc[0]):,} proteins', loc='left')
    ax.set_xlim(0, 100)
    ax.set_yticks(ypos, list(MODELS.values()))
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel(f'Proteins with a true site among the top {K_MAIN} (%)')

fig.savefig(f'{FIG_DIR}/figure_topk{K_MAIN}_all_models.pdf')
buffer = io.BytesIO()
fig.savefig(buffer, format='png', dpi=300)
Image.open(buffer).convert('RGB').save(f'{FIG_DIR}/figure_topk{K_MAIN}_all_models.tif', dpi=(300, 300))
plt.show()

# %%
import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from statsmodels.stats.proportion import proportion_confint

# ---------------------------------------------------------------------------
# Supplementary figure: calibration of the final model
#   Residues of a set are sorted by calibrated probability and split into N_BINS bins of equal size;
#   each bin gives its mean probability and the observed fraction of positives (95% Wilson interval).
#   (A) Test set, log scales, with the intervals.
#   (B) All unmasked residues of our O-GlcNAcAtlas 5.0 dataset, by set (training, validation, test),
#   0-ZOOM on both axes (every bin lies below it). Validation residues were used to fit the Platt
#   scaling, so their agreement with the diagonal is expected.
#   Grey line: perfect calibration. Negatives are unverified, so the fractions are lower bounds.
#   Input: atlas_site_scores.parquet (NB14 cell 3; its test rows match the NB05 test scores).
#   Plotting only; nothing is rescored.
#   Frontiers figure rules: 180 mm wide, text >= 8 pt, panels labelled (A)-(B), no red or green;
#   PDF and RGB TIFF at 300 dpi.
# ---------------------------------------------------------------------------
FINAL_DIR = 'data_processed/final_model'
FIG_DIR = 'results/figures/final_model'
os.makedirs(FIG_DIR, exist_ok=True)
BLUE, GREY, LIGHT, INK = '#2a78d6', '#8a8984', '#a9a8a2', '#4a4945'
N_BINS = 10
ZOOM = 0.15
SETS = {'train': ('Training', LIGHT), 'val': ('Validation', INK), 'test': ('Test', BLUE)}

def equal_count_bins(p, y, n_bins):
    """Residues sorted by probability, split into n_bins bins of equal size: mean probability,
    observed fraction of positives and its 95% Wilson interval."""
    order = np.argsort(p, kind='stable')
    rows = []
    for idx in np.array_split(order, n_bins):
        k, n = int(y[idx].sum()), len(idx)
        low, high = proportion_confint(k, n, alpha=0.05, method='wilson')
        rows.append({'sites': n, 'positives': k, 'mean_probability': p[idx].mean(), 'observed_fraction': k / n,
                     'ci_low': low, 'ci_high': high})
    return pd.DataFrame(rows)

scores = pd.read_parquet(f'{FINAL_DIR}/atlas_site_scores.parquet')
bins = {split: equal_count_bins(d['probability'].to_numpy(), d['positive'].to_numpy(int), N_BINS)
        for split, d in scores.groupby('split')}
for split, (label, _) in SETS.items():
    d = scores[scores['split'] == split]
    print(f'{label}: {len(d):,} S/T, {int(d["positive"].sum()):,} positive; mean probability '
          f'{d["probability"].mean():.4f}, fraction positive {d["positive"].mean():.4f}')
assert all((b['mean_probability'].max() < ZOOM) and (b['observed_fraction'].max() < ZOOM) for b in bins.values())

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.spines.top': False, 'axes.spines.right': False})
fig, (ax_rel, ax_all) = plt.subplots(1, 2, figsize=(180 / 25.4, 100 / 25.4), layout='constrained')

rel = bins['test']
low_lim = 0.5 * min(rel['ci_low'].min(), rel['mean_probability'].min())
high_lim = 2 * max(rel['ci_high'].max(), rel['mean_probability'].max())
ax_rel.plot([low_lim, high_lim], [low_lim, high_lim], color=GREY, linewidth=1)
test_line = ax_rel.errorbar(rel['mean_probability'], rel['observed_fraction'],
                            yerr=[rel['observed_fraction'] - rel['ci_low'], rel['ci_high'] - rel['observed_fraction']],
                            color=BLUE, marker='o', markersize=4, linewidth=1.5, elinewidth=1, capsize=0)
ax_rel.set(xscale='log', yscale='log', xlim=(low_lim, high_lim), ylim=(low_lim, high_lim),
           xlabel='Mean calibrated probability (log scale)', ylabel='Observed fraction positive (log scale)')
ax_rel.set_title(f'(A) Test set, {N_BINS} bins', loc='left')

ax_all.plot([0, ZOOM], [0, ZOOM], color=GREY, linewidth=1, zorder=0)
for split, (label, colour) in SETS.items():
    b = bins[split]
    ax_all.plot(b['mean_probability'], b['observed_fraction'], linestyle='none', marker='o', markersize=4, color=colour,
                label=f'{label}\n{b["sites"].sum():,} S/T\n{b["positives"].sum():,} positive')
ax_all.set(xlim=(0, ZOOM), ylim=(0, ZOOM), xlabel='Mean calibrated probability', ylabel='Fraction labelled positive')
ax_all.set_title(f'(B) All residues of our dataset, {N_BINS} bins per set', loc='left')

for ax in (ax_rel, ax_all):
    ax.set_aspect('equal')
# set legend in the empty upper left of (B): every bin with probability < 0.1 has a fraction < 0.05
ax_all.legend(frameon=False, loc='upper left')
fig.legend([test_line, Line2D([], [], color=GREY, linewidth=1)],
           ['(A) O-GlcNAcFinder, test set (bars: 95% confidence interval)', 'Perfect calibration'],
           loc='outside lower center', ncols=2, frameon=False)

fig.savefig(f'{FIG_DIR}/figure_supplementary_calibration.pdf')
buffer = io.BytesIO()
fig.savefig(buffer, format='png', dpi=300)
Image.open(buffer).convert('RGB').save(f'{FIG_DIR}/figure_supplementary_calibration.tif', dpi=(300, 300))
plt.show()
