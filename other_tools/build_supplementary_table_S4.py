"""Supplementary Table S4: cut-off of each model and the metrics at that cut-off.

Three sheets, every tool and O-GlcNAcFinder scored on the same residues within each row:
  Test set: proteins with < 30% identity to the training data of both models (filter a), from the
    per-tool test metric files of NB06-NB11 (data_processed/other_tools/<tool>/test_metrics_filtered*.csv).
  External, all proteins: all studies pooled, every model on every protein (NB12 cell 10, metrics_all_tools.csv).
  External, < 30% identity: all studies pooled, proteins with < 30% identity to the training data of both
    models (NB12 cell 11, metrics_pairwise_homologs_removed.csv).
Residues called positive on the test set and the < 30% external set = recall x positive sites / precision
(rounded; checked to be integers). Outputs: supplementary_table_S4.xlsx in ../manuscript/supplementary/.
"""
import pandas as pd
from openpyxl.styles import Font

D = '../data_processed'
OUT = '../manuscript/supplementary/Supplementary_Table_S4.xlsx'
OURS = 'O-GlcNAcFinder'
# Cut-off rules as applied in the benchmark notebooks (NB06-NB12)
CUTOFFS = {
    OURS: 'Calibrated probability >= 0.082 (maximum MCC on the validation set)',
    'LM-OGlcNAc-Site': 'At least 2 of its 3 networks above their own thresholds (Ankh 0.496, ESM-2 0.797, '
                       'ProtT5 0.669), as in its released code',
    'O-GlcNAcPRED-DL': 'Score >= 0.5',
    'Stack-OglyPred-PLM': 'Second output > 0.5, as in the authors\' notebook',
    'RNN-[600,75]': 'Score >= 0.60 (threshold reported by the authors)',
    'RNN-225': 'Score >= 0.60 (threshold reported by the authors)',
    'Arcanjo et al. (2025)': 'Score >= 0.5',
}
# Names in the notebooks' files -> names in the manuscript
NAMES = {'Our model': OURS, 'RNN-[600,75] (Seber)': 'RNN-[600,75]', 'RNN-225 (Seber and Braatz)': 'RNN-225',
         'BSB 2025 (Arcanjo et al.)': 'Arcanjo et al. (2025)'}
# (tool, metric file, text identifying the tool's row) as in NB15 cell 0
TEST_FILES = [('LM-OGlcNAc-Site', 'LM-OGlcNAc-Site/test_metrics_filtered_snapshot.csv', 'layer 33'),
              ('O-GlcNAcPRED-DL', 'O-GlcNAcPRED-DL/test_metrics_filtered_server.csv', 'matched'),
              ('RNN-[600,75]', 'Seber/test_metrics_filtered.csv', 'RNN-[600,75]'),
              ('RNN-225', 'Seber/test_metrics_filtered.csv', 'RNN-225'),
              ('Stack-OglyPred-PLM', 'Stack-OglyPred-PLM/test_metrics_filtered.csv', 'Stack'),
              ('Arcanjo et al. (2025)', 'BSB2025/test_metrics_filtered.csv', 'BSB')]
METRICS = ['MCC', 'precision', 'recall', 'called positive']

def called(r):
    """Residues called positive from recall, positives and precision."""
    n = r['recall'] * r['positive'] / r['precision']
    assert abs(n - round(n)) < 1e-6, n
    return int(round(n))

test_rows = []
for tool, file, key in TEST_FILES:
    df = pd.read_csv(f'{D}/other_tools/{file}')
    if 'organisms' in df.columns:
        df = df[df['organisms'] == 'all organisms']
    df = df[df['subset'].str.contains('homologs')]
    t = df[df['model'].str.contains(key, regex=False)]
    o = df[df['model'] == 'Our model']
    assert len(t) == 1 and len(o) == 1, tool
    t, o = t.iloc[0], o.iloc[0]
    row = {'Tool': tool, 'Tool cut-off': CUTOFFS[tool], 'Proteins': int(t['proteins']), 'S/T': int(t['sites']),
           'Positive sites': int(t['positive'])}
    for m in METRICS:
        for who, r in [('tool', t), (OURS, o)]:
            row[f'{m.capitalize() if m != "MCC" else m} {who}'] = called(r) if m == 'called positive' else r[m]
    test_rows.append(row)
test = pd.DataFrame(test_rows)

a = pd.read_csv(f'{D}/external_validation/metrics_all_tools.csv')
a = a[a['unit'] == 'All studies pooled'].copy()
a['Model'] = a['model'].replace(NAMES)
external_all = a.rename(columns={'proteins': 'Proteins', 'sites': 'S/T', 'positive': 'Positive sites',
                                 'precision': 'Precision', 'recall': 'Recall', 'called positive': 'Called positive'})
external_all['Cut-off'] = external_all['Model'].map(CUTOFFS)
external_all = external_all[['Model', 'Cut-off', 'Proteins', 'S/T', 'Positive sites', 'MCC', 'Precision', 'Recall',
                             'Called positive']]

p = pd.read_csv(f'{D}/external_validation/metrics_pairwise_homologs_removed.csv')
p = p[p['unit'] == 'All studies pooled'].copy()
p['Tool'] = p['tool'].replace(NAMES)
external_pair = pd.DataFrame({'Tool': p['Tool'], 'Tool cut-off': p['Tool'].map(CUTOFFS), 'Proteins': p['proteins'],
                              'S/T': p['sites'], 'Positive sites': p['positive']})
for m, col in [('MCC', 'MCC'), ('precision', 'Precision'), ('recall', 'Recall')]:
    external_pair[f'{col} tool'] = p[f'{m} tool'].to_numpy()
    external_pair[f'{col} {OURS}'] = p[f'{m} ours'].to_numpy()
for who in ('tool', 'ours'):
    external_pair[f'Called positive {"tool" if who == "tool" else OURS}'] = [
        called({'recall': r[f'recall {who}'], 'positive': r['positive'], 'precision': r[f'precision {who}']})
        for _, r in p.iterrows()]

order = list(CUTOFFS)
for df, col in [(test, 'Tool'), (external_all, 'Model'), (external_pair, 'Tool')]:
    assert df[col].isin(order).all() and df[col].notna().all()
    df.sort_values(col, key=lambda s: s.map(order.index), inplace=True)
assert external_all['Cut-off'].notna().all() and external_pair['Tool cut-off'].notna().all()

def bold_winners(ws, df, pairs):
    """Bold the higher MCC (3 decimals; both when tied): within each tool / O-GlcNAcFinder pair of a row
    (pairs=True), or the best row of the MCC column (pairs=False). Precision and recall are not marked:
    each alone favours a strict or a loose cut-off."""
    cols = list(df.columns)
    bold = Font(bold=True)
    for metric in ('MCC',):
        if pairs:
            a, b = cols.index(f'{metric} tool'), cols.index(f'{metric} {OURS}')
            for i, (x, y) in enumerate(zip(df.iloc[:, a].round(3), df.iloc[:, b].round(3)), start=2):
                for j, v in ((a, x), (b, y)):
                    if v == max(x, y):
                        ws.cell(row=i, column=j + 1).font = bold
        else:
            j = cols.index(metric)
            values = df.iloc[:, j].round(3)
            for i, v in enumerate(values, start=2):
                if v == values.max():
                    ws.cell(row=i, column=j + 1).font = bold

with pd.ExcelWriter(OUT, engine='openpyxl') as xl:
    for name, df, pairs in [('Test set', test, True), ('External, all proteins', external_all, False),
                            ('External, <30% identity', external_pair, True)]:
        df.round(3).to_excel(xl, sheet_name=name, index=False)
        ws = xl.sheets[name]
        ws.column_dimensions['A'].width = 24
        ws.column_dimensions['B'].width = 60
        ws.freeze_panes = 'C2'
        bold_winners(ws, df.reset_index(drop=True), pairs)
for name, df in [('Test set', test), ('External, all proteins', external_all), ('External, <30% identity', external_pair)]:
    print(f'##### {name}')
    print(df.drop(columns=[c for c in df.columns if 'cut-off' in c.lower()]).to_string(index=False,
                                                                                         float_format='{:.3f}'.format))
