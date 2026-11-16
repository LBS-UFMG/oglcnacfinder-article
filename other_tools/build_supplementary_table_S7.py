"""Supplementary Table S7: metrics of O-GlcNAcFinder for each species, test set and external studies.

Sources (values unchanged): data_processed/final_model/test_metrics_by_species.csv (NB05 cell 6) and
data_processed/external_validation/metrics_by_species.csv (NB12 cell 12). MCC, precision and recall at the
cut-off (calibrated probability >= 0.082). Test set: every species, with whether it is drawn in Figure 6
(at least 20 positive sites). Output: ../manuscript/supplementary/Supplementary_Table_S7.xlsx
"""
import pandas as pd

OUT = '../manuscript/supplementary/Supplementary_Table_S7.xlsx'
COLS = {'organism': 'Species', 'proteins': 'Proteins', 'sites': 'S/T', 'positive': 'Positive sites',
        'positive rate': 'Positive rate (%)', 'AUROC': 'AUROC', 'AUPRC': 'AUPRC', 'MCC': 'MCC',
        'precision': 'Precision', 'recall': 'Recall', 'training proteins': 'Training proteins',
        'training positive sites': 'Training positive sites'}
SUBSETS = {'all proteins': 'All proteins',
           'homologs (≥ 30% identity) removed': 'Proteins with >= 30% identity to a training or validation protein removed'}

test = pd.read_csv('../data_processed/final_model/test_metrics_by_species.csv')
assert test['proteins'].sum() == 736 and test['positive'].sum() == 1929
test_out = test[list(COLS)].rename(columns=COLS)
test_out['Positive rate (%)'] = 100 * test_out['Positive rate (%)']
test_out.insert(len(COLS), 'In Figure 6', test['drawn'].map({True: 'Yes', False: 'No'}))
assert (test_out['In Figure 6'] == 'Yes').sum() == 6
# Yeast marked as in Figure 6: its Atlas 5.0 sites come from yeast expressing human OGT (Li et al. 2021)
YEAST = 'Saccharomyces cerevisiae S288C'
assert (test_out['Species'] == YEAST).sum() == 1
test_out['Species'] = test_out['Species'].replace({YEAST: YEAST + '*'})
YEAST_NOTE = ('* All Saccharomyces cerevisiae sites in the O-GlcNAcAtlas 5.0 come from one study of yeast expressing '
              'human OGT (Li et al., 2021), since S. cerevisiae has no OGT or OGA of its own (Halim et al., 2015; '
              'Nakanishi et al., 2017).')

ext = pd.read_csv('../data_processed/external_validation/metrics_by_species.csv', encoding='utf-8')
assert set(ext['subset']) == set(SUBSETS), set(ext['subset'])
ext_out = ext[list(COLS)].rename(columns=COLS)
ext_out.insert(1, 'Subset', ext['subset'].map(SUBSETS))
ext_out['Positive rate (%)'] = 100 * ext_out['Positive rate (%)']

with pd.ExcelWriter(OUT, engine='openpyxl') as xl:
    for name, df in [('Test set', test_out), ('External studies', ext_out)]:
        df.round(3).to_excel(xl, sheet_name=name, index=False)
        ws = xl.sheets[name]
        ws.freeze_panes = 'B2'
        for k, col in enumerate(df.columns, start=1):
            width = 32 if col == 'Species' else 60 if col == 'Subset' else max(10, len(col) + 2)
            ws.column_dimensions[ws.cell(row=1, column=k).column_letter].width = width
        if name == 'Test set':
            ws.cell(row=len(df) + 3, column=1, value=YEAST_NOTE)
for df in (test_out, ext_out):
    print(df.round(3).to_string(index=False).encode('ascii', 'replace').decode())
