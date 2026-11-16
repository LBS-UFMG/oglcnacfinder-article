"""Supplementary Table S3: every O-GlcNAc or glycosylation site predictor found, and why it was or was not
compared with O-GlcNAcFinder.

Rows: competitor_table.csv (46 tools). Display names, missing years/DOIs and the reason of every row were
checked against each paper and, where relevant, its server or code (October-November 2026); the checks of
2026-11-03/04 are recorded in supp_table_S3_checks.md. Each reason was approved by the authors.
Outputs: supplementary_table_S3.csv and ../manuscript/supplementary/Supplementary_Table_S3.xlsx.
"""
import pandas as pd

GROUPS = {'A': 'Dedicated O-GlcNAc predictor', 'B': 'Multi-PTM or generic glycosylation predictor',
          'C': 'Other modification or not a site predictor', 'D': 'Generic O-glycosylation, named in other papers'}
POOLED = 'Predicts a pooled O-linked glycosylation class, no O-GlcNAc-specific output.'
NO_HOST = ' Web server also not reachable (host not found).'

# Tool name in competitor_table.csv -> (display name, year, DOI, O-GlcNAc-specific output, compared, reason).
# None keeps the CSV value.
ROWS = {
    'Hong et al.': (None, None, None, 'Yes', 'No',
                    'No trained model, prediction script or web server released (training code and data only).'),
    'DeepO-GlcNAc': (None, None, None, 'Yes', 'No',
                     'Web server not reachable (October 2026), no code or trained model linked from the paper.'),
    'Khalid et al. (ESM-2)': ('Khalid et al.', None, None, 'Yes', 'No',
                              'No web server and no trained model released (data and training notebook only).'),
    'O-GlcNAcPRED-DL': (None, None, None, 'Yes', 'Yes', 'Compared: released human and mouse models.'),
    'Seber & Braatz (RNN-225)': ('RNN-225 (Seber and Braatz)', None, None, 'Yes', 'Yes',
                                 'Compared: released code and trained model.'),
    'Seber (RNN-[600,75])': ('RNN-[600,75] (Seber)', None, None, 'Yes', 'Yes',
                             'Compared: released code and trained model.'),
    'LM-OGlcNAc-Site': (None, None, None, 'Yes', 'Yes',
                        'Compared: released code and trained models (web server not reachable).'),
    'Mauri et al. models (OGP)': ('Mauri et al.', None, None, 'Yes', 'No',
                                  'Benchmark study, no predictor released (code repository not reachable).'),
    'O-GlcNAcPRED-II': (None, None, None, 'Yes', 'No', 'Web server not reachable, no trained model released.'),
    'OGTSite': (None, None, None, 'Yes', 'No', 'Web server not reachable, no trained model released.'),
    'PGlcS': (None, None, None, 'Yes', 'No', 'No web server, trained model or prediction code released.'),
    'Wu et al. (Lee group)': ('Wu et al.', None, None, 'Yes', 'No', 'No web server, code or trained model released.'),
    'O-GlcNAcPRED': (None, None, None, 'Yes', 'No', 'Web server not reachable, no trained model released.'),
    'O-GlcNAcScan (OGlcNAcScan)': ('O-GlcNAcScan', None, None, 'Yes', 'No',
                                   'Web server not reachable, no trained model released (no paper of its own, '
                                   'cited to dbOGAP).'),
    'YinOYang 1.2': (None, None, None, 'Yes', 'No',
                     'Web server submission fails (server configuration error), no code or trained model released.'),
    'OGTcomPred': (None, None, None, 'Yes', 'No',
                   'Released script requires disorder predictions from an unspecified predictor (SPOT-Disorder '
                   'assumed, downloads unavailable), so its input could not be produced.'),
    'Arcanjo et al. (2025)': (None, None, None, 'Yes', 'Yes', 'Compared: our earlier model, released code and model.'),
    'CLASPP': (None, None, None, 'No', 'No', POOLED),
    'MTPrompt-PTM': (None, None, None, 'No', 'No', POOLED),
    'Stack-OglyPred-PLM': (None, None, None, 'Yes', 'Yes',
                           'Compared: released trained model for nucleocytoplasmic O-GlcNAc.'),
    'Alkuhlani et al. (O-glycosylation PLM)': ('Alkuhlani et al.', None, None, 'No', 'No',
                                               'Predicts generic human O-glycosylation, no O-GlcNAc-specific output.'),
    'Captor': (None, None, None, 'No', 'No',
               'Predicts generic human O-glycosylation, no O-GlcNAc-specific output. No trained model released.'),
    'MusiteDeep': (None, None, None, 'No', 'No', POOLED),
    'SPRINT-Gly': (None, None, None, 'No', 'No', 'Predicts mucin-type O-GalNAc glycosylation, not O-GlcNAc.'),
    'PTM-ssMP': (None, None, None, 'Yes', 'No',
                 'Web server not reachable (host not found), no trained model or code released.'),
    'Multiple-kernel SVM': (None, None, None, 'Yes', 'No',
                            'No web server, code or trained model released (data only).'),
    'GlycoMine-struct': (None, None, None, 'No', 'No', POOLED + NO_HOST),
    'GlycoMine': (None, None, None, 'No', 'No', POOLED + NO_HOST),
    'GlycoEP': (None, None, None, 'No', 'No', POOLED + ' Web server also not reachable (redirect fails).'),
    'GPP': (None, None, None, 'No', 'No', POOLED + ' Web server prediction page under maintenance (November 2026).'),
    'EnsembleGly': (None, None, None, 'No', 'No', POOLED + NO_HOST),
    'GlycoMine_PU': (None, 2019, '10.1186/s12859-019-2700-1', 'No', 'No',
                     'Predicts pooled O-linked glycosylation classes (Ser and Thr), no O-GlcNAc-specific output.'
                     + NO_HOST),
    'iGlycoS-PseAAC': (None, None, '10.1109/tcbb.2020.2968441', 'No', 'No',
                       'Predicts generic O-linked glycosylation on serine, no O-GlcNAc-specific output.'),
    'O-GlyThr': (None, None, None, 'No', 'No',
                 'Predicts generic human O-linked glycosylation on threonine, no O-GlcNAc-specific output.'),
    'PTG-PLM': (None, None, None, 'No', 'No', 'Predicts N-glycosylation and lysine glycation only, no O-linked model.'),
    'Gana & Vasudevan': ('Gana and Vasudevan', None, None, 'No', 'No', 'Analysis study, no predictor released.'),
    'DAPPLE 2': (None, None, None, 'No', 'No', 'Transfers known sites between homologs, does not predict new sites.'),
    'NetOGlyc-4.0': (None, None, None, 'No', 'No',
                     'Predicts mucin-type O-GalNAc glycosylation. O-GlcNAc sites were deliberately excluded from its '
                     'training data.'),
    'GlycoPP': (None, None, None, 'No', 'No',
                'Predicts glycosylation of prokaryotic proteins only (pooled O-linked class), not eukaryotic O-GlcNAc.'),
    'CKSAAP_OGlySite': (None, None, None, 'No', 'No',
                        'Predicts mucin-type O-GalNAc glycosylation in mammalian proteins, not O-GlcNAc.'),
    'DeepGly': (None, None, None, 'No', 'No', 'Predicts lysine glycation, not glycosylation.'),
    'DictyOGlyc 1.1': (None, None, None, 'No', 'No',
                       'Predicts O-alpha-GlcNAc on secreted and membrane proteins of Dictyostelium discoideum, not '
                       'nucleocytoplasmic O-GlcNAc.'),
    'Yang & Han': ('Yang and Han', None, '10.1016/j.compbiolchem.2017.09.005', 'No', 'No',
                   'Predicts generic O-glycosylation (UniProt mammalian proteins), no O-GlcNAc-specific output.'),
    'Chen et al.': (None, None, '10.1007/s11517-015-1268-9', 'No', 'No',
                    'Predicts mucin-type O-glycosylation (Swiss-Prot), no O-GlcNAc-specific output.'),
    'Hassan et al.': (None, None, '10.4137/BBI.S26864', 'No', 'No', POOLED),
    'Sasaki et al. SVM tool': ('SVMGlyc (Sasaki et al.)', 2009, '10.2197/ipsjtbio.2.25', 'No', 'No',
                               'Predicts a pooled O-linked glycosylation class (Swiss-Prot annotations), no '
                               'O-GlcNAc-specific output.'),
}

survey = pd.read_csv('competitor_table.csv')
assert len(survey) == 46 and set(survey['Tool']) == set(ROWS), set(survey['Tool']) ^ set(ROWS)
rows = []
for _, r in survey.iterrows():
    name, year, doi, output, compared, reason = ROWS[r['Tool']]
    rows.append({'Tool': name or r['Tool'], 'Year': int(year or r['Year']),
                 'DOI': doi or r['DOI'], 'Group': GROUPS[r['Group'][0]],
                 'O-GlcNAc-specific output': output, 'Compared': compared, 'Reason': reason})
table = pd.DataFrame(rows)
table['order'] = table['Group'].map({g: k for k, g in enumerate(GROUPS.values())})
table = table.sort_values(['order', 'Year', 'Tool'], ascending=[True, False, True]).drop(columns='order')
assert table['DOI'].notna().all() and (table['Compared'] == 'Yes').sum() == 6

counts = table['Group'].value_counts()
print(counts.to_string())
print(f"Compared: {(table['Compared'] == 'Yes').sum()} | O-GlcNAc-specific output: "
      f"{(table['O-GlcNAc-specific output'] == 'Yes').sum()}")
table.to_csv('supplementary_table_S3.csv', index=False)
with pd.ExcelWriter('../manuscript/supplementary/Supplementary_Table_S3.xlsx', engine='openpyxl') as xl:
    table.to_excel(xl, sheet_name='Table S3', index=False)
    ws = xl.sheets['Table S3']
    for col, width in zip('ABCDEFG', (28, 6, 34, 44, 12, 10, 90)):
        ws.column_dimensions[col].width = width
    ws.freeze_panes = 'A2'
