"""Supplementary Table S6: per-site and per-protein results of O-GlcNAcFinder on the five external studies.

Source: data_processed/external_validation/external_validation_details.xlsx (NB12 cell 3), unchanged values.
Layout only: studies named by first author and year (order of Figure 4), internal protein IDs dropped (proteins
identified by UniProt accessions), readable column names, True/False as Yes/No, and a sheet defining the columns.
Output: ../manuscript/supplementary/Supplementary_Table_S6.xlsx
"""
import pandas as pd

SRC = '../data_processed/external_validation/external_validation_details.xlsx'
OUT = '../manuscript/supplementary/Supplementary_Table_S6.xlsx'
STUDIES = {'Lei 2025': 'Lei et al. (2025)', 'Mouse brain 2025': 'Hou et al. (2025)',
           'HCC liver 2026': 'Hou et al. (2026)', 'NOTISE 2026': 'Griffin et al. (2026)', 'Fu 2026': 'Fu et al. (2026)'}
OVERLAP = {'accessions': 'UniProt accessions',
           'max_identity': 'Highest identity to a training or validation protein',
           'closest_seen_protein': 'Closest training or validation protein',
           'same_protein': 'Same protein as a training or validation protein',
           'homolog_30': '>= 30% identity to a training or validation protein'}
SITE_COLS = {'study': 'Study', 'position': 'Position', 'residue': 'Residue', 'probability': 'Calibrated probability',
             'called': 'Called positive', 'rank_in_protein': 'Rank in protein', 'st_in_protein': 'S/T in protein',
             **OVERLAP}
PROTEIN_COLS = {'study': 'Study', 'length': 'Length', 'st_scored': 'S/T scored', 'positive': 'Positive sites',
                'called': 'Called positive', 'true_positive': 'True positives', 'false_negative': 'False negatives',
                'false_positive': 'False positives', 'recall': 'Recall', 'precision': 'Precision',
                'protein_AUROC': 'AUROC of the protein', **OVERLAP}
DEFINITIONS = [
    ('Study', 'External study, by first author and year (cited in the main text).'),
    ('UniProt accessions', 'UniProt entries of the protein sequence as reported by the study.'),
    ('Position, Residue', 'Position and residue (S or T) of a site resolved by the study.'),
    ('Calibrated probability', 'Probability of O-GlcNAcylation given by O-GlcNAcFinder.'),
    ('Called positive', 'Positive sites sheet: whether the site is at or above the cut-off (0.082). Proteins '
                        'sheet: number of residues of the protein at or above the cut-off.'),
    ('Rank in protein', 'Rank of the site among the scored serines and threonines of its protein (1 = highest '
                        'probability; tied residues share the best of their ranks).'),
    ('S/T in protein, S/T scored', 'Serines and threonines of the protein scored by O-GlcNAcFinder (masked '
                                   'residues excluded).'),
    ('Positive sites', 'Sites resolved by the study in the protein.'),
    ('True positives, False negatives', 'Positive sites called positive, and not called positive.'),
    ('False positives', 'Residues called positive with no site reported by the study. They are not proven '
                        'negatives, since a site may not have been detected.'),
    ('Recall, Precision', 'True positives / positive sites, and true positives / residues called positive (empty '
                          'when no residue is called).'),
    ('AUROC of the protein', 'AUROC over the scored residues of the protein (empty when the protein has no '
                             'resolved site).'),
    ('Highest identity to a training or validation protein', 'Highest sequence identity (identical residues / '
                                                             'length of the shorter protein) to any training or '
                                                             'validation protein of O-GlcNAcFinder, from the '
                                                             'MMseqs2 alignments (0 when no alignment was found).'),
    ('Closest training or validation protein', 'The training or validation protein with that identity (empty when '
                                               'no alignment was found).'),
    ('Same protein as a training or validation protein', 'Yes when the protein is a training or validation protein.'),
    ('>= 30% identity to a training or validation protein', 'Yes when the highest identity is 30% or more.'),
]

src = pd.read_excel(SRC, sheet_name=None)
assert set(src['positive sites']['study']) == set(STUDIES) and set(src['proteins']['study']) == set(STUDIES)

def tidy(df, cols, sort):
    out = df[list(cols)].rename(columns=cols).copy()
    out['Study'] = out['Study'].map(STUDIES)
    for c in out.columns:
        if out[c].dtype == bool:
            out[c] = out[c].map({True: 'Yes', False: 'No'})
    out['order'] = out['Study'].map(list(STUDIES.values()).index)
    return out.sort_values(['order'] + sort).drop(columns='order')

sites = tidy(src['positive sites'], SITE_COLS, ['UniProt accessions', 'Position'])
proteins = tidy(src['proteins'], PROTEIN_COLS, ['UniProt accessions'])
assert len(sites) == 7009 and len(proteins) == 3458
assert proteins.groupby('Study')['Positive sites'].sum().to_dict() == sites['Study'].value_counts().to_dict()

with pd.ExcelWriter(OUT, engine='openpyxl') as xl:
    for name, df in [('Positive sites', sites), ('Proteins', proteins)]:
        df.to_excel(xl, sheet_name=name, index=False)
        ws = xl.sheets[name]
        ws.freeze_panes = 'B2'
        for k, col in enumerate(df.columns, start=1):
            ws.column_dimensions[ws.cell(row=1, column=k).column_letter].width = min(max(12, len(col) + 2), 30)
    pd.DataFrame(DEFINITIONS, columns=['Column', 'Definition']).to_excel(xl, sheet_name='Columns', index=False)
    xl.sheets['Columns'].column_dimensions['A'].width = 52
    xl.sheets['Columns'].column_dimensions['B'].width = 110
print(sites.groupby('Study', sort=False).size().to_dict())
print(proteins.groupby('Study', sort=False).size().to_dict())
print(sites.head(2).to_string(index=False))
print(proteins.head(2).to_string(index=False))
