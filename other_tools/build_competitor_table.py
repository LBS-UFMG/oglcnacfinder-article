"""Build the complete table of possible competitors (CSV for manual checking + Markdown view).

Every value comes from the survey outputs in this folder (survey_*.json, availability_*.json,
paper_notes_2026-10-06.md, lookup_dois.py output). 'not found' = searched but absent; 'not checked' = not searched yet.
"""
import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
CHECKED = '2026-10-06'

COLUMNS = ['#', 'Group', 'Tool', 'Year', 'Reference', 'DOI', 'Glycan target', 'Training data', 'Method',
           'Server URL (as printed)', f'Server status {CHECKED}', 'Code / data', f'Code status {CHECKED}',
           'Evidence', 'Notes', 'Manual check']

A = 'A. Dedicated O-GlcNAc'
B = 'B. Multi-PTM / generic glycosylation'
C = 'C. Off-target or not a predictor'
D = 'D. Named only (generic O-glycosylation), not looked up'

ROWS = [
    # --- A. dedicated O-GlcNAc site predictors ---
    (A, 'YinOYang 1.2', '2002', 'Gupta & Brunak, Pac Symp Biocomput 2002:310-322 (also Gupta, PhD thesis, DTU)', '10.1142/9789812799623_0029',
     'O-GlcNAc', 'not recorded', 'neural network', 'https://services.healthtech.dtu.dk/services/YinOYang-1.2/', 'page up; submission fails (WebfaceConfigError, 2026-10-12)',
     'standalone package (Linux, SunOS, OSF1, IRIX) via DTU academic request form, academic licence; no source code', 'download links checked 2026-10-12',
     'abstract + live check', 'NOT BENCHMARKED (server error, no code; dropped 2026-10-12); comparator in O-GlcNAcPRED-II, O-GlcNAcPRED-DL, DeepO-GlcNAc, Mauri 2021'),
    (A, 'O-GlcNAcScan (OGlcNAcScan)', '2011', 'no paper of its own; always cited to dbOGAP: Wang et al., BMC Bioinformatics 12:91', '10.1186/1471-2105-12-91',
     'O-GlcNAc', 'dbOGAP', 'SVM', 'not found', '-', 'not found', '-',
     'O-GlcNAcPRED-II, O-GlcNAcPRED-DL, DeepO-GlcNAc PDFs', 'NOT BENCHMARKED (no reachable server and no released trained model; decided 2026-10-13, from the 2026-10-06 checks); O-GlcNAcPRED-DL (2024) could not access it'),
    (A, 'O-GlcNAcPRED', '2013', 'Jia et al., Mol BioSyst 9:2909-2913', '10.1039/c3mb70326f',
     'O-GlcNAc', 'not recorded', 'SVM + ANBPB features', 'no URL in abstract', '-', 'not found', '-', 'abstract', 'NOT BENCHMARKED (no reachable server and no released trained model; decided 2026-10-13, from the 2026-10-06 checks)'),
    (A, 'Wu et al. (Lee group)', '2014', 'Wu et al., "Characterization and identification of protein O-GlcNAcylation sites with substrate specificity", BMC Bioinformatics 15(S16):S1',
     '10.1186/1471-2105-15-s16-s1', 'O-GlcNAc', 'dbOGAP, 375 sites, 168 proteins', 'MDDLogo-clustered SVMs (PSSM features)', 'none given (web tool mentioned, no URL)', '-', 'none', '-',
     'PDF', 'EXCLUDED 2026-10-12: no URL, code or model; is the "MDDLogo-clustered SVM" in O-GlcNAcPRED-II Table 3; follow-up = OGTSite'),
    (A, 'OGTSite', '2015', 'Kao et al., BMC Bioinformatics 16(S18):S10', '10.1186/1471-2105-16-s18-s10',
     'O-GlcNAc', '410 sites (dbOGAP, OGlycBase, UniProtKB)', 'profile HMM + SVM (OGT substrate motifs)', 'http://csb.cse.yzu.edu.tw/OGTSite/', 'down',
     'data on the website', 'down', 'full text + live check', 'NOT BENCHMARKED (no reachable server and no released trained model; decided 2026-10-13, from the 2026-10-06 checks); Mauri 2021 could still run it'),
    (A, 'PGlcS', '2015', 'Zhao et al., J Theor Biol 380:524-529', '10.1016/j.jtbi.2015.06.026',
     'O-GlcNAc', 'dbOGAP, 167 proteins, 339 sites', 'SVM, k-means negatives', 'none (future work)', '-', 'supplement: C++ feature-extraction scripts (PSIPRED, PSI-BLAST, DISOPRED2 parsers), Tables S1-S2', 'checked 2026-10-12: no trained model, no SVM code',
     'PDF + supplement', 'EXCLUDED 2026-10-12: no trained model; own test MCC 0.0697; Khalid 2024 cites this paper as "O-GlcNAcScan"'),
    (A, 'O-GlcNAcPRED-II', '2018', 'Jia, Zuo & Zou, Bioinformatics 34(12):2029-2036', '10.1093/bioinformatics/bty039',
     'O-GlcNAc', 'dbOGAP + Jochmann 2014; 945 pos / 50,914 neg', 'Rotation Forest ensemble (RF, KNN, NB, SVM); window 23; KPCA oversampling + fuzzy undersampling',
     'http://121.42.167.206/OGlcPred/', 'down', 'train/test sets in Supplementary Material', 'not checked',
     'PDF + live check', 'NOT BENCHMARKED (no reachable server and no released trained model; decided 2026-10-13, from the 2026-10-06 checks); CV MCC 0.79 vs independent-set MCC 0.10'),
    (A, 'Mauri et al. models (OGP)', '2021', 'Mauri et al., Adv Appl Bioinform Chem 14:87-102', '10.2147/AABC.S294867',
     'O-GlcNAc', '236 mammalian proteins; 565 pos / 40,271 neg', 'RF, GBT, SVM', 'none', '-', 'https://gitlab.in2p3.fr/cmsb-public/OGP', 'timed out',
     'PDF + live check', 'EXCLUDED as comparator 2026-10-12 (benchmark paper, no released predictor; GitLab unreachable 2026-10-06/12); key reference for motivation; best PPV ~3% at real proportions'),
    (A, 'LM-OGlcNAc-Site', '2023', 'IJMS 24(21):16000', '10.3390/ijms242116000',
     'O-GlcNAc', 'not recorded', 'Ankh + ESM-2 + ProtT5 embeddings, decision fusion', 'http://kcdukkalab.org/LMOGlcNAcSite', 'down',
     'https://github.com/KCLabMTU/LM-OGlcNAc-Site (code + data)', 'up (last push 2024-06)', 'abstract + full text + live check', ''),
    (A, 'O-GlcNAcPRED-DL', '2024', 'Hu et al., J Proteome Res 23:95-106', '10.1021/acs.jproteome.3c00458',
     'O-GlcNAc', 'O-GlcNAcAtlas 2.0, human + mouse', 'CNN + BiLSTM ensemble, Word2Vec, window 29', 'http://oglcnac.org/pred_dl', 'up (runs in the browser)',
     'https://github.com/hufengzhu/O-GlcNAcPRED-DL-master ; https://github.com/oglcnac/oglcnac-source (TF.js models)', 'up',
     'PDF + live check', 'ensemble weights chosen on the independent test sets'),
    (A, 'DeepO-GlcNAc', '2024', 'Zhang, Deng et al., Front Cell Dev Biol 12:1456728', '10.3389/fcell.2024.1456728',
     'O-GlcNAc', 'dbPTM 2022, human; 2,696 pos / 20,556 neg', 'CNN + SE attention + LSTM, window 21', 'http://124.220.189.245:8000/', 'down (2026-10-06, 2026-10-12)',
     'not linked in paper; unofficial: https://github.com/seniorpigeon/DeepO-GlcNAc , https://github.com/Drtindg/DeepO-GlcNAc', 'repos exist; authorship unconfirmed',
     'PDF + live check', 'NOT BENCHMARKED (server down, no code linked from paper; dropped 2026-10-12); seniorpigeon = co-first author Tingzhi Deng, training script only; random split before window-level CD-HIT'),
    (A, 'Khalid et al. (ESM-2)', '2024', 'Khalid et al., PLoS ONE 19(12):e0316215', '10.1371/journal.pone.0316215',
     'O-GlcNAc', 'O-GlcNAc database v1.3, ~1,100 human proteins (890/112/111 train/val/test proteins)', 'fine-tuned ESM-2 35M, token classification, proteins of 500-800 aa', 'none', '-',
     'data + training notebook: https://www.kaggle.com/datasets/esm2oglcnac/o-glcnac-prediction-model', 'notebook obtained (user, 2026-10-06); weights not published',
     'PDF + notebook', 'NOT BENCHMARKED (no server and no released trained model; data + training notebook only; decided 2026-10-13); notebook: labels not shifted for the <cls> token (off by one residue); metrics are weighted over all residues incl. padding; no AUROC; paper numbers hard-coded in a plotting cell, saved test output differs (acc 0.145)'),
    (A, 'Seber & Braatz (RNN-225)', '2024', 'Comput Chem Eng 189:108818 (bioRxiv 2023)', '10.1016/j.compchemeng.2024.108818',
     'O-GlcNAc', 'several literature sources, mammalian proteins (v1/v3/v5 datasets; repo README)', 'LSTM on one-hot +/-20 windows', 'none', '-',
     'https://github.com/PedroSeber/O-GlcNAcylation_Prediction (MIT)', 'opened 2026-10-12: code, model RNN-225_20-window_dict.pt, training data',
     'abstract + repo README, citation.bib, Predict.py', 'older model; Predict.py keeps it "for compatibility"; NB08'),
    (A, 'Seber (RNN-[600,75])', '2024', 'Seber, arXiv 2402.17131 (preprint, v3 2025-09-17; no journal reference on arXiv 2026-10-12)', 'arXiv:2402.17131',
     'O-GlcNAc', 'Wulff-Fuentes et al. 2021, Sci Data (10.1038/s41597-021-00810-4); in repo as OVSlab_allSpecies_O-GlcNAcome_PS.csv', 'LSTM on one-hot +/-20 windows, new loss function', 'none', '-',
     'https://github.com/PedroSeber/O-GlcNAcylation_Prediction (MIT)', 'opened 2026-10-12: code, model RNN-[600,75]_20-window_dict.pt, training data',
     'repo README, citation.bib, Predict.py; arXiv metadata', 'newer model, default in Predict.py; same repo as Seber & Braatz; NB08'),
    (A, 'Hong et al.', '2025', 'Bioinformatics (correction: 10.1093/bioinformatics/btaf281)', '10.1093/bioinformatics/btaf034',
     'O-GlcNAc per earlier survey; title says "O-glycosylation" - verify', 'not recorded', 'sparse RNN + 3D local-environment features', 'none', '-',
     'https://github.com/pseokyoung/o-glcnac-prediction (code, data, features)', 'up (last push 2025-03)', 'abstract + live check', 'EXCLUDED 2026-10-12: O-GlcNAc-specific but no trained model, prediction script or server (training code + data only)'),
    # --- B. multi-PTM / generic glycosylation predictors ---
    (B, 'PTM-ssMP', '2018', 'Int J Biol Sci', '10.7150/ijbs.24121', 'O-GlcNAc is one of 9 PTM types', 'not recorded', 'not recorded',
     'http://bioinformatics.ustc.edu.cn/PTM-ssMP/index/', 'down (DNS)', 'none', '-', 'full text + live check', 'Mauri 2021 used it as a data source'),
    (B, 'Multiple-kernel SVM', '2017', 'Wang et al., PeerJ 5:e3261', '10.7717/peerj.3261', 'O-linked, includes O-GlcNAc data', 'not recorded', 'multiple-kernel SVM',
     'none', '-', 'MATLAB; raw data as supplementary files', 'not checked', 'full text', 'EXCLUDED 2026-10-12: has an O-GlcNAc output but only raw data released (no model, code or server)'),
    (B, 'Stack-OglyPred-PLM', '2024', 'Bioinformatics', '10.1093/bioinformatics/btae643', 'human O-linked (intra- + extracellular)', 'not recorded', 'ProtT5 + Ankh, stacking',
     'none', '-', 'https://github.com/PakhrinLab/Stack-OglyPred-PLM', 'up', 'full text + live check', 'INCLUDED 2026-10-12: nucleocytoplasmic (GlcNAc) model only (dbPTM O-linked sites on DeepLoc-predicted nucleocytoplasmic proteins); trained .h5 on Google Drive'),
    (B, 'MTPrompt-PTM', '2025', 'Biomolecules 15(6):843', '10.3390/biom15060843', 'generic O-linked (S, T)', 'not recorded', 'multi-task prompt PLM',
     'none', '-', 'https://github.com/hanye311/MTPrompt-PTM', 'not opened', 'full text', 'EXCLUDED 2026-10-12: pooled O-linked class (GlcNAc, GalNAc, Xyl, Fuc, Man...), no O-GlcNAc-specific output; model released'),
    (B, 'GlycoMine', '2015', 'Li et al., Bioinformatics 31(9):1411-1419', '10.1093/bioinformatics/btu852', 'human N-, C-, O-linked; O type never stated',
     '649 O-sites, 208 proteins', 'random forest, window 15', 'http://www.structbioinfor.org/Lab/GlycoMine/', 'down (DNS)', 'datasets were on the server', 'down',
     'PDF + live check', 'O-GlcNAcPRED-DL could not reach it in 2022'),
    (B, 'GlycoMine-struct', '2016', 'Li et al., Sci Rep 6:34595', '10.1038/srep34595', 'human N- and mixed O-linked', 'UniProt, PhosphoSitePlus, SysPTM, O-GlycBase 6.0',
     'random forest + PDB structural features', 'http://glycomine.erc.monash.edu/Lab/GlycoMine_Struct/', 'down (DNS)', 'none', '-', 'full text + live check', 'needs a PDB structure'),
    (B, 'GlycoEP', '2013', 'Chauhan et al., PLoS ONE 8(6):e67008', '10.1371/journal.pone.0067008', 'eukaryotic N-, O-, C-linked', 'Swiss-Prot June 2011', 'SVM',
     'http://www.imtech.res.in/raghava/glycoep/', 'redirects to a malformed address', 'none', '-', 'full text + live check', 'O-linked MCC 0.20'),
    (B, 'Captor', '2021', 'Zhu et al., J Bioinform Comput Biol (Crossref 2021; DeepO-GlcNAc cites 2022)', '10.1142/s0219720021500293',
     'human "O-glycosylation"; type not stated', 'not stated', 'SVM, window 31', 'none', '-', 'https://github.com/YanZhu06/Captor (data + MATLAB code)', 'up (last push 2021-08)',
     'abstract + README + live check', 'EXCLUDED 2026-10-12: generic human O-glycosylation (type not stated); no trained model, MATLAB retraining only'),
    (B, 'SPRINT-Gly', '2019', 'Taherzadeh et al., Bioinformatics 35:4140', '10.1093/bioinformatics/btz215', 'N-linked and mucin-type (O-GalNAc) O-linked only (p2), human + mouse', 'six sources incl. UniProt, dbPTM, GlycoProtDB, UniCarbKB', 'DNN (N) + SVM (O)',
     'https://sparks-lab.org/server/sprint-gly/', 'up (one sequence at a time)', 'https://gitlab.com/matthew.campbell1980/sprint-gly (per paper)', 'not opened', 'PDF', 'EXCLUDED 2026-10-12: mucin-type O-glycosylation, not O-GlcNAc'),
    (B, 'EnsembleGly', '2007', 'BMC Bioinformatics 8:438', '10.1186/1471-2105-8-438', 'N-, O-, C-linked', 'O-GlycBase 6.0 (per GlycoMine)', 'SVM ensemble',
     'http://www.turing.cs.iastate.edu/EnsembleGly/ (as printed in GlycoEP)', 'not checked', 'not checked', 'not checked', 'named in GlycoEP / GlycoMine; Crossref', ''),
    (B, 'GPP', '2008', 'Hamby & Hirst, BMC Bioinformatics 9:500', '10.1186/1471-2105-9-500', 'N- and O-linked', 'not checked', 'random forest',
     'not checked', 'not checked', 'not checked', 'not checked', 'named in PTG-PLM / GlycoMine; Crossref', ''),
    (B, 'GlycoMine_PU', '', 'not found', 'not found', 'N-, O-, C-linked', 'not checked', 'positive-unlabelled learning',
     'not checked', 'not checked', 'not checked', 'not checked', 'named in PTG-PLM', ''),
    (B, 'Alkuhlani et al. (O-glycosylation PLM)', '2023', 'Int J Intell Comput Inf Sci', '10.21608/ijicis.2023.160986.1218', 'O-glycosylation (type not checked)', 'not checked',
     'pre-trained PLM + ML', 'not checked', 'not checked', 'not checked', 'not checked', 'cited by Khalid 2024; Crossref', 'EXCLUDED 2026-10-12: generic human O-glycosylation (OGP repository); no code or server mentioned'),
    (B, 'MusiteDeep', '2020', 'Nucleic Acids Res', '10.1093/nar/gkaa275', 'no O-GlcNAc mention in full text', '-', 'deep learning, multi-PTM',
     'not checked', 'not checked', 'not checked', 'not checked', 'full text', ''),
    (B, 'CLASPP', '2026', 'PLoS Comput Biol', '10.1371/journal.pcbi.1014616', 'no O-GlcNAc mention in full text', '-', 'unified PTM model',
     'not checked', 'not checked', 'not checked', 'not checked', 'full text', ''),
    # --- C. off-target or not predictors ---
    (C, 'NetOGlyc-4.0', '2013', 'Steentoft et al., EMBO J', '10.1038/emboj.2013.79', 'O-GalNAc (mucin-type); likely O-GlcNAc sites excluded', 'SimpleCell, 12 human cell lines',
     'SVM', 'http://www.cbs.dtu.dk/services/NetOGlyc-4.0/', 'timed out (DTU successor page not checked)', 'none', '-', 'full text + live check',
     'used as a comparator by DeepO-GlcNAc (AUC 0.82)'),
    (C, 'CKSAAP_OGlySite', '2008', 'BMC Bioinformatics 9:101', '10.1186/1471-2105-9-101', 'mucin-type O-GalNAc, mammalian', 'not checked', 'SVM, CKSAAP',
     'not checked', 'not checked', 'not checked', 'not checked', 'named in GlycoMine; Crossref', ''),
    (C, 'GlycoPP', '2012', 'PLoS ONE', '10.1371/journal.pone.0040155', 'prokaryotic N- and O-linked', 'not checked', 'SVM',
     'not checked', 'not checked', 'not checked', 'not checked', 'named in PTG-PLM; Crossref', ''),
    (C, 'PTG-PLM', '2022', 'Alkuhlani et al., Axioms 11(9):469', '10.3390/axioms11090469', 'N-glycosylation + lysine glycation only', 'UniProt (N-gly), PredGly (K-gly)',
     'CNN on PLM embeddings', 'none', '-', 'https://github.com/Alhasanalkuhlani/PTG-PLM', 'up (last push 2022-09)', 'full text + live check', 'no O-linked model'),
    (C, 'DAPPLE 2', '2016', 'Trost et al., J Proteome Res 15:2760-2767', '10.1021/acs.jproteome.6b00304', 'homology transfer of known sites (20 PTM types)', '15 PTM databases',
     'homology-based', 'http://saphire.usask.ca/saphire/dapple2', 'timed out', 'none', '-', 'abstract + live check', 'does not predict new sites'),
    (C, 'Gana & Vasudevan', '2019', 'BMC Mol Cell Biol 20:21', '10.1186/s12860-019-0200-9', 'O-GlcNAc vs N-glycosylation (negatives)', 'dbOGAP + PhosphoSitePlus with PDB structures',
     'ridge-regression linear probability model (explanatory)', 'none', '-', 'data: Additional file 1 (glycos_public.xlsx)', 'not checked', 'PDF', 'not a released tool'),
    # --- D. named only ---
    (D, 'Yang & Han', '2017', 'Comput Biol Chem 71:258-263', '', 'O-glycosylation', '', '', '', '', '', '', 'cited in Gana 2019', 'EXCLUDED 2026-10-12: generic O-glycosylation (99 UniProt mammalian proteins); doi 10.1016/j.compbiolchem.2017.09.005'),
    (D, 'Hassan et al.', '2015', 'Bioinform Biol Insights 9:103-109', '', 'O-glycosylation', '', 'RF + GA-tuned PSO', '', '', '', '', 'cited in Gana 2019', 'EXCLUDED 2026-10-12: O-GlycBase v6, all O-glycan types; doi 10.4137/BBI.S26864'),
    (D, 'Chen et al.', '2015', 'Med Biol Eng Comput 53(6):535-544', '', 'O-glycosylation', '', 'multi-scale AA composition + feature selection', '', '', '', '', 'cited in Gana 2019', 'EXCLUDED 2026-10-12: Swiss-Prot O-glycosylation framed as mucin-type; doi 10.1007/s11517-015-1268-9'),
    (D, "Sasaki et al. SVM tool", '', 'not recorded', '', 'O-glycosylation', '', 'SVM', '', '', '', '', 'named in GlycoMine', 'EXCLUDED 2026-10-12: N- and O-glycosylation from Swiss-Prot, no tool; probable paper doi 10.2197/ipsjtbio.2.25 (IPSJ Trans Bioinform 2009)'),
]

bad = [r[1] for r in ROWS if len(r) != len(COLUMNS) - 2]
assert not bad, f'wrong number of fields: {bad}'
ordered = sorted(ROWS, key=lambda r: (-int(r[2]) if r[2] else 1, r[0], r[1].lower()))  # newest first; no year last; then group, tool
rows = [[i + 1, *r, ''] for i, r in enumerate(ordered)]
with open(HERE / 'competitor_table.csv', 'w', newline='', encoding='utf-8-sig') as f:  # utf-8-sig: opens correctly in Excel
    w = csv.writer(f)
    w.writerow(COLUMNS)
    w.writerows(rows)

md = [f'# Possible competitors: complete table ({CHECKED})', '',
      'Generated by `build_competitor_table.py`; the same content is in `competitor_table.csv` (with an empty "Manual check" column).',
      '"not found" = searched but absent; "not checked" = not searched yet; status columns are live checks on the date above.', '']
md.append('| ' + ' | '.join(COLUMNS) + ' |')
md.append('|' + '---|' * len(COLUMNS))
for r in rows:
    cells = [str(c).replace('|', '/') for c in r]
    if cells[5].startswith('10.'):
        cells[5] = f'[{cells[5]}](https://doi.org/{cells[5]})'
    md.append('| ' + ' | '.join(cells) + ' |')
(HERE / 'competitor_table.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
records = [dict(zip(COLUMNS, r)) for r in rows]
html = (HERE / 'competitor_table_template.html').read_text(encoding='utf-8')
html = html.replace('__DATA__', json.dumps(records, ensure_ascii=False)).replace('__CHECKED__', CHECKED)
(HERE / 'competitor_table.html').write_text(html, encoding='utf-8')
print(f'{len(rows)} rows -> competitor_table.csv, competitor_table.md, competitor_table.html')
