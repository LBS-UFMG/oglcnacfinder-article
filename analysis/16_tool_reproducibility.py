# %%
import os, subprocess
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))
subprocess.run(['pip', 'install', '-q', './O-GlcNAcFinder'], check=True)

import numpy as np
import pandas as pd
from oglcnacfinder import read_fasta

# ---------------------------------------------------------------------------
# Check of the O-GlcNAcFinder package against the NB14 human proteome predictions
#   LDB3 (O75112), CIC (Q96RK0, 2,517 residues, several windows), YEATS2 (Q9ULM3) and 5 random
#   human proteins are scored with the command-line tool; logits are compared with NB14.
# ---------------------------------------------------------------------------
OUT_DIR = 'data_processed/proteome_predictions'
CHECK_FASTA, CHECK_TSV = '/content/check.fasta', '/content/check.tsv'

human = {h.split('|')[1]: s for h, s in read_fasta(f'{OUT_DIR}/human_swissprot_canonical.fasta')}
ids = ['O75112', 'Q96RK0', 'Q9ULM3'] + list(np.random.default_rng(0).choice(sorted(human), 5, replace=False))
with open(CHECK_FASTA, 'w') as f:
    f.writelines(f'>{a}\n{human[a]}\n' for a in ids)
subprocess.run(['oglcnacfinder', CHECK_FASTA, '-m', 'data_processed/final_model/final_model.pt', '-o', CHECK_TSV],
               check=True)

got = pd.read_csv(CHECK_TSV, sep='\t')
ref = pd.read_parquet(f'{OUT_DIR}/human_proteome_predictions.parquet',
                      columns=['accession', 'position', 'logit', 'probability', 'called'])
cmp = got.merge(ref, left_on=['protein_id', 'position'], right_on=['accession', 'position'], suffixes=('', '_nb14'))
print(f'{len(got):,} S/T scored, {len(cmp):,} matched to NB14; logits max |diff| '
      f'{np.abs(cmp["logit"] - cmp["logit_nb14"]).max():.1e}; calls identical: {(cmp["called"] == cmp["called_nb14"]).all()}')
ldb3 = got[got['protein_id'] == 'O75112'].assign(rank=lambda d: d['probability'].rank(ascending=False, method='first'))
print(ldb3.nsmallest(3, 'rank')[['position', 'residue', 'probability', 'rank']].to_string(index=False))
assert len(cmp) == len(got) and np.abs(cmp['logit'] - cmp['logit_nb14']).max() < 1e-4
