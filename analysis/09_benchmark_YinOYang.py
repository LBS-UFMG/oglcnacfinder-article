# %%
import os
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# YinOYang 1.2 (Gupta & Brunak 2002, Pac Symp Biocomput 2002:310-322), DTU Health Tech
#   web server: https://services.healthtech.dtu.dk/services/YinOYang-1.2/
#   Server restrictions (as stated on the page): at most 2,000 sequences and 200,000 amino
#   acids per submission; each sequence 21-4,000 amino acids. Letters other than the 20
#   standard amino acids are converted to X.
#   Test proteins longer than 4,000 residues are split into two overlapping pieces, the first
#   and the last 4,000 residues; each site will be scored from the piece where it lies farther
#   from an edge (cut at the middle of the overlap), and sites in the overlap, scored twice,
#   check that splitting does not change the scores.
#   This cell sizes the input against the limits and packs it, in protein order, into
#   submissions within both limits.
# ---------------------------------------------------------------------------
OUT_DIR = 'data_processed/other_tools/YinOYang'
os.makedirs(OUT_DIR, exist_ok=True)
MAX_SEQS, MAX_RESIDUES, MIN_LEN, MAX_LEN = 2000, 200_000, 21, 4000

proteins = pd.read_parquet('data_processed/atlas5_proteins.parquet')
test = proteins[proteins['split'] == 'test'].reset_index(drop=True)
lengths = test['sequence'].str.len()
assert (lengths >= MIN_LEN).all()

# Input sequences: whole proteins, or two pieces for proteins longer than MAX_LEN
pieces = []
for pid, seq in zip(test['protein_id'], test['sequence']):
    if len(seq) <= MAX_LEN:
        pieces.append({'name': pid, 'protein_id': pid, 'start': 1, 'sequence': seq})
    else:
        assert len(seq) <= 2 * MAX_LEN - 1000, 'overlap would be shorter than 1,000 residues'
        start2 = len(seq) - MAX_LEN + 1
        pieces.append({'name': f'{pid}_1', 'protein_id': pid, 'start': 1, 'sequence': seq[:MAX_LEN]})
        pieces.append({'name': f'{pid}_2', 'protein_id': pid, 'start': start2, 'sequence': seq[start2 - 1:]})
pieces = pd.DataFrame(pieces)
pieces['length'] = pieces['sequence'].str.len()

# Submissions: pieces in order, a new submission when either limit would be exceeded
batch, n_seqs, residues = 0, 0, 0
batches = []
for n in pieces['length']:
    if n_seqs and (n_seqs == MAX_SEQS or residues + n > MAX_RESIDUES):
        batch, n_seqs, residues = batch + 1, 0, 0
    batches.append(batch)
    n_seqs, residues = n_seqs + 1, residues + n
pieces['submission'] = np.array(batches) + 1
pieces.drop(columns='sequence').to_csv(f'{OUT_DIR}/submission_plan.csv', index=False)

long_ = test[lengths > MAX_LEN]
overlaps = [2 * MAX_LEN - len(s) for s in long_['sequence']]
extra = sorted(set(''.join(test['sequence'])) - set('ACDEFGHIKLMNPQRSTVWY'))
print('Test set against the YinOYang server limits')
print(f'  Proteins:              {len(test)} ({lengths.sum():,} residues; median {int(lengths.median()):,}, '
      f'longest {lengths.max():,})')
print(f'  Shorter than {MIN_LEN}:       none')
print(f'  Longer than {MAX_LEN:,}:     {len(long_)} -> split into 2 overlapping pieces each '
      f'(overlaps of {min(overlaps):,}-{max(overlaps):,} residues)')
print(f'  Non-standard letters:  {", ".join(extra) if extra else "none"} (read as unknown by the server)')
print(f'\nSubmissions (at most {MAX_SEQS:,} sequences and {MAX_RESIDUES:,} residues each): '
      f'{pieces["submission"].nunique()}')
for b, grp in pieces.groupby('submission'):
    n_split = grp['name'].str.contains('_[12]$').sum()
    note = f' (incl. {n_split} pieces of long proteins)' if n_split else ''
    print(f'  {b}: {len(grp):>4} sequences, {grp["length"].sum():>7,} residues{note}')

# %%
# ---------------------------------------------------------------------------
# Submission files: one FASTA file per submission, in the order of the plan above.
#   Header: the piece name (protein ID, with _1/_2 for the pieces of long proteins).
#   Sequences wrapped at 60 residues per line.
# ---------------------------------------------------------------------------
SUB_DIR = f'{OUT_DIR}/submissions'
os.makedirs(SUB_DIR, exist_ok=True)

for b, grp in pieces.groupby('submission'):
    path = f'{SUB_DIR}/submission_{b}.fasta'
    with open(path, 'w') as f:
        for name, seq in zip(grp['name'], grp['sequence']):
            f.write(f'>{name}\n' + '\n'.join(seq[i:i + 60] for i in range(0, len(seq), 60)) + '\n')
    print(f'{path}: {len(grp)} sequences, {grp["length"].sum():,} residues')

# Every piece name is unique and every test protein is covered
assert pieces['name'].is_unique and set(pieces['protein_id']) == set(test['protein_id'])
