"""Best local alignment of human LDB3 378-414 to every LDB3 sequence of several vertebrates."""
import glob
from Bio import SeqIO
from Bio.Align import PairwiseAligner, substitution_matrices

human = str(next(SeqIO.parse('O75112.fasta', 'fasta')).seq)
q0 = 377  # 0-based start of residue 378
query = human[q0:414]
print('human 378-414', query)
al = PairwiseAligner(mode='local', substitution_matrix=substitution_matrices.load('BLOSUM62'),
                     open_gap_score=-10, extend_gap_score=-1)
names = {'10090': 'mouse', '9913': 'cow', '9031': 'chicken', '8364': 'Xenopus', '7955': 'zebrafish',
         '9258': 'platypus', '9823': 'pig', '9615': 'dog'}
for f in sorted(glob.glob('orth_*.fasta')):
    org = names.get(f[5:-6], f)
    best = None
    for r in SeqIO.parse(f, 'fasta'):
        a = al.align(query, str(r.seq))[0]
        if best is None or a.score > best[0]:
            best = (a.score, r.id, a)
    if best is None:
        continue
    score, rid, a = best
    (qs, qe), = [a.aligned[0][0]] if len(a.aligned[0]) == 1 else [(a.aligned[0][0][0], a.aligned[0][-1][1])]
    ts, te = a.aligned[1][0][0], a.aligned[1][-1][1]
    q = query[a.aligned[0][0][0]:a.aligned[0][-1][1]]
    t = str(a.query[ts:te])
    # residue-level view of the human positions 382, 385, 399, 400
    m = {}
    for (q1, q2), (t1, t2) in zip(*a.aligned):
        for i in range(q2 - q1):
            m[q0 + q1 + i + 1] = a.query[t1 + i]
    key = ' '.join(f'{human[p-1]}{p}->{m.get(p, "-")}' for p in (382, 385, 399, 400))
    print(f'{org:10s} {rid:28s} score {score:5.0f} | {key}')
    print(' ' * 11, 'H', q)
    print(' ' * 11, 'O', t)
