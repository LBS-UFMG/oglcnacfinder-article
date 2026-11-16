# %%
import os
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))
import pandas as pd

# ---------------------------------------------------------------------------
# Atlas 5.0 sites per protein for the O-GlcNAcFinder website (task D1)
#   From the paper dataset (atlas5_proteins.parquet, NB01): one row per protein with its kept ID,
#   every ID of its merged group (D18), organism(s), the positive S/T and the S/T covered by
#   ambiguous peptides (masked, D11), and the sequence. Loaded with `php spark atlas:load`.
# ---------------------------------------------------------------------------
OUT = 'data_processed/website/atlas5_sites.tsv'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

p = pd.read_parquet('data_processed/atlas5_proteins.parquet')
positions = lambda xs: ','.join(str(int(x)) for x in sorted(xs))
table = pd.DataFrame({'protein_id': p['protein_id'],
                      'ids': p['merged_ids'].str.replace('; ', ';'),
                      'organism': p['organism'].fillna(''),
                      'organisms': p['organisms'].fillna('').str.replace('; ', ';'),
                      'positive': p['positive'].map(positions),
                      'ambiguous': p['masked'].map(positions),
                      'sequence': p['sequence']})
assert not table.apply(lambda c: c.astype(str).str.contains('[\t\n]')).any().any()
table.to_csv(OUT, sep='\t', index=False, lineterminator='\n')
print(f'{len(table):,} proteins; {p["positive"].str.len().sum():,} positive and {p["masked"].str.len().sum():,} '
      f'ambiguous S/T -> {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)')

# %%
import base64, gzip, json, subprocess
import numpy as np
import torch
from tqdm.auto import tqdm
subprocess.run(['pip', 'install', '-q', './O-GlcNAcFinder', 'biopython'], check=True)
from oglcnacfinder import __version__, read_fasta
from oglcnacfinder.atlas import transfer_sites

# ---------------------------------------------------------------------------
# Proteome browser data for the O-GlcNAcFinder website
#   From the five reference-proteome predictions (NB14), one row per protein: accession, Swiss-Prot
#   or unreviewed, gene, protein name, sequence, and the calibrated probability of every S/T in
#   sequence order (float32, little-endian, base64). Also per protein: S/T count, S/T called at the
#   cut-off, highest probability, and number of candidate new sites (NB14 cell 5 filters).
#   Atlas 5.0 sites on the proteome sequence, as for jobs submitted to the website: entries with the
#   same accession (isoform suffix ignored, any ID of a merged Atlas protein) or the identical
#   sequence (table of cell 0); sites transferred by the worker function transfer_sites (identical
#   sequence, or global alignment keeping identical residues). Columns: positive, ambiguous,
#   positions known only through alignment, and the matched entries (JSON).
#   One gzipped TSV per species and a summary (proteomes.json). Loaded with `php spark proteome:load`.
# ---------------------------------------------------------------------------
REFERENCE = {'human': ('Homo sapiens', 'UP000005640', 9606), 'mouse': ('Mus musculus', 'UP000000589', 10090),
             'arabidopsis': ('Arabidopsis thaliana', 'UP000006548', 3702), 'rice': ('Oryza sativa', 'UP000059680', 39947),
             'fly': ('Drosophila melanogaster', 'UP000000803', 7227)}
PRED_DIR = 'data_processed/proteome_predictions/reference_proteomes'
WEB_DIR = 'data_processed/website'
CUTOFF = torch.load('data_processed/final_model/final_model.pt', map_location='cpu')['cutoff']['probability']
COLUMNS = ['accession', 'reviewed', 'gene', 'protein', 'length', 'st', 'called', 'max_probability', 'candidates',
           'atlas_positive', 'atlas_ambiguous', 'atlas_aligned', 'atlas_matches', 'sequence', 'probabilities']

# Atlas entries exactly as the website stores them (cell 0 table)
atlas = pd.read_csv(f'{WEB_DIR}/atlas5_sites.tsv', sep='\t', keep_default_na=False, dtype=str)
to_list = lambda s: [int(x) for x in s.split(',') if x]
entries, atlas_by_acc, atlas_by_seq = {}, {}, {}
for r in atlas.itertuples():
    entries[r.protein_id] = {'protein_id': r.protein_id, 'organism': r.organism or None, 'sequence': r.sequence,
                             'positive': to_list(r.positive), 'ambiguous': to_list(r.ambiguous)}
    for base in {i.split('-')[0].strip() for i in r.ids.split(';')}:
        atlas_by_acc.setdefault(base, set()).add(r.protein_id)
    atlas_by_seq.setdefault(r.sequence, set()).add(r.protein_id)

positions = lambda xs: ','.join(str(x) for x in sorted(xs))
summary = {'model': f'O-GlcNAcFinder {__version__}', 'atlas': '5.0', 'cutoff': CUTOFF, 'proteomes': []}
for name, (species, upid, taxon) in REFERENCE.items():
    meta = json.load(open(f'{PRED_DIR}/{name}_{upid}_{taxon}.json'))
    prot = pd.read_parquet(f'{PRED_DIR}/{name}_proteins.parquet', columns=['accession', 'reviewed', 'gene', 'protein'])
    seqs = {h.split('|')[1]: s for h, s in read_fasta(f'{PRED_DIR}/{name}_{upid}_{taxon}.fasta')}
    pred = (pd.read_parquet(f'{PRED_DIR}/{name}_proteome_predictions.parquet', columns=['accession', 'position', 'probability'])
            .sort_values(['accession', 'position']))
    probs = {a: g for a, g in pred.groupby('accession', sort=False)}
    no_st = pred.iloc[:0]   # proteins without S/T have no prediction rows
    cand = pd.read_parquet(f'{PRED_DIR}/{name}_new_site_candidates.parquet', columns=['accession'])['accession'].value_counts()

    rows, n_aligned_proteins = [], 0
    for r in tqdm(list(prot.itertuples()), desc=name):
        seq = seqs[r.accession]
        g = probs.get(r.accession, no_st)
        st = [i + 1 for i, c in enumerate(seq) if c in 'ST']
        assert g['position'].tolist() == st, f'{r.accession}: scored positions differ from the S/T of the sequence'
        p = g['probability'].to_numpy(dtype='<f4')
        ids = sorted(atlas_by_acc.get(r.accession, set()) | atlas_by_seq.get(seq, set()))
        classes, aligned, matches = transfer_sites(seq, [entries[i] for i in ids]) if ids else ({}, set(), [])
        n_aligned_proteins += bool(aligned)
        rows.append({'accession': r.accession, 'reviewed': int(r.reviewed), 'gene': r.gene, 'protein': r.protein,
                     'length': len(seq), 'st': len(st), 'called': int((p >= CUTOFF).sum()),
                     'max_probability': f'{p.max():.6f}' if len(p) else '', 'candidates': int(cand.get(r.accession, 0)),
                     'atlas_positive': positions(k for k, v in classes.items() if v == 'positive'),
                     'atlas_ambiguous': positions(k for k, v in classes.items() if v == 'ambiguous'),
                     'atlas_aligned': positions(aligned),
                     'atlas_matches': json.dumps(matches, separators=(',', ':')) if ids else '',
                     'sequence': seq, 'probabilities': base64.b64encode(p.tobytes()).decode()})
    table = pd.DataFrame(rows, columns=COLUMNS)
    assert len(table) == len(prot) and set(probs) <= set(table['accession']) and table['st'].sum() == len(pred)
    assert not table.drop(columns='atlas_matches').apply(lambda c: c.astype(str).str.contains('[\t\n]')).any().any()
    out = f'{WEB_DIR}/proteome_{name}.tsv.gz'
    with gzip.open(out, 'wt', encoding='utf-8', newline='') as f:
        table.to_csv(f, sep='\t', index=False, lineterminator='\n')

    n_pos = table['atlas_positive'].str.count(',').add(table['atlas_positive'].ne('')).sum()
    summary['proteomes'].append({'name': name, 'species': species, 'proteome': upid, 'taxon': taxon,
                                 'uniprot_release': meta['uniprot_release'], 'uniprot_release_date': meta['uniprot_release_date'],
                                 'proteins': len(table), 'swissprot': int(table['reviewed'].sum()), 'st': int(table['st'].sum()),
                                 'called': int(table['called'].sum()), 'candidate_proteins': int((table['candidates'] > 0).sum()),
                                 'atlas_proteins': int(table['atlas_matches'].ne('').sum()), 'atlas_positive': int(n_pos)})
    s = summary['proteomes'][-1]
    print(f'{species}: {s["proteins"]:,} proteins ({s["swissprot"]:,} Swiss-Prot), {s["st"]:,} S/T, {s["called"]:,} called, '
          f'{s["candidate_proteins"]:,} candidate proteins; Atlas: {s["atlas_proteins"]:,} proteins, {s["atlas_positive"]:,} positive S/T '
          f'({n_aligned_proteins:,} proteins with sites by alignment) -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)')

with open(f'{WEB_DIR}/proteomes.json', 'w') as f:
    json.dump(summary, f, indent=1)
print(f'Cut-off {CUTOFF:.4f}; {summary["model"]} -> {WEB_DIR}/proteomes.json')
