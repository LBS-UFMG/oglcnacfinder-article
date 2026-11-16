# %%
import os
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))

ATLAS_FILES = {
    'unambiguous': 'data/Atlas 5.0_unambiguous sites_20251208.csv',
    'ambiguous': 'data/Atlas 5.0_ambiguous sites_20251208.csv',
}
os.makedirs('data_processed', exist_ok=True)

for name, path in ATLAS_FILES.items():
    assert os.path.exists(path), f'Missing: {path}'
    print(f'{name:12s} {os.path.getsize(path) / 1e6:.1f} MB  {path}')

# %%
import pandas as pd

def read_atlas(path):
    for enc in ('utf-8', 'latin-1'):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc), enc
        except UnicodeDecodeError:
            continue
    raise ValueError(f'Cannot decode {path}')

atlas = {}
for name, path in ATLAS_FILES.items():
    df, enc = read_atlas(path)
    atlas[name] = df
    print(f'{name:12s} {df.shape[0]:>6,} records x {df.shape[1]} columns  (encoding: {enc})')

print('\nColumns identical in both files:', list(atlas['unambiguous'].columns) == list(atlas['ambiguous'].columns))
print(list(atlas['unambiguous'].columns))

# %%
pd.set_option('display.width', 200, 'display.max_columns', 30, 'display.max_colwidth', 40)

for name, df in atlas.items():
    print(f'\n===== {name} =====')
    print(df[['accession', 'accession_source', 'species', 'peptide_seq', 'site_residue',
              'position_in_peptide', 'position_in_protein']].head(3).to_string())
    for col in ['accession_source', 'site_residue', 'species']:
        print(f'\n{col}:', df[col].value_counts(dropna=False).head(12).to_dict())
    pos = df['position_in_protein']
    print('\nposition_in_protein non-integer:', (~pos.fillna('').str.fullmatch(r'\d+')).sum(),
          '| examples:', pos[~pos.fillna('').str.fullmatch(r'\d+')].unique()[:8].tolist())

# %%
# Flow-diagram log: records, distinct sites and distinct proteins after each step
flow = []

def log_step(step):
    """Log records, distinct sites and distinct proteins in each file. Proteins are counted
    per Atlas accession until a sequence is chosen, then per chosen sequence (protein_id).
    For the ambiguous file, sites are distinct (protein, reported position) pairs as written in
    the Atlas; those rows are later located by their peptide, so this count is descriptive only."""
    row = {'step': step}
    for n, d in atlas.items():
        id_col = 'protein_id' if 'protein_id' in d else 'accession'
        pos_col = 'pos' if 'pos' in d else 'position_in_protein'
        row[f'{n}_records'] = len(d)
        row[f'{n}_sites'] = d[[id_col, pos_col]].drop_duplicates().shape[0]
        row[f'{n}_proteins'] = d[id_col].nunique()
    flow.append(row)
    print(f"\n{step}")
    for n in atlas:
        print(f"  {n:12s} records={row[f'{n}_records']:>7,}  sites={row[f'{n}_sites']:>7,}  proteins={row[f'{n}_proteins']:>6,}")

log_step('Raw Atlas 5.0 records')

def unify_case(s):
    """Map case variants to the most frequent spelling (e.g. 'arabidopsis' -> 'Arabidopsis')."""
    counts = pd.concat([d[s.name] for d in atlas.values()]).value_counts()
    canonical = {}
    for label in counts.index:  # most frequent first
        canonical.setdefault(label.casefold(), label)
    return s.map(lambda x: canonical.get(x.casefold(), x) if isinstance(x, str) else x)

for name, df in atlas.items():
    for col in df.columns:
        df[col] = df[col].str.strip()
    df['site_residue'] = df['site_residue'].str.upper()
    for col in ['species', 'accession_source']:
        before = df[col].nunique()
        df[col] = unify_case(df[col])
        print(f'{name:12s} {col:17s} distinct labels: {before} -> {df[col].nunique()}')

log_step('After label normalization (no rows removed)')

# %%
import re, requests

for name, df in atlas.items():
    print(f'\n===== {name} =====')
    for src, g in df.groupby(df['accession_source'].fillna('<missing>')):
        acc = g['accession'].dropna().unique()
        print(f'{src:10s} {len(acc):>5,} accessions  e.g. {list(acc[:5])}')
    up = df.loc[df['accession_source'] == 'UniProt', 'accession'].dropna().unique()
    print('UniProt accessions with isoform suffix (e.g. P12345-2):', sum('-' in a for a in up))

# 'From' databases accepted by the UniProt ID-mapping service (queried live)
cfg = requests.get('https://rest.uniprot.org/configure/idmapping/fields', timeout=60).json()
froms = sorted({r['name'] for g in cfg['groups'] for r in g['items'] if r.get('from')})
print('\nUniProt ID-mapping "from" databases:', len(froms))
print([f for f in froms if re.search(r'TAIR|Araport|Gramene|Ensembl|EMBL|GenBank|RefSeq|GI|PDB|FlyBase|Rice|MSU|RAP', f, re.I)])

# %%
import urllib.request
from datetime import date

MSU7_URL = ('https://rice.uga.edu/pub/data/Eukaryotic_Projects/o_sativa/annotation_dbs/'
            'pseudomolecules/version_7.0/all.dir/all.pep')
MSU7_PATH = 'data/msu7_all.pep'
if not os.path.exists(MSU7_PATH):
    urllib.request.urlretrieve(MSU7_URL, MSU7_PATH)
print(f'MSU v7 proteome: {os.path.getsize(MSU7_PATH) / 1e6:.1f} MB (downloaded/checked {date.today()})')

def read_fasta(path):
    seqs, name = {}, None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                name = line[1:].split()[0]
                seqs[name] = []
            elif name:
                seqs[name].append(line)
    return {k: ''.join(v).rstrip('*') for k, v in seqs.items()}

msu7 = read_fasta(MSU7_PATH)
print(f'{len(msu7):,} protein models')

for name, df in atlas.items():
    rice = df.loc[df['accession_source'] == 'MSU ID', 'accession'].dropna().unique()
    missing = [a for a in rice if a not in msu7]
    print(f'{name:12s} rice accessions: {len(rice):,}  found in MSU v7: {len(rice) - len(missing):,}  '
          f'not found: {len(missing)}  e.g. {missing[:8]}')

# %%
# Rice IDs absent from MSU v7: number of rows and sites affected (dropped in a later cell)
rice_missing = {a for df in atlas.values()
                for a in df.loc[df['accession_source'] == 'MSU ID', 'accession'].dropna() if a not in msu7}
print('Rice IDs not in MSU v7:', sorted(rice_missing))
for name, df in atlas.items():
    m = df['accession'].isin(rice_missing)
    print(f"{name:12s} rows: {m.sum()}  sites: {df.loc[m, 'position_in_protein'].nunique()}")

# %%
import time

def uniprot_idmap(ids, from_db):
    """Map ids with the UniProt ID-mapping service; returns {from_id: [(accession, reviewed, organism, length), ...]}."""
    r = requests.post('https://rest.uniprot.org/idmapping/run',
                      data={'from': from_db, 'to': 'UniProtKB', 'ids': ','.join(ids)}, timeout=60)
    r.raise_for_status()
    job = r.json()['jobId']
    while True:
        s = requests.get(f'https://rest.uniprot.org/idmapping/status/{job}', timeout=60).json()
        if s.get('jobStatus') in (None, 'FINISHED'):
            break
        time.sleep(3)
    tsv = requests.get(f'https://rest.uniprot.org/idmapping/uniprotkb/results/stream/{job}',
                       params={'format': 'tsv', 'fields': 'accession,reviewed,organism_name,length'}, timeout=300).text
    out = {}
    for line in tsv.strip().split('\n')[1:]:
        f = line.split('\t')
        out.setdefault(f[0], []).append(tuple(f[1:5]))
    return out

# accessions per source (both files together)
both = pd.concat(atlas.values())
src_ids = {s: sorted(g['accession'].dropna().unique()) for s, g in both.groupby('accession_source')}
genbank = src_ids.get('GenBank', [])
routes = {
    'TAIR':    ('Araport', src_ids.get('TAIR', [])),
    'FlyBase': ('FlyBase', src_ids.get('FlyBase', [])),
    'GenBank (CDS)':    ('EMBL-GenBank-DDBJ_CDS', [a for a in genbank if not re.match(r'^[A-Z]{2}_\d', a)]),
    'GenBank (RefSeq)': ('RefSeq_Protein', [a for a in genbank if re.match(r'^[A-Z]{2}_\d', a)]),
    'NCBI (GI)': ('GI_number', src_ids.get('NCBI', [])),
}

idmap, summary = {}, []
for label, (from_db, ids) in routes.items():
    res = uniprot_idmap(ids, from_db) if ids else {}
    idmap[label] = res
    summary.append({'source': label,
                    'distinct IDs': len(ids),
                    'maps to a single UniProt entry': sum(len(v) == 1 for v in res.values()),
                    'maps to several UniProt entries': sum(len(v) > 1 for v in res.values()),
                    'not found in UniProt': len(ids) - len(res)})
print(pd.DataFrame(summary).to_string(index=False))

# PDB chains: check entry status with the PDBe API before mapping
for pdb in sorted({c.split('_')[0].lower() for c in src_ids.get('PDB', [])}):
    st = requests.get(f'https://www.ebi.ac.uk/pdbe/api/pdb/entry/status/{pdb}', timeout=60).json()[pdb][0]
    print(f"PDB {pdb.upper()}: status={st['status_code']}  superseded_by={st.get('superceded_by')}")
idmap['PDB'] = {}  # obsolete entry with no chain-level UniProt mapping: its rows are dropped

# %%
for label in ['TAIR', 'FlyBase', 'GenBank (RefSeq)', 'NCBI (GI)']:
    multi = {k: v for k, v in idmap[label].items() if len(v) > 1}
    n_entries = pd.Series([len(v) for v in multi.values()])
    n_reviewed = pd.Series([sum(e[1] == 'reviewed' for e in v) for v in multi.values()])
    print(f'\n===== {label}: {len(multi)} IDs map to several UniProt entries =====')
    print(f'UniProt entries per ID: min {n_entries.min()}, median {n_entries.median():.0f}, max {n_entries.max()}')
    print('IDs by number of reviewed (Swiss-Prot) entries:', n_reviewed.value_counts().sort_index().to_dict())
    for k, v in list(multi.items())[:3]:
        print(f'  {k}: ' + '; '.join(f'{a} ({r}, {o[:25]}, {l} aa)' for a, r, o, l in v[:6]))

# %%
import io

# ---------------------------------------------------------------------------
# Fetch UniProt sequences
#   - Rows whose accession source is UniProt (or missing) use their accession directly.
#   - Rows from other databases contribute every candidate UniProt entry returned by the
#     ID mapping; one candidate per ID is chosen later, by checking the reported peptide
#     and site against each candidate sequence.
#   - Accessions are resolved with the ID-mapping service (UniProtKB_AC-ID), which also
#     follows secondary (merged) accessions to the current primary entry.
#   - Isoform accessions (e.g. P12345-2) are fetched individually, as isoform sequences.
# ---------------------------------------------------------------------------

def uniprot_idmap_table(ids, from_db, fields):
    """Run a UniProt ID-mapping job; return (DataFrame with 'From' + requested fields, UniProt release)."""
    r = requests.post('https://rest.uniprot.org/idmapping/run',
                      data={'from': from_db, 'to': 'UniProtKB', 'ids': ','.join(ids)}, timeout=60)
    r.raise_for_status()
    job = r.json()['jobId']
    while requests.get(f'https://rest.uniprot.org/idmapping/status/{job}', timeout=60).json().get('jobStatus') not in (None, 'FINISHED'):
        time.sleep(3)
    res = requests.get(f'https://rest.uniprot.org/idmapping/uniprotkb/results/stream/{job}',
                       params={'format': 'tsv', 'fields': fields}, timeout=600)
    res.raise_for_status()
    return pd.read_csv(io.StringIO(res.text), sep='\t', dtype=str), res.headers.get('X-UniProt-Release')

# 1. Accessions to fetch
is_uniprot_row = both['accession_source'].isin(['UniProt']) | both['accession_source'].isna()
atlas_uniprot = set(both.loc[is_uniprot_row, 'accession'].dropna())
isoforms = sorted(a for a in atlas_uniprot if '-' in a)
canonical_requests = sorted(a for a in atlas_uniprot if '-' not in a)
candidates = sorted({e[0] for res in idmap.values() for entries in res.values() for e in entries})
to_fetch = sorted(set(canonical_requests) | set(candidates))
print(f'Atlas UniProt accessions: {len(atlas_uniprot):,} ({len(isoforms)} isoforms)')
print(f'Mapping candidates:       {len(candidates):,}')
print(f'Canonical accessions to fetch: {len(to_fetch):,}')

# 2. Canonical entries (secondary accessions resolved to their primary entry)
assert 'UniProtKB_AC-ID' in froms
up, release = uniprot_idmap_table(to_fetch, 'UniProtKB_AC-ID',
                                  'accession,reviewed,organism_name,organism_id,length,sequence')
print(f'\nUniProt release: {release}')
resolved = up.groupby('From')['Entry'].apply(list)
inactive = set(up.loc[up['Sequence'].isna(), 'From'])
active = resolved[~resolved.index.isin(inactive)]
print(f'Found as current primary accession: {sum(active.index == active.str[0]):,}')
print(f'Secondary accession -> primary:     {sum(active.index != active.str[0]):,}')
print(f'Inactive entry (no sequence data):  {len(inactive):,}   (recovered later through UniParc)')
print(f'Not found (obsolete/deleted):       {len(set(to_fetch) - set(resolved.index)):,}')

# 3. Isoform sequences, fetched one by one
isoform_seqs, isoform_missing = {}, []
for acc in isoforms:
    r = requests.get(f'https://rest.uniprot.org/uniprotkb/{acc}.fasta', timeout=60)
    if r.ok and r.text.startswith('>'):
        isoform_seqs[acc] = ''.join(r.text.split('\n')[1:])
    else:
        isoform_missing.append(acc)
print(f'\nIsoforms fetched: {len(isoform_seqs)}  not found: {len(isoform_missing)}  e.g. {isoform_missing[:5]}')

# %%
# ---------------------------------------------------------------------------
# Recover obsolete UniProt accessions through UniParc
#   UniParc keeps every sequence ever linked to a UniProt accession. All of them are
#   kept as candidates; the candidate is chosen later by checking the reported peptide
#   and site against each sequence (same rule as for IDs mapping to several entries).
#   Accessions with no UniParc sequence are dropped.
# ---------------------------------------------------------------------------
obsolete = sorted(set(to_fetch) - set(resolved.index))
uniparc_candidates = {}
for acc in obsolete:
    r = requests.get('https://rest.uniprot.org/uniparc/stream',
                     params={'query': f'dbid:{acc}', 'format': 'fasta'}, timeout=120)
    r.raise_for_status()
    seqs = {}
    for block in r.text.split('>')[1:]:
        lines = block.strip().split('\n')
        seqs[lines[0].split()[0]] = ''.join(lines[1:])
    uniparc_candidates[acc] = seqs
    print(f'{acc}: {len(seqs)} UniParc sequence(s)  ' + ', '.join(f'{u} ({len(s)} aa)' for u, s in seqs.items()))

# %%
# Accessions with no UniProt entry and no UniParc sequence: the rows affected (dropped in the next cell)
bad = [a for a in obsolete if not uniparc_candidates[a]]
for name, df in atlas.items():
    rows = df[df['accession'].isin(bad)]
    print(f'\n===== {name}: {len(rows)} rows =====')
    print(rows[['accession', 'species', 'gene_name', 'protein_name', 'peptide_seq',
                'site_residue', 'position_in_protein', 'pmid']].to_string())

# %%
# ---------------------------------------------------------------------------
# Remove rows whose protein cannot be identified (one logged step per reason)
# ---------------------------------------------------------------------------

def drop_rows(is_dropped, step, files=None):
    """Remove rows where is_dropped(df) is True (in the given files, default both) and log the counts."""
    for name in (files or list(atlas)):
        df = atlas[name]
        atlas[name] = df[~is_dropped(df)].copy()
    log_step(step)

def mapping_route(source, accession):
    """ID-mapping route used for a non-UniProt accession (see the mapping cell)."""
    if source == 'GenBank':
        return 'GenBank (RefSeq)' if re.match(r'^[A-Z]{2}_\d', accession) else 'GenBank (CDS)'
    return {'TAIR': 'TAIR', 'FlyBase': 'FlyBase', 'NCBI': 'NCBI (GI)'}.get(source)

def has_no_uniprot_mapping(df):
    route = [mapping_route(s, a) for s, a in zip(df['accession_source'], df['accession'])]
    return pd.Series([r is not None and a not in idmap[r] for r, a in zip(route, df['accession'])], index=df.index)

# Rows without a protein accession
drop_rows(lambda df: df['accession'].isna(),
          'Removed: rows without a protein accession')

# PDB chains of an obsolete entry with no chain-level UniProt mapping
drop_rows(lambda df: df['accession_source'] == 'PDB',
          'Removed: PDB chains of an obsolete PDB entry')

# Rice IDs absent from the MSU v7 proteome
drop_rows(lambda df: (df['accession_source'] == 'MSU ID') & ~df['accession'].isin(msu7.keys()),
          'Removed: rice ID not in MSU v7')

# Accessions with no UniProt entry and no UniParc sequence (engineered antibody, placeholders,
# malformed ID); obsolete accessions with UniParc sequences are kept and recovered later
unrecoverable = [a for a in obsolete if not uniparc_candidates[a]]
drop_rows(lambda df: df['accession'].isin(unrecoverable),
          'Removed: accessions with no UniProt/UniParc entry')

# Non-UniProt IDs that the UniProt ID-mapping service could not map
drop_rows(has_no_uniprot_mapping,
          'Removed: non-UniProt IDs without a UniProt mapping')

# %%
# ---------------------------------------------------------------------------
# Parse site positions and residues (unambiguous sites only)
#   - Position: a plain integer, or a single residue + number (e.g. 'S216', 'T886').
#     Strings listing several positions (e.g. '321 ,T322') or other formats are dropped.
#   - Residue: a letter in site_residue; a residue embedded in the position string is used
#     when site_residue is missing. Non-letter values are treated as missing; missing
#     residues are read from the protein sequence in a later step, once the sequence is chosen.
#   - Only serine and threonine sites are kept (tyrosine and cysteine sites are dropped).
#   Ambiguous rows are not filtered here: they are only used to mask residues, and are
#   located by searching their peptide in the protein sequence.
# ---------------------------------------------------------------------------

def parse_position(value):
    """Return (position, residue_from_string), or (None, None) if the string is not a single site."""
    if not isinstance(value, str):
        return None, None
    value = value.strip().upper()
    if re.fullmatch(r'\d+', value):
        return int(value), None
    m = re.fullmatch(r'([A-Z])(\d+)', value)
    if m:
        return int(m.group(2)), m.group(1)
    return None, None

df = atlas['unambiguous']
parsed = df['position_in_protein'].map(parse_position)
df['pos'] = parsed.str[0]
residue = df['site_residue'].where(df['site_residue'].str.fullmatch(r'[A-Z]', na=False))
df['res'] = residue.fillna(parsed.str[1])
print(f'positions parsed: {df["pos"].notna().sum():,}/{len(df):,}   '
      f'residue known: {df["res"].notna().sum():,}   residue missing: {df["res"].isna().sum():,}')

drop_rows(lambda df: df['pos'].isna(),
          'Removed: missing or multi-position site positions', files=['unambiguous'])

drop_rows(lambda df: df['res'].notna() & ~df['res'].isin(['S', 'T']),
          'Removed: sites on residues other than S/T', files=['unambiguous'])

# %%
# ---------------------------------------------------------------------------
# Choose one sequence per Atlas protein ID
#   Candidates: the UniProt entry (or entries) an accession resolves to, the isoform
#   sequence, the MSU v7 model for rice, or every UniProt entry an external ID maps to.
#   A candidate "supports" a row when:
#     - unambiguous row: the residue at the site position is S/T (and equals the reported
#       residue, if any) and the reported peptide occurs in the sequence covering that position;
#     - ambiguous row: the reported peptide occurs in the sequence.
#   The candidate supporting the most rows is chosen; ties are broken by preferring
#   reviewed (Swiss-Prot) entries, then entries whose organism name contains the Atlas
#   species label, then the lowest accession.
# ---------------------------------------------------------------------------

def clean_peptide(p):
    return re.sub(r'[^A-Z]', '', p.upper()) if isinstance(p, str) else ''

# UniProt entries by requested accession: [(entry, sequence, reviewed, organism)]
no_sequence = up[up['Sequence'].isna()]
print(f'UniProt entries returned without a sequence: {len(no_sequence)}')
if len(no_sequence):
    print(no_sequence.head(5).to_string())
uniprot_entries = {}
for r in up.dropna(subset=['Sequence']).itertuples(index=False):
    organism = r.Organism if isinstance(r.Organism, str) else ''
    uniprot_entries.setdefault(r.From, []).append((r.Entry, r.Sequence, r.Reviewed == 'reviewed', organism))

def candidates_for(accession, source):
    if source == 'MSU ID':
        return [(accession, msu7[accession], False, '')]
    if accession in isoform_seqs:
        return [(accession, isoform_seqs[accession], False, '')]
    route = mapping_route(source, accession)
    if route is None:                                   # UniProt accession (or missing source)
        return uniprot_entries.get(accession, [])
    return [e for mapped in idmap[route].get(accession, []) for e in uniprot_entries.get(mapped[0], [])]

def supports_site(seq, pos, res, peptide):
    if not (1 <= pos <= len(seq)) or seq[pos - 1] not in 'ST':
        return False
    if isinstance(res, str) and seq[pos - 1] != res:
        return False
    if not peptide:
        return True
    start = seq.find(peptide)
    while start != -1:                                  # any occurrence covering the site
        if start < pos <= start + len(peptide):
            return True
        start = seq.find(peptide, start + 1)
    return False

unamb, amb = atlas['unambiguous'], atlas['ambiguous']
unamb['peptide'] = unamb['peptide_seq'].map(clean_peptide)
amb['peptide'] = amb['peptide_seq'].map(clean_peptide)
u_by_acc = dict(list(unamb.groupby('accession')))
a_by_acc = dict(list(amb.groupby('accession')))
empty = unamb.iloc[0:0]

chosen = {}          # Atlas accession -> (entry, sequence)
no_support = []
ids = pd.concat([unamb, amb])[['accession', 'accession_source']].drop_duplicates()
for acc, source in zip(ids['accession'], ids['accession_source']):
    if acc in chosen or acc in no_support:
        continue
    u_rows, a_rows = u_by_acc.get(acc, empty), a_by_acc.get(acc, empty)
    species = pd.concat([u_rows['species'], a_rows['species']]).dropna()
    species = species.iloc[0].casefold() if len(species) else ''
    scored = []
    for entry, seq, reviewed, organism in candidates_for(acc, source):
        n = sum(supports_site(seq, int(p), r, pep) for p, r, pep in zip(u_rows['pos'], u_rows['res'], u_rows['peptide']))
        n += sum(bool(pep) and pep in seq for pep in a_rows['peptide'])
        scored.append((-n, not reviewed, species not in organism.casefold(), entry, seq))
    scored.sort()
    if scored and scored[0][0] < 0:
        chosen[acc] = (scored[0][3], scored[0][4])
    else:
        no_support.append(acc)

print(f'Atlas protein IDs: {len(chosen) + len(no_support):,}   sequence chosen: {len(chosen):,}   '
      f'no candidate supports any row: {len(no_support):,}  e.g. {no_support[:6]}')

# %%
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# UniParc sequence versions for IDs with no supporting candidate
#   For each such ID, every UniParc sequence linked to its UniProt accession(s) is added
#   as a candidate (UniParc keeps all sequence versions, including those of inactive
#   entries). The candidates are scored with the same rule as above.
# ---------------------------------------------------------------------------

def uniparc_sequences(accession):
    """All UniParc sequences cross-referenced to a UniProt accession: {UPI: sequence}."""
    r = requests.get('https://rest.uniprot.org/uniparc/stream',
                     params={'query': f'dbid:{accession}', 'format': 'fasta'}, timeout=120)
    r.raise_for_status()
    seqs = {}
    for block in r.text.split('>')[1:]:
        lines = block.strip().split('\n')
        seqs[lines[0].split()[0]] = ''.join(lines[1:])
    return seqs

def uniprot_accessions_for(accession, source):
    """UniProt accessions behind an Atlas ID (none for rice MSU IDs)."""
    if source == 'MSU ID':
        return []
    route = mapping_route(source, accession)
    if route is None:
        return [accession]
    return [m[0] for m in idmap[route].get(accession, [])]

def score_candidates(acc, candidates):
    """Return candidates sorted by: rows supported (desc), reviewed, species match, accession."""
    u_rows, a_rows = u_by_acc.get(acc, empty), a_by_acc.get(acc, empty)
    species = pd.concat([u_rows['species'], a_rows['species']]).dropna()
    species = species.iloc[0].casefold() if len(species) else ''
    scored = []
    for entry, seq, reviewed, organism in candidates:
        n = sum(supports_site(seq, int(p), r, pep) for p, r, pep in zip(u_rows['pos'], u_rows['res'], u_rows['peptide']))
        n += sum(bool(pep) and pep in seq for pep in a_rows['peptide'])
        scored.append((-n, not reviewed, species not in organism.casefold(), entry, seq))
    return sorted(scored)

source_of = dict(zip(ids['accession'], ids['accession_source']))
recovered, still_unsupported = {}, []
for acc in tqdm(no_support, desc='UniParc'):
    source = source_of[acc]
    extra = [(upi, seq, False, '')
             for uacc in uniprot_accessions_for(acc, source)
             for upi, seq in uniparc_sequences(uacc).items()]
    scored = score_candidates(acc, candidates_for(acc, source) + extra)
    if scored and scored[0][0] < 0:
        recovered[acc] = (scored[0][3], scored[0][4])
    else:
        still_unsupported.append(acc)

chosen.update(recovered)
print(f'Recovered with a UniParc version: {len(recovered):,}   still unsupported: {len(still_unsupported):,}')
print('Still unsupported, by source:', pd.Series([source_of[a] for a in still_unsupported]).value_counts(dropna=False).to_dict())

# %%
# ---------------------------------------------------------------------------
# Diagnose IDs whose rows are not supported by any candidate sequence
#   (these IDs also failed with every UniParc sequence version in the previous cell).
#   For each unambiguous row, against the current candidates (UniProt entries, MSU models):
#   is the peptide found, and if so, how far is the reported site position from the S/T
#   positions inside the found peptide?
# ---------------------------------------------------------------------------
diag = []
for acc in still_unsupported:
    source = source_of[acc]
    cands = candidates_for(acc, source)
    for _, row in u_by_acc.get(acc, empty).iterrows():
        pep, pos = row['peptide'], int(row['pos'])
        found = [(seq, seq.find(pep)) for _, seq, _, _ in cands if pep and pep in seq]
        if not pep:
            status, offset = 'no peptide reported', None
        elif not found:
            status, offset = 'peptide not in any candidate', None
        else:
            seq, start = found[0]
            st_positions = [start + i + 1 for i, aa in enumerate(pep) if aa in 'ST']
            offset = min((p - pos for p in st_positions), key=abs) if st_positions else None
            status = 'peptide found, position off'
        diag.append({'source': source, 'status': status, 'offset': offset})

diag = pd.DataFrame(diag)
print(pd.crosstab(diag['source'], diag['status']))
off = diag.loc[diag['status'] == 'peptide found, position off', 'offset']
print('\nMost common offsets (found position - reported position):', off.value_counts().head(10).to_dict())

# %%
# ---------------------------------------------------------------------------
# Apply the sequence checks
#   1. Rows of Atlas IDs with no supporting sequence are dropped.
#   2. Each remaining row is checked against its protein's chosen sequence:
#        - unambiguous: the site residue is S/T (equal to the reported residue, if any)
#          and the reported peptide occurs in the sequence covering the site;
#        - ambiguous: the reported peptide occurs in the sequence.
#   3. Missing residues are read from the sequence.
#   From here on, proteins and sites are counted per chosen sequence (protein_id); several
#   Atlas IDs can point to the same sequence, so this change is logged as its own step.
# ---------------------------------------------------------------------------

drop_rows(lambda df: ~df['accession'].isin(chosen.keys()),
          'Removed: no candidate sequence supports the reported sites')

for name, df in atlas.items():
    df['protein_id'] = df['accession'].map(lambda a: chosen[a][0])
    df['sequence'] = df['accession'].map(lambda a: chosen[a][1])
log_step('Counted per chosen sequence (no rows removed)')

unamb = atlas['unambiguous']
unamb['site_ok'] = [supports_site(seq, int(p), r, pep)
                    for seq, p, r, pep in zip(unamb['sequence'], unamb['pos'], unamb['res'], unamb['peptide'])]
drop_rows(lambda df: ~df['site_ok'], 'Removed: site not supported by the chosen sequence', files=['unambiguous'])

amb = atlas['ambiguous']
amb['peptide_ok'] = [bool(pep) and pep in seq for seq, pep in zip(amb['sequence'], amb['peptide'])]
drop_rows(lambda df: ~df['peptide_ok'], 'Removed: peptide not found in the chosen sequence', files=['ambiguous'])

unamb = atlas['unambiguous']
unamb['res'] = [seq[int(p) - 1] for seq, p in zip(unamb['sequence'], unamb['pos'])]
print('\nResidues of unambiguous sites:', unamb['res'].value_counts().to_dict())

# %%
# ---------------------------------------------------------------------------
# Build the per-protein dataset
#   - Proteins: those with at least one unambiguous site. Proteins reported only in the
#     ambiguous file are excluded, since no site on them is localized.
#   - Positive: every S/T reported as an unambiguous site (one label per residue, however
#     many papers report it). A residue that is unambiguous in any paper is positive, even
#     if another paper reports it only as an ambiguous candidate.
#   - Masked: every S/T inside every occurrence of an ambiguous peptide, unless positive.
#     Masked residues are neither positive nor negative.
#   - Negative: every other S/T in the protein (as in O-GlcNAcPRED-DL: the other S/T
#     residues of the same proteins).
# ---------------------------------------------------------------------------

def st_positions_in_peptide(seq, peptide):
    """1-based positions of S/T inside every occurrence of the peptide."""
    found, start = set(), seq.find(peptide)
    while start != -1:
        found.update(start + i + 1 for i, aa in enumerate(peptide) if aa in 'ST')
        start = seq.find(peptide, start + 1)
    return found

unamb, amb = atlas['unambiguous'], atlas['ambiguous']
dataset_ids = set(unamb['protein_id'])
ambiguous_only = set(amb['protein_id']) - dataset_ids
print(f'Proteins with >=1 unambiguous site: {len(dataset_ids):,}')
print(f'Proteins only in the ambiguous file (excluded): {len(ambiguous_only):,}')

positives = unamb.groupby('protein_id')['pos'].apply(lambda s: set(s.astype(int)))
masked = {}
for pid, seq, pep in zip(amb['protein_id'], amb['sequence'], amb['peptide']):
    if pid in dataset_ids:
        masked.setdefault(pid, set()).update(st_positions_in_peptide(seq, pep))

species = unamb.groupby('protein_id')['species'].agg(
    lambda s: s.value_counts().index[0] if s.notna().any() else None)
n_species = unamb.groupby('protein_id')['species'].nunique()
print(f'Proteins reported under more than one species label: {(n_species > 1).sum()}')

sequences = unamb.drop_duplicates('protein_id').set_index('protein_id')['sequence']
records = []
for pid in sorted(dataset_ids):
    seq = sequences[pid]
    pos = positives[pid]
    msk = masked.get(pid, set()) - pos
    st = {i + 1 for i, aa in enumerate(seq) if aa in 'ST'}
    records.append({'protein_id': pid, 'species': species[pid], 'length': len(seq), 'sequence': seq,
                    'positive': sorted(pos), 'masked': sorted(msk), 'n_negative': len(st - pos - msk)})
proteins = pd.DataFrame(records)

n_pos, n_msk, n_neg = proteins['positive'].str.len().sum(), proteins['masked'].str.len().sum(), proteins['n_negative'].sum()
print(f'\nFinal dataset: {len(proteins):,} proteins')
print(f'  positive S/T sites: {n_pos:,}')
print(f'  masked S/T (ambiguous candidates): {n_msk:,}')
print(f'  negative S/T: {n_neg:,}   (positive fraction: {n_pos / (n_pos + n_neg):.2%})')
print('\nProteins and positive sites per species:')
print(proteins.groupby('species').agg(proteins=('protein_id', 'size'),
                                      sites=('positive', lambda s: s.str.len().sum()))
      .sort_values('sites', ascending=False).to_string())

proteins.to_parquet('data_processed/atlas5_proteins.parquet', index=False)
pd.DataFrame(flow).to_csv('data_processed/atlas5_flow.csv', index=False)

# %%
# ---------------------------------------------------------------------------
# Species labels: look up each Atlas label in NCBI Taxonomy (report only, no changes)
# ---------------------------------------------------------------------------
EUTILS = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'

def ncbi_taxonomy(term):
    """NCBI Taxonomy search: list of (taxid, scientific name) matching a label."""
    ids = requests.get(EUTILS + 'esearch.fcgi', params={'db': 'taxonomy', 'term': term, 'retmode': 'json'},
                       timeout=60).json()['esearchresult']['idlist']
    time.sleep(0.4)
    if not ids:
        return []
    summ = requests.get(EUTILS + 'esummary.fcgi', params={'db': 'taxonomy', 'id': ','.join(ids), 'retmode': 'json'},
                        timeout=60).json()['result']
    time.sleep(0.4)
    return [(t, summ[t]['scientificname']) for t in summ['uids']]

labels = sorted(atlas['unambiguous']['species'].dropna().unique())
taxonomy_hits = {label: ncbi_taxonomy(label) for label in tqdm(labels, desc='NCBI Taxonomy')}

print(f'{"Atlas label":55s} hits  NCBI scientific name (taxid)')
for label, hits in taxonomy_hits.items():
    shown = '; '.join(f'{name} ({t})' for t, name in hits[:3]) or '-'
    print(f'{label:55s} {len(hits):>4}  {shown}')

# %%
# ---------------------------------------------------------------------------
# Species of each protein
#   1. Organism of the chosen sequence as recorded by UniProt (NCBI taxid), named with the
#      NCBI Taxonomy scientific name.
#   2. Sequences without a UniProt organism (MSU rice models, UniParc sequences): the Atlas
#      species label, if NCBI Taxonomy returns exactly one taxon for it.
#   Proteins resolved by neither rule are listed for review.
# ---------------------------------------------------------------------------

# UniProt organism taxid per entry; isoforms take the organism of their canonical entry
entry_taxid = dict(zip(up['Entry'], up['Organism (ID)']))
missing_parents = sorted({p.split('-')[0] for p in proteins['protein_id']
                          if '-' in p and p.split('-')[0] not in entry_taxid})
if missing_parents:
    extra, _ = uniprot_idmap_table(missing_parents, 'UniProtKB_AC-ID', 'accession,organism_id')
    entry_taxid.update(dict(zip(extra['Entry'], extra['Organism (ID)'])))

def taxid_of(protein_id):
    t = entry_taxid.get(protein_id.split('-')[0])
    return str(int(float(t))) if isinstance(t, str) and t.strip() else None

proteins['taxid'] = proteins['protein_id'].map(taxid_of)
proteins['species_source'] = proteins['taxid'].notna().map({True: 'UniProt organism', False: None})

# Rule 2: Atlas label with exactly one NCBI Taxonomy hit
no_org = proteins['taxid'].isna()
single_hit = {label: hits[0][0] for label, hits in taxonomy_hits.items() if len(hits) == 1}
from_label = proteins.loc[no_org, 'species'].map(single_hit)
proteins.loc[no_org, 'taxid'] = from_label
proteins.loc[no_org & from_label.notna(), 'species_source'] = 'Atlas label (single NCBI hit)'

# Scientific names for all taxids
taxids = sorted(proteins['taxid'].dropna().unique())
names = {}
for i in range(0, len(taxids), 200):
    summ = requests.get(EUTILS + 'esummary.fcgi', params={'db': 'taxonomy', 'id': ','.join(taxids[i:i + 200]),
                                                          'retmode': 'json'}, timeout=60).json()['result']
    names.update({t: summ[t]['scientificname'] for t in summ['uids']})
    time.sleep(0.4)
proteins['organism'] = proteins['taxid'].map(names)

print(proteins['species_source'].value_counts(dropna=False).to_string())
unresolved = proteins[proteins['organism'].isna()]
print(f'\nUnresolved: {len(unresolved)} proteins')
print(unresolved.groupby('species').agg(proteins=('protein_id', 'size'),
                                        sites=('positive', lambda s: s.str.len().sum())).to_string())

# %%
# ---------------------------------------------------------------------------
# Rule 3 (UniParc sequences still without an organism): take the organism from the UniParc
# cross-references when they all agree on a single taxon; otherwise leave unresolved.
# ---------------------------------------------------------------------------

def uniparc_taxa(upi):
    """Set of (taxid, scientific name) recorded in the cross-references of a UniParc entry."""
    r = requests.get(f'https://rest.uniprot.org/uniparc/{upi}', params={'format': 'json'}, timeout=120)
    r.raise_for_status()
    taxa = set()
    for xref in r.json().get('uniParcCrossReferences', []):
        org = xref.get('organism') or {}
        if org.get('taxonId'):
            taxa.add((str(org['taxonId']), org.get('scientificName', '')))
    return taxa

pending = proteins.index[proteins['organism'].isna()]
for i in tqdm(pending, desc='UniParc organisms'):
    taxa = uniparc_taxa(proteins.at[i, 'protein_id'])
    if len(taxa) == 1:
        taxid, _ = taxa.pop()
        proteins.at[i, 'taxid'] = taxid
        proteins.at[i, 'species_source'] = 'UniParc cross-references'
    else:
        proteins.at[i, 'species_source'] = f'unresolved ({len(taxa)} taxa in UniParc)'

# NCBI scientific names for any new taxids
new = sorted(set(proteins['taxid'].dropna()) - set(names))
if new:
    summ = requests.get(EUTILS + 'esummary.fcgi', params={'db': 'taxonomy', 'id': ','.join(new), 'retmode': 'json'},
                        timeout=60).json()['result']
    names.update({t: summ[t]['scientificname'] for t in summ['uids']})
proteins['organism'] = proteins['taxid'].map(names)

print(proteins['species_source'].value_counts(dropna=False).to_string())
still = proteins[proteins['organism'].isna()]
if len(still):
    print(still[['protein_id', 'species', 'species_source']].to_string())

# %%
# Check: does UniParc list the expected organism for each unresolved protein?
expected_name = {'mouse': 'Mus musculus', 'rat': 'Rattus norvegicus', 'Drosophila': 'Drosophila melanogaster'}
expected_taxid = {label: [t for t, n in ncbi_taxonomy(name) if n == name][0] for label, name in expected_name.items()}
print('Expected taxids (NCBI Taxonomy):', expected_taxid)

check = []
for i in proteins.index[proteins['organism'].isna()]:
    pid, label = proteins.at[i, 'protein_id'], proteins.at[i, 'species']
    taxa = uniparc_taxa(pid)
    check.append({'protein_id': pid, 'atlas_label': label,
                  'expected_in_uniparc': expected_taxid.get(label) in {t for t, _ in taxa},
                  'uniparc_taxa': '; '.join(sorted(n for _, n in taxa))[:110]})
print(pd.DataFrame(check).to_string(index=False))

# %%
# ---------------------------------------------------------------------------
# Rule 4 (sequences shared by several species in UniParc): the Atlas labels 'mouse', 'rat'
# and 'Drosophila' are read as Mus musculus, Rattus norvegicus and Drosophila melanogaster;
# the taxon is assigned only where UniParc lists it for that sequence (proteins where it is
# not listed stay unresolved and stop the notebook at the check below).
# ---------------------------------------------------------------------------
for c in check:
    if c['expected_in_uniparc']:
        i = proteins.index[proteins['protein_id'] == c['protein_id']][0]
        proteins.at[i, 'taxid'] = expected_taxid[c['atlas_label']]
        proteins.at[i, 'species_source'] = 'Atlas label confirmed by UniParc'
names.update({expected_taxid[k]: v for k, v in expected_name.items()})
proteins['organism'] = proteins['taxid'].map(names)

assert proteins['organism'].notna().all()
print(proteins['species_source'].value_counts().to_string())
print('\nProteins and positive sites per organism:')
print(proteins.groupby('organism').agg(proteins=('protein_id', 'size'),
                                       sites=('positive', lambda s: s.str.len().sum()))
      .sort_values('sites', ascending=False).to_string())

proteins.to_parquet('data_processed/atlas5_proteins.parquet', index=False)

# %%
# ---------------------------------------------------------------------------
# Merge proteins with identical sequences
#   Some sequences are reported under several IDs (e.g. a canonical UniProt entry and its
#   isoform 1, or identical entries from two species). The model sees only the sequence, so
#   each group becomes one protein: positive = union of the positives (a residue reported
#   under any ID is positive); masked = union of the masked residues, minus the positives.
#   Kept ID: canonical UniProt accession first, then isoform, then UniParc (UPI), then
#   alphabetical. The group's IDs and organisms are recorded in 'merged_ids' and 'organisms'.
# ---------------------------------------------------------------------------
def id_rank(pid):
    return (pid.startswith('UPI'), '-' in pid, pid)

groups = proteins.groupby('sequence')['protein_id'].apply(list)
groups = groups[groups.str.len() > 1]
by_id = proteins.set_index('protein_id')

# Label agreement before merging: residues positive under one ID but negative under another
conflict = 0
for ids in groups:
    pos_any = set().union(*(set(by_id.at[p, 'positive']) for p in ids))
    conflict += sum(len(pos_any - set(by_id.at[p, 'positive']) - set(by_id.at[p, 'masked'])) for p in ids)
print(f'Groups of identical sequences: {len(groups)}   proteins: {groups.str.len().sum()}   '
      f'groups with >1 organism: {sum(by_id.loc[ids, "organism"].nunique() > 1 for ids in groups)}')
print(f'Residues positive under one ID and negative under another: {conflict}')

kept_of, merged = {}, []
for ids in groups:
    ids = sorted(ids, key=id_rank)
    kept_of.update({p: ids[0] for p in ids})
    r = by_id.loc[ids[0]].to_dict()
    pos = set().union(*(set(by_id.at[p, 'positive']) for p in ids))
    msk = set().union(*(set(by_id.at[p, 'masked']) for p in ids)) - pos
    st = {i + 1 for i, aa in enumerate(r['sequence']) if aa in 'ST'}
    r.update(protein_id=ids[0], positive=sorted(pos), masked=sorted(msk), n_negative=len(st - pos - msk),
             merged_ids='; '.join(ids), organisms='; '.join(sorted(by_id.loc[ids, 'organism'].unique())))
    merged.append(r)

singles = proteins[~proteins['protein_id'].isin(kept_of)].copy()
singles['merged_ids'], singles['organisms'] = singles['protein_id'], singles['organism']
proteins = (pd.concat([singles, pd.DataFrame(merged)], ignore_index=True)
            .sort_values('protein_id').reset_index(drop=True))
assert not proteins['sequence'].duplicated().any()

# Atlas rows now point to the kept ID (flow log: records unchanged, sites and proteins merged)
for d in atlas.values():
    d['protein_id'] = d['protein_id'].map(kept_of).fillna(d['protein_id'])
log_step('Merged: proteins with identical sequences (no rows removed)')
unamb = atlas['unambiguous']
assert proteins['positive'].str.len().sum() == \
    unamb.assign(pos=unamb['pos'].astype(int))[['protein_id', 'pos']].drop_duplicates().shape[0]

n_pos, n_msk, n_neg = proteins['positive'].str.len().sum(), proteins['masked'].str.len().sum(), proteins['n_negative'].sum()
print(f'\nFinal dataset: {len(proteins):,} proteins')
print(f'  positive S/T sites: {n_pos:,}')
print(f'  masked S/T (ambiguous candidates): {n_msk:,}')
print(f'  negative S/T: {n_neg:,}   (positive fraction: {n_pos / (n_pos + n_neg):.2%})')

proteins.to_parquet('data_processed/atlas5_proteins.parquet', index=False)
pd.DataFrame(flow).to_csv('data_processed/atlas5_flow.csv', index=False)

# %%
import subprocess

# MMseqs2 static Linux build (AVX2), as in the MMseqs2 installation instructions
MMSEQS = '/content/mmseqs/bin/mmseqs'
if not os.path.exists(MMSEQS):
    subprocess.run('wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -O /content/mmseqs.tar.gz '
                   '&& tar xzf /content/mmseqs.tar.gz -C /content', shell=True, check=True)
version = subprocess.run([MMSEQS, 'version'], capture_output=True, text=True).stdout.strip()
print('MMseqs2 version:', version)
with open('data_processed/software_versions.txt', 'w') as f:
    f.write(f'MMseqs2 {version}\n')

# %%
# ---------------------------------------------------------------------------
# Exhaustive all-vs-all alignment (homology diagnostic)
#   Every protein is aligned against every other one, bypassing the k-mer prefilter
#   (the "fake prefiltering" procedure of the MMseqs2 user guide). Sequence identity is the
#   number of identical aligned residues over the number of aligned columns, including gaps
#   (--alignment-mode 3). Alignments are kept up to E-value 0.001 with no identity or
#   coverage limit; those thresholds are applied afterwards.
# ---------------------------------------------------------------------------
WORK = '/content/mmseqs_allvsall'
ALLVSALL = 'data_processed/atlas5_allvsall.tsv'
os.makedirs(WORK, exist_ok=True)

def run(cmd):
    """Run a shell command, streaming its output (MMseqs2 prints its own progress)."""
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end='')
    assert p.wait() == 0, f'command failed: {cmd}'

import hashlib

# Checksum of the protein set (IDs and sequences) the cached alignment was computed on
CHECKSUM_FILE = 'data_processed/atlas5_allvsall_proteins.sha256'
checksum = hashlib.sha256('\n'.join(f'{p}\t{s}' for p, s in
                                    sorted(zip(proteins['protein_id'], proteins['sequence']))).encode()).hexdigest()

if not os.path.exists(ALLVSALL):
    with open(f'{WORK}/proteins.fasta', 'w') as f:
        for pid, seq in zip(proteins['protein_id'], proteins['sequence']):
            f.write(f'>{pid}\n{seq}\n')
    run(f'{MMSEQS} createdb {WORK}/proteins.fasta {WORK}/db')
    # Fake prefilter: every query points to the full list of targets
    run(f'ln -sf {WORK}/db.index {WORK}/pref')
    run(f"""awk -v size=$(wc -c < {WORK}/db.index) '{{ print $1"\\t0\\t"size; }}' {WORK}/db.index > {WORK}/pref.index""")
    run(f"""awk 'BEGIN {{ printf("%c%c%c%c",7,0,0,0); exit; }}' > {WORK}/pref.dbtype""")
    run(f'{MMSEQS} align {WORK}/db {WORK}/db {WORK}/pref {WORK}/aln --alignment-mode 3 -e 0.001 '
        f'--min-seq-id 0 -c 0 --max-accept 100000 --threads {os.cpu_count()}')
    run(f'{MMSEQS} convertalis {WORK}/db {WORK}/db {WORK}/aln {ALLVSALL} '
        f'--format-output query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits')
    with open(CHECKSUM_FILE, 'w') as f:
        f.write(checksum)
elif os.path.exists(CHECKSUM_FILE):
    assert open(CHECKSUM_FILE).read().strip() == checksum, \
        'the cached alignment was computed on a different protein set; delete it to recompute'
else:
    # Alignment cached before checksums were recorded: compare protein IDs with the saved split
    split_ids = set(pd.read_csv('data_processed/atlas5_split_graphpart.csv')['AC'].astype(str))
    assert split_ids == set(proteins['protein_id']), \
        'the cached alignment was computed on a different protein set; delete it to recompute'
    with open(CHECKSUM_FILE, 'w') as f:
        f.write(checksum)

aln = pd.read_csv(ALLVSALL, sep='\t', header=None,
                  names=['query', 'target', 'fident', 'alnlen', 'qlen', 'tlen', 'qcov', 'tcov', 'evalue', 'bits'])
aln = aln[aln['query'] != aln['target']].copy()
print(f'\nCPU cores: {os.cpu_count()}   pairs with E <= 0.001 (excluding self-hits): {len(aln):,}')

# %%
# ---------------------------------------------------------------------------
# Split diagnostics 1: connected components under different linking rules
#   Two proteins are linked when identity >= threshold and, if a coverage threshold is set,
#   the alignment covers at least that fraction of BOTH sequences (MMseqs2 --cov-mode 0
#   definition). Connected components = transitive closure of the links.
#   Two identity definitions are compared:
#     - per alignment: identical residues / aligned columns (MMseqs2 --alignment-mode 3);
#     - per shorter protein: identical residues / length of the shorter protein (GraphPart).
#   For each rule: number of components, size of the largest ones, and whether whole
#   components can be packed into 80/10/10 train/val/test.
# ---------------------------------------------------------------------------

def components(edges, nodes):
    """Connected components (union-find); returns protein -> component root."""
    parent = {n: n for n in nodes}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return pd.Series({n: find(n) for n in nodes})

def best_split(sizes, targets=(0.8, 0.1, 0.1)):
    """Greedy packing of components (largest first) into the split furthest below its target."""
    total, filled = sizes.sum(), [0, 0, 0]
    for s in sorted(sizes, reverse=True):
        i = max(range(3), key=lambda k: targets[k] - filled[k] / total)
        filled[i] += s
    return [f / total for f in filled]

aln['id_shortest'] = (aln['fident'] * aln['alnlen']).round() / aln[['qlen', 'tlen']].min(axis=1)
nodes = list(proteins['protein_id'])
n_sites = proteins.set_index('protein_id')['positive'].str.len()
min_cov = aln[['qcov', 'tcov']].min(axis=1)

rows = []
for id_label, id_col in [('per alignment', 'fident'), ('per shorter protein', 'id_shortest')]:
    for identity in (0.3, 0.4):
        for coverage in (None, 0.5, 0.8):
            keep = (aln[id_col] >= identity) & ((min_cov >= coverage) if coverage else True)
            comp = components(zip(aln.loc[keep, 'query'], aln.loc[keep, 'target']), nodes)
            sizes = comp.value_counts()
            sites_largest = n_sites[comp[comp == sizes.index[0]].index].sum()
            train, val, test = best_split(sizes.values)
            rows.append({'identity definition': id_label, 'identity': identity, 'coverage': coverage or 'none',
                         'components': len(sizes), 'singletons': int((sizes == 1).sum()),
                         'largest (proteins)': int(sizes.iloc[0]),
                         'largest (% proteins)': round(100 * sizes.iloc[0] / len(nodes), 1),
                         'largest (% sites)': round(100 * sites_largest / n_sites.sum(), 1),
                         'packed train/val/test (%)': f'{100*train:.0f}/{100*val:.0f}/{100*test:.0f}'})

diag_components = pd.DataFrame(rows)
print(diag_components.to_string(index=False))
diag_components.to_csv('data_processed/atlas5_split_diag_components.csv', index=False)

# %%
# ---------------------------------------------------------------------------
# Near-identical proteins in the dataset (possible duplicates under different IDs)
#   Pairs with >= 95% identity (per shorter protein) whose alignment covers >= 90% of both.
# ---------------------------------------------------------------------------
near = aln[(aln['id_shortest'] >= 0.95) & (min_cov >= 0.9)]
pairs = {tuple(sorted(p)) for p in zip(near['query'], near['target'])}
in_pairs = {p for pair in pairs for p in pair}
identical = proteins['sequence'].duplicated(keep=False)
print(f'Proteins with an identical sequence under another ID: {identical.sum()}')
print(f'Near-identical pairs: {len(pairs)}   proteins involved: {len(in_pairs)}   '
      f'their positive sites: {n_sites[list(in_pairs)].sum():,}')
print(list(pairs)[:10])

# %%
# ---------------------------------------------------------------------------
# Split diagnostics 2: GraphPart under three linking rules
#   Identity per shorter protein (identical residues / length of the shorter protein), 30%.
#   GraphPart moves or removes proteins until no linked pair falls in different splits,
#   aiming at 80/10/10 train/val/test. Rules compared: no coverage requirement, and
#   >= 0.5 / >= 0.8 coverage of both sequences.
# ---------------------------------------------------------------------------
subprocess.run('pip install -q graph-part==1.0.2', shell=True, check=True)
SPLIT_NAMES = {0: 'train', 1: 'test', 2: 'val'}   # GraphPart output codes

GP = '/content/graphpart_diagnostics'
os.makedirs(GP, exist_ok=True)
with open(f'{GP}/proteins.fasta', 'w') as f:
    for pid, seq in zip(proteins['protein_id'], proteins['sequence']):
        f.write(f'>{pid}\n{seq}\n')

results = []
for coverage in (None, 0.5, 0.8):
    keep = (min_cov >= coverage) if coverage else pd.Series(True, index=aln.index)
    edges = aln.loc[keep & (aln['id_shortest'] >= 0.3), ['query', 'target', 'id_shortest']]
    tag = f'cov{coverage or "none"}'
    edges.to_csv(f'{GP}/edges_{tag}.csv', header=False, index=False)
    print(f'\n===== coverage {coverage or "none"}: {len(edges):,} edges >= 30% identity =====')
    run(f'graphpart precomputed --fasta-file {GP}/proteins.fasta --edge-file {GP}/edges_{tag}.csv '
        f'--metric-column 2 --threshold 0.3 --test-ratio 0.1 --val-ratio 0.1 --out-file {GP}/split_{tag}.csv')
    split = pd.read_csv(f'{GP}/split_{tag}.csv')
    split['split'] = split['cluster'].astype(int).map(SPLIT_NAMES)
    split['sites'] = proteins.set_index('protein_id').loc[split['AC'].astype(str), 'positive'].str.len().values
    row = {'coverage': coverage or 'none', 'proteins removed': len(proteins) - len(split)}
    for name in ('train', 'val', 'test'):
        part = split[split['split'] == name]
        row[f'{name} proteins'] = len(part)
        row[f'{name} sites'] = int(part['sites'].sum())
    results.append(row)

diag_graphpart = pd.DataFrame(results)
print('\n', diag_graphpart.to_string(index=False))
diag_graphpart.to_csv('data_processed/atlas5_split_diag_graphpart.csv', index=False)

# %%
# ---------------------------------------------------------------------------
# Split diagnostics 3: size distribution of the connected components
#   Identity per shorter protein >= 30%, for each coverage rule. Saved: the number of
#   components of each size, and the 10 largest components (proteins, positive sites).
# ---------------------------------------------------------------------------
size_tables, top_tables = [], []
for coverage in (None, 0.5, 0.8):
    keep = (aln['id_shortest'] >= 0.3) & ((min_cov >= coverage) if coverage else True)
    comp = components(zip(aln.loc[keep, 'query'], aln.loc[keep, 'target']), nodes)
    sizes = comp.value_counts()
    label = coverage or 'none'

    dist = sizes.value_counts().sort_index().rename_axis('component size').reset_index(name='components')
    dist.insert(0, 'coverage', label)
    size_tables.append(dist)

    top = sizes.head(10).rename_axis('component').reset_index(name='proteins')
    top['sites'] = [n_sites[comp[comp == c].index].sum() for c in top['component']]
    top['% proteins'] = (100 * top['proteins'] / len(nodes)).round(1)
    top.insert(0, 'coverage', label)
    top_tables.append(top.drop(columns='component'))

    print(f'\n===== coverage {label}: {len(sizes):,} components =====')
    print(dist.drop(columns='coverage').to_string(index=False))
    print('\n10 largest components:')
    print(top_tables[-1].drop(columns='coverage').to_string(index=False))

pd.concat(size_tables).to_csv('data_processed/atlas5_split_diag_component_sizes.csv', index=False)
pd.concat(top_tables).to_csv('data_processed/atlas5_split_diag_largest_components.csv', index=False)

# %%
# ---------------------------------------------------------------------------
# Split diagnostics 4: the largest connected component
#   Identity per shorter protein >= 30%, no coverage limit (the rule used for the split).
#   Checks whether the largest component is one protein family or a chain of unrelated
#   proteins joined through weak or short alignments. Lists its proteins (gene and protein
#   names as reported in the Atlas), the identity of its links, and how many links rest on
#   alignments covering < 50% of either protein (typically a domain shared with a longer
#   protein). Saved: one row per protein.
# ---------------------------------------------------------------------------
keep = aln['id_shortest'] >= 0.3
comp = components(zip(aln.loc[keep, 'query'], aln.loc[keep, 'target']), nodes)
largest = comp[comp == comp.value_counts().index[0]].index

# Links inside the component, in both directions (an alignment may be reported only one way)
edges = aln.loc[keep & aln['query'].isin(largest) & aln['target'].isin(largest),
                ['query', 'target', 'id_shortest']].assign(min_cov=min_cov)
both_dirs = pd.concat([edges, edges.rename(columns={'query': 'target', 'target': 'query'})])
pairs = (both_dirs.assign(pair=[tuple(sorted(p)) for p in zip(both_dirs['query'], both_dirs['target'])])
         .groupby('pair')[['id_shortest', 'min_cov']].max())
links = both_dirs.groupby('query').agg(links=('target', 'nunique'), max_identity=('id_shortest', 'max'))

unamb = atlas['unambiguous']
atlas_names = (unamb.groupby('protein_id')[['gene_name', 'protein_name']]
               .agg(lambda s: s.dropna().value_counts().index[0] if s.notna().any() else None))
members = (proteins.set_index('protein_id').loc[largest, ['organism', 'length']]
           .assign(sites=n_sites[largest]).join(atlas_names).join(links)
           .rename_axis('protein_id').sort_values(['protein_name', 'organism']))

print(f'Largest component: {len(largest)} proteins   positive sites: {members["sites"].sum()}   '
      f'linked pairs: {len(pairs)}')
print(f'Identity of the links (quantiles 0/25/50/75/100%): '
      f'{pairs["id_shortest"].quantile([0, .25, .5, .75, 1]).round(2).tolist()}')
print(f'Links whose alignment covers < 50% of either protein: {(pairs["min_cov"] < 0.5).sum()}')
print(f'\nOrganisms: {members["organism"].value_counts().to_dict()}')
print(f'\n{members.to_string()}')

members.to_csv('data_processed/atlas5_split_diag_largest_component_members.csv')

# %%
import hashlib, os, subprocess
import pandas as pd

# ---------------------------------------------------------------------------
# Homology-partitioned train/val/test split (GraphPart, 80/10/10)
#   Proteins are linked when identical residues / length of the shorter protein >= 30%,
#   using the exhaustive all-vs-all alignment (E <= 0.001). GraphPart assigns whole groups
#   of linked proteins to splits and moves or removes proteins until no pair in different
#   splits is linked. The assignment is computed once and saved; later runs reload it,
#   after checking that it was computed on the same protein set (IDs and sequences).
# ---------------------------------------------------------------------------
SPLIT_FILE = 'data_processed/atlas5_split_graphpart.csv'
SPLIT_CHECKSUM = 'data_processed/atlas5_split_graphpart_proteins.sha256'
SPLIT_NAMES = {0: 'train', 1: 'test', 2: 'val'}   # GraphPart output codes

proteins = pd.read_parquet('data_processed/atlas5_proteins.parquet')
aln = pd.read_csv('data_processed/atlas5_allvsall.tsv', sep='\t', header=None,
                  names=['query', 'target', 'fident', 'alnlen', 'qlen', 'tlen', 'qcov', 'tcov', 'evalue', 'bits'])
aln = aln[aln['query'] != aln['target']].copy()
aln['id_shortest'] = (aln['fident'] * aln['alnlen']).round() / aln[['qlen', 'tlen']].min(axis=1)
checksum = hashlib.sha256('\n'.join(f'{p}\t{s}' for p, s in
                                    sorted(zip(proteins['protein_id'], proteins['sequence']))).encode()).hexdigest()

if not os.path.exists(SPLIT_FILE):
    subprocess.run('pip install -q graph-part==1.0.2', shell=True, check=True)
    gp_dir = '/content/graphpart'
    os.makedirs(gp_dir, exist_ok=True)
    with open(f'{gp_dir}/proteins.fasta', 'w') as f:
        for pid, seq in zip(proteins['protein_id'], proteins['sequence']):
            f.write(f'>{pid}\n{seq}\n')
    aln.loc[aln['id_shortest'] >= 0.3, ['query', 'target', 'id_shortest']].to_csv(
        f'{gp_dir}/edges.csv', header=False, index=False)
    subprocess.run(f'graphpart precomputed --fasta-file {gp_dir}/proteins.fasta --edge-file {gp_dir}/edges.csv '
                   f'--metric-column 2 --threshold 0.3 --test-ratio 0.1 --val-ratio 0.1 '
                   f'--out-file {SPLIT_FILE}', shell=True, check=True)
    with open(SPLIT_CHECKSUM, 'w') as f:
        f.write(checksum)
elif os.path.exists(SPLIT_CHECKSUM):
    assert open(SPLIT_CHECKSUM).read().strip() == checksum, \
        'the cached split was computed on a different protein set; delete it to recompute'
else:
    # Split cached before checksums were recorded: its protein IDs must match the dataset
    assert set(pd.read_csv(SPLIT_FILE)['AC'].astype(str)) == set(proteins['protein_id']), \
        'the cached split was computed on a different protein set; delete it to recompute'
    with open(SPLIT_CHECKSUM, 'w') as f:
        f.write(checksum)
print(subprocess.run('pip show graph-part | grep -i version', shell=True, capture_output=True, text=True).stdout.strip())

assignment = pd.read_csv(SPLIT_FILE)
split_of = dict(zip(assignment['AC'].astype(str), assignment['cluster'].astype(int).map(SPLIT_NAMES)))
assert set(split_of) <= set(proteins['protein_id']), 'the split file refers to proteins not in the dataset'
proteins['split'] = proteins['protein_id'].map(split_of)
print(f'Proteins without a split (removed by GraphPart): {proteins["split"].isna().sum()}')

# Independent check: no linked pair (>= 30% identity) across splits
pair_split = aln[['query', 'target']].apply(lambda c: c.map(split_of))
both_assigned = pair_split.notna().all(axis=1)
cross = aln[both_assigned & (pair_split['query'] != pair_split['target'])]
print(f'Highest identity between proteins in different splits: {cross["id_shortest"].max():.3f}')
assert cross['id_shortest'].max() < 0.3

summary = proteins.groupby('split').agg(
    proteins=('protein_id', 'size'),
    positive=('positive', lambda s: s.str.len().sum()),
    masked=('masked', lambda s: s.str.len().sum()),
    negative=('n_negative', 'sum'))
summary['% positive'] = (100 * summary['positive'] / (summary['positive'] + summary['negative'])).round(2)
summary = summary.loc[['train', 'val', 'test']]
print('\n', summary.to_string())

summary.to_csv('data_processed/atlas5_split_summary.csv')
proteins.to_parquet('data_processed/atlas5_proteins.parquet', index=False)

# %%
# ===========================================================================
# External validation: site tables of the five external studies (decision 12b)
#   Studies first published >= 2025-01-01 (Atlas 5.0 covers the literature to the end of 2024).
#   Each table is read as published, one row per reported site and sample; nothing is filtered
#   here. Flags kept for the final notebook: `localized` (site assigned to one residue) and
#   the reported residue, peptide and position in the peptide where the table gives them.
#     Lei 2025       PMID 40245499  Drosophila heads   mmc4.xlsx, sheets "Unambiguous sites" and
#                                                      "Ambiguous sites" (accession, position)
#     Mouse brain    PMID 40885482  mouse brain        mmc3.xlsx, sheet "O-GlcNAc" (localization
#                                                      probability >= 75%; residue given)
#     HCC liver      PMID 41811088  human liver        si_005.xlsx, Table S4 (ptmRS site
#                                                      probability; peptide and residue given)
#     NOTISE (H1)    PMID 41634442  human HEK293T      MOESM3.xlsx, Table S2; "Site ID
#                                                      Constraints" = one position (localized) or
#                                                      "(1 of ...)" (region, not localized)
#     Fu 2026 (H2)   PMID 42153620  human HEK293T,     si_006.xlsx, Tables S5A-C (O-Pair level
#                                   HepG2, Jurkat      1/1b; one row per glycopeptide match)
#   Input folder on Drive: data/external_validation/<study folder>/ (files as downloaded).
# ===========================================================================
EXT_DIR = 'data/external_validation'
ext_rows = []

def add(df, study, sample, organism):
    ext_rows.append(df.assign(study=study, sample=sample, organism=organism))

# Lei et al. 2025, Drosophila heads
lei = f'{EXT_DIR}/pmid40245499_drosophila/1-s2.0-S0968089625001324-mmc4.xlsx'
for sheet, localized in [('Unambiguous sites', True), ('Ambiguous sites', False)]:
    d = pd.read_excel(lei, sheet_name=sheet, header=1)
    d.columns = ['accession', 'position']
    add(d.assign(localized=localized, detail=sheet.split()[0].lower()), 'Lei 2025', 'Drosophila heads',
        'Drosophila melanogaster')

# Mouse brain (PMID 40885482)
d = pd.read_excel(f'{EXT_DIR}/pmid40885482_mouse_brain/mmc3.xlsx', sheet_name='O-GlcNAc')
d = d.rename(columns={'Accession': 'accession', 'Position': 'position', 'Amino acid': 'residue'})
add(d[['accession', 'position', 'residue']].assign(localized=True, detail='localization probability >= 75%'),
    'Mouse brain 2025', 'mouse brain', 'Mus musculus')

# HCC and normal liver (PMID 41811088)
d = pd.read_excel(f'{EXT_DIR}/pmid41811088_hcc_liver/pr5c00939_si_005.xlsx', sheet_name='Table S4')
d = d.rename(columns={'Protein Accession': 'accession', 'Position': 'position', 'Target Amino Acid': 'residue',
                      'Peptide Sequence': 'peptide', 'Position in Peptide': 'position_in_peptide'})
add(d[['accession', 'position', 'residue', 'peptide', 'position_in_peptide']]
    .assign(localized=True, detail='site probability ' + d['Site Probability'].astype(str)),
    'HCC liver 2026', 'human liver (HCC and normal)', 'Homo sapiens')

# NOTISE, HEK293T (PMID 41634442), Table S2
d = pd.read_excel(f'{EXT_DIR}/pmid41634442_notise_hek293t/41589_2025_2108_MOESM3_ESM.xlsx', sheet_name='Table S2',
                  header=1)
constraint = d['Site ID Constraints'].astype(str).str.strip()
d = d.assign(accession=d['Accession'], localized=constraint.str.fullmatch(r'\d+'),
             position=pd.to_numeric(constraint.where(constraint.str.fullmatch(r'\d+')), errors='coerce'),
             detail=constraint)
add(d[['accession', 'position', 'localized', 'detail']], 'NOTISE 2026', 'HEK293T', 'Homo sapiens')

# Fu et al. 2026 (PMID 42153620), Tables S5A-C: one row per glycopeptide-spectrum match
for sheet, sample in [('Sheet1', 'HEK293T'), ('Sheet2', 'HepG2'), ('Sheet3', 'Jurkat')]:
    d = pd.read_excel(f'{EXT_DIR}/pmid42153620_hek293t_hepg2_jurkat/ac6c00972_si_006.xlsx', sheet_name=sheet, header=1)
    d = d.rename(columns={'UniProt Accession': 'accession', 'Site Position': 'position', 'Peptide': 'peptide',
                          'Glycosite Position in Peptide': 'position_in_peptide', 'O-Pair Localization Level': 'detail'})
    d['residue'] = d['Site'].astype(str).str[0]
    add(d[['accession', 'position', 'residue', 'peptide', 'position_in_peptide', 'detail']].assign(localized=True),
        'Fu 2026', sample, 'Homo sapiens')

ext_sites = pd.concat(ext_rows, ignore_index=True)
ext_sites['accession'] = ext_sites['accession'].astype(str).str.strip()
ext_sites['position'] = pd.to_numeric(ext_sites['position'], errors='coerce').astype('Int64')
ext_sites = ext_sites[['study', 'sample', 'organism', 'accession', 'position', 'residue', 'peptide',
                       'position_in_peptide', 'localized', 'detail']]

# One row per study, sample and site (Fu 2026 lists every glycopeptide match; the first match is kept)
n_rows = len(ext_sites)
repeated = (ext_sites['study'] == 'Fu 2026') & ext_sites.duplicated(['study', 'sample', 'accession', 'position'])
ext_sites = ext_sites[~repeated].reset_index(drop=True)
ext_sites.to_parquet('data_processed/external_sites_raw.parquet', index=False)

print(f'Rows read: {n_rows:,}   after removing repeated matches of the same site: {len(ext_sites):,}')
print('Accession fields with more than one accession:',
      ext_sites.loc[ext_sites['accession'].str.contains(';'), ['study', 'accession']].values.tolist())
summary = (ext_sites.groupby(['study', 'sample'], sort=False)
           .agg(sites=('position', 'size'), localized=('localized', 'sum'), proteins=('accession', 'nunique'))
           .reset_index())
print()
print(summary.to_string(index=False))
print('\nReported residues (where given):', ext_sites['residue'].value_counts().to_dict())

# %%
import json, re, time, urllib.error, urllib.parse, urllib.request
from tqdm.auto import tqdm

# ===========================================================================
# External validation: protein sequences the studies' positions refer to (decision 12b)
#   For each study and accession, the sequence must agree with every site the study reports on
#   that protein: the residue at each position is the reported residue (S, T or Y where the
#   table gives it; S, T or Y otherwise) and, where a peptide is given, the peptide occurs at
#   (position - position in peptide + 1). Candidate sequences, in this order:
#     1. the current UniProtKB entry (secondary accessions resolve to their current entry);
#     2. earlier versions of the entry from UniSave (UniProt's archive of entry versions),
#        newest sequence version first, then older ones (also for entries deleted from UniProt).
#   The first candidate that agrees with all of the study's sites on that protein is used.
#   Accession fields listing several proteins ("A; B") try each accession in turn.
#   Identical sequences share one identifier (EXT00001, ...); the tools score every S/T of every
#   sequence in data_processed/external_proteins.parquet.
# ===========================================================================
def get(url, timeout=120, attempts=4):
    """Download a URL; network errors, timeouts and rate limits (HTTP 429) are retried."""
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read().decode()
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == attempts - 1:
                raise
            time.sleep(10 * (attempt + 1))
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(10 * (attempt + 1))

def fasta_records(text):
    """(header, sequence) pairs of a FASTA text."""
    records = []
    for block in text.split('>')[1:]:
        header, *lines = block.strip().split('\n')
        records.append((header, ''.join(l.strip() for l in lines)))
    return records

ext_sites = pd.read_parquet('data_processed/external_sites_raw.parquet')
ext_sites['candidates'] = ext_sites['accession'].str.split(r'\s*;\s*')
accessions = sorted({a for c in ext_sites['candidates'] for a in c})

# 1. Current UniProtKB entries: canonical accessions in batches, isoforms one by one
current = {}
canonical = [a for a in accessions if re.fullmatch(r'[A-Z0-9]{6,10}', a)]
for i in tqdm(range(0, len(canonical), 200), desc='UniProt (current entries)'):
    text = get('https://rest.uniprot.org/uniprotkb/accessions?format=fasta&accessions=' + ','.join(canonical[i:i + 200]))
    for header, seq in fasta_records(text):
        current[header.split('|')[1]] = seq
for a in tqdm([a for a in accessions if a not in current], desc='UniProt (isoforms, secondary accessions)'):
    try:
        records = fasta_records(get(f'https://rest.uniprot.org/uniprotkb/{urllib.parse.quote(a)}.fasta', 60))
        if records:
            current[a] = records[0][1]
    except urllib.error.HTTPError:
        pass

# 2. UniSave versions, fetched only when needed (cached per accession)
unisave_cache = {}
def unisave_sequences(acc):
    """Distinct sequence versions of an entry, newest first: [(label, sequence), ...]."""
    if acc not in unisave_cache:
        out = []
        try:
            entries = json.loads(get(f'https://rest.uniprot.org/unisave/{acc}?format=json', 60))['results']
            newest = {}   # sequence version -> newest entry version carrying it
            for e in entries:
                sv, ev = e.get('sequenceVersion'), e['entryVersion']
                if sv is not None and ev > newest.get(sv, -1):
                    newest[sv] = ev
            for sv in sorted(newest, reverse=True):
                records = fasta_records(get(f'https://rest.uniprot.org/unisave/{acc}?format=fasta&versions={newest[sv]}', 60))
                if records:
                    out.append((f'UniSave entry version {newest[sv]} (sequence version {sv})', records[0][1]))
        except urllib.error.HTTPError:
            pass
        unisave_cache[acc] = out
    return unisave_cache[acc]

def agrees(seq, group):
    """True if the sequence agrees with every positioned site of a study on one protein."""
    for pos, res, pep, pip in zip(group['position'], group['residue'], group['peptide'], group['position_in_peptide']):
        if pd.isna(pos):
            continue
        pos = int(pos)
        if not 1 <= pos <= len(seq):
            return False
        allowed = res if isinstance(res, str) and res else 'STY'
        if seq[pos - 1] not in allowed:
            return False
        if isinstance(pep, str) and pep and not pd.isna(pip):
            start = pos - int(pip)
            if start < 0 or seq[start:start + len(pep)] != pep.upper():
                return False
    return True

resolved = []   # one row per study and accession field
groups = list(ext_sites.groupby(['study', 'accession'], sort=False))
for (study, acc_field), group in tqdm(groups, desc='Matching sequences to reported sites'):
    choice = None
    for acc in group['candidates'].iloc[0]:
        options = [('UniProtKB current', current[acc])] if acc in current else []
        for label, seq in options:
            if agrees(seq, group):
                choice = (acc, label, seq)
                break
        if choice is None:
            for label, seq in unisave_sequences(acc):
                if agrees(seq, group):
                    choice = (acc, label, seq)
                    break
        if choice:
            break
    resolved.append({'study': study, 'accession': acc_field, 'resolved_accession': choice[0] if choice else None,
                     'sequence_source': choice[1] if choice else 'not found', 'sequence': choice[2] if choice else None})
resolved = pd.DataFrame(resolved)

# Identical sequences share one identifier
seqs = resolved.dropna(subset=['sequence'])
unique = seqs.drop_duplicates('sequence')[['sequence']].reset_index(drop=True)
unique['seq_id'] = [f'EXT{i + 1:05d}' for i in range(len(unique))]
resolved = resolved.merge(unique, on='sequence', how='left')
organism_of = ext_sites.drop_duplicates(['study', 'accession']).set_index(['study', 'accession'])['organism']
resolved['organism'] = [organism_of[(s, a)] for s, a in zip(resolved['study'], resolved['accession'])]

ext_proteins = (resolved.dropna(subset=['seq_id'])
                .groupby('seq_id').agg(sequence=('sequence', 'first'),
                                       accessions=('resolved_accession', lambda x: sorted(set(x))),
                                       organisms=('organism', lambda x: sorted(set(x))),
                                       studies=('study', lambda x: sorted(set(x)))).reset_index())
ext_proteins['length'] = ext_proteins['sequence'].str.len()
ext_proteins.to_parquet('data_processed/external_proteins.parquet', index=False)

ext_sites = ext_sites.drop(columns='candidates').merge(
    resolved[['study', 'accession', 'resolved_accession', 'sequence_source', 'seq_id']], on=['study', 'accession'], how='left')
ext_sites.to_parquet('data_processed/external_sites_mapped.parquet', index=False)

# Summary
resolved['source'] = resolved['sequence_source'].str.replace(r' \(.*|entry version \d+', '', regex=True).str.strip()
print('Sequence used per study and accession:')
print(pd.crosstab(resolved['study'], resolved['source']).to_string())
not_found = resolved[resolved['seq_id'].isna()]
print(f'\nNot found ({len(not_found)} study-accession pairs; their sites are left without a sequence):')
print(not_found.groupby('study')['accession'].apply(lambda x: ', '.join(x[:8]) + (' ...' if len(x) > 8 else '')).to_string())
lost = ext_sites['seq_id'].isna()
print(f'\nSites without a sequence: {lost.sum():,} of {len(ext_sites):,}')
n_st = ext_proteins['sequence'].str.count('[ST]')
print(f'External proteins: {len(ext_proteins):,} distinct sequences, {ext_proteins["length"].sum():,} residues, '
      f'{n_st.sum():,} S/T; longer than 1,021 residues: {(ext_proteins["length"] > 1021).sum():,} '
      f'({n_st[ext_proteins["length"] > 1021].sum():,} S/T)')

# %%
import os
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))
import pandas as pd

# ---------------------------------------------------------------------------
# Dataset per species: proteins, S/T sites (masked residues excluded) and positive sites, in the whole
# dataset (Atlas 5.0 as processed above) and in the training, validation and test splits.
#   Species: the organism of each protein (for sequences reported under several species and merged
#   above, the organism of the kept ID). Output: dataset_by_species.csv.
# ---------------------------------------------------------------------------
proteins = pd.read_parquet('data_processed/atlas5_proteins.parquet')
proteins['st_sites'] = [sum(aa in 'ST' for aa in seq) - len(masked)
                        for seq, masked in zip(proteins['sequence'], proteins['masked'])]
proteins['positive_sites'] = proteins['positive'].str.len()

SETS = {'All': proteins, 'Training': proteins[proteins['split'] == 'train'],
        'Validation': proteins[proteins['split'] == 'val'], 'Test': proteins[proteins['split'] == 'test']}
by_species = pd.concat({name: d.groupby('organism').agg(proteins=('protein_id', 'size'), sites=('st_sites', 'sum'),
                                                        positive=('positive_sites', 'sum'))
                        for name, d in SETS.items()}, axis=1).fillna(0).astype(int)
by_species = by_species.sort_values(('All', 'positive'), ascending=False)
by_species.loc['Total'] = by_species.sum()
by_species.to_csv('data_processed/dataset_by_species.csv')

pd.set_option('display.width', 250)
print(f'Species: {len(by_species) - 1}   (sites: S/T, masked residues excluded)\n')
print(by_species.to_string(formatters={c: '{:,}'.format for c in by_species.columns}))
