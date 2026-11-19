# %%
import os, hashlib
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))

def size(p):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(p) for f in fs) if os.path.isdir(p) else os.path.getsize(p)

# Candidate folders and files for the data record: one line per entry, size in MB
for top in ['data', 'data_processed']:
    for name in sorted(os.listdir(top)):
        p = f'{top}/{name}'
        print(f'{size(p) / 1e6:10.1f} MB  {p}{"/" if os.path.isdir(p) else ""}')
for sub in ['data_processed/other_tools', 'data_processed/proteome_predictions', 'data_processed/external_validation',
            'data_processed/final_model', 'data/external_validation']:
    if os.path.isdir(sub):
        print(f'\n{sub}/')
        for name in sorted(os.listdir(sub)):
            print(f'{size(f"{sub}/{name}") / 1e6:10.1f} MB    {name}')

# SHA-256 of the external studies' supplementary files (not redistributed; checksums go in the record)
print('\nSHA-256, data/external_validation:')
for d, _, fs in os.walk('data/external_validation'):
    for f in sorted(fs):
        p = os.path.join(d, f)
        print(hashlib.sha256(open(p, 'rb').read()).hexdigest(), p)

# %%
import os
from collections import Counter
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))

for top in ['data_processed/other_tools', 'data_processed/training', 'data_processed/website',
            'data_processed/proteome_predictions/reference_proteomes', 'data_processed/proteome_predictions/chunks',
            'data_processed/competitors', 'data_processed/benchmark_summary', 'data_processed/literature']:
    print(f'\n{top}/')
    small = Counter()
    for d, _, fs in sorted(os.walk(top)):
        for f in sorted(fs):
            p = os.path.join(d, f)
            s = os.path.getsize(p)
            if s >= 1e6:
                print(f'{s / 1e6:10.1f} MB  {os.path.relpath(p, top)}')
            else:
                small[os.path.relpath(d, top)] += 1
    for d, n in sorted(small.items()):
        print(f'{"":13s} {n} small files in {d}/')

# %%
import glob
import os
os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))

# ---------------------------------------------------------------------------
# Zenodo data record of the article: file manifest.
#   Each group of files becomes one zip archive in the record (Zenodo accepts at
#   most 100 files per record). File lists follow release/zenodo_record_plan.md.
#   This cell only lists and checks the files; nothing is copied or written.
# ---------------------------------------------------------------------------
RECORD_VERSION = '0.26.1119'
OUT_DIR = f'zenodo/data_record_{RECORD_VERSION}'
DP = 'data_processed'
OT = f'{DP}/other_tools'
PP = f'{DP}/proteome_predictions'


def files_in(folder, recursive=True):
    """Every file under folder (or only directly in it), as sorted relative paths."""
    pattern = f'{folder}/**/*' if recursive else f'{folder}/*'
    return sorted(p for p in glob.glob(pattern, recursive=recursive) if os.path.isfile(p))


def explicit(*paths):
    """Paths that must exist; a missing one stops the cell."""
    missing = [p for p in paths if not os.path.isfile(p)]
    assert not missing, f'missing files: {missing}'
    return list(paths)


def matching(*patterns):
    """Files matching glob patterns; a pattern with no match stops the cell."""
    out = []
    for pattern in patterns:
        hits = sorted(p for p in glob.glob(pattern) if os.path.isfile(p))
        assert hits, f'no file matches {pattern}'
        out += hits
    return out


# Group 4: files directly in each tool folder (subfolders excluded), except our
# rebuild of another tool's training proteins
TOOL_TOP_LEVEL_EXCLUDED = {f'{OT}/Stack-OglyPred-PLM/their_training_proteins_uniprot.fasta'}
tool_top_level = [p for tool in ['LM-OGlcNAc-Site', 'Stack-OglyPred-PLM', 'O-GlcNAcPRED-DL', 'Seber']
                  for p in files_in(f'{OT}/{tool}', recursive=False) if p not in TOOL_TOP_LEVEL_EXCLUDED]

SPECIES_SP = ['human', 'mouse']   # Swiss-Prot canonical subset used for the candidate new sites
GROUPS = {
    '1_atlas_files': explicit(
        'data/Atlas 5.0_unambiguous sites_20251208.csv',
        'data/Atlas 5.0_ambiguous sites_20251208.csv',
        f'{OT}/Atlas4.0/Atlas 4.0_unambiguous sites_20241217.csv',
        f'{OT}/Atlas4.0/Atlas 4.0_ambiguous sites_20241217.csv'),
    '2_processed_dataset': explicit(
        f'{DP}/atlas5_proteins.parquet',
        f'{DP}/atlas5_split_graphpart.csv', f'{DP}/atlas5_split_summary.csv',
        f'{DP}/atlas5_split_graphpart_report.json',
        f'{DP}/atlas5_allvsall_proteins.sha256', f'{DP}/atlas5_split_graphpart_proteins.sha256',
        f'{DP}/atlas5_allvsall.tsv', f'{DP}/atlas5_flow.csv', f'{DP}/dataset_by_species.csv',
        f'{DP}/external_proteins.parquet', f'{DP}/external_sites_raw.parquet',
        f'{DP}/external_sites_mapped.parquet', f'{DP}/software_versions.txt')
        + matching(f'{DP}/atlas5_split_diag_*.csv'),
    '3_model_outputs': files_in(f'{DP}/final_model') + files_in(f'{DP}/external_validation'),
    '4_other_tools_outputs': files_in(f'{OT}/BSB2025') + tool_top_level
        + explicit(f'{OT}/Atlas4.0/atlas4_proteins.fasta', f'{OT}/Atlas4.0/test_vs_atlas4_alignments.tsv'),
    '5_proteome_predictions': files_in(f'{PP}/reference_proteomes', recursive=False)
        + matching(*[f'{PP}/{sp}_{suffix}' for sp in SPECIES_SP for suffix in [
            'proteome_predictions.parquet', 'proteins.parquet', 'proteome_protein_summary.csv',
            'swissprot_canonical.fasta', 'swissprot_canonical.json', 'uniprot_fields.tsv',
            'uniprot_fields.json', 'new_site_candidates.parquet', 'new_site_candidates_p0.*.csv']],
                   f'{PP}/probability_bins_vs_atlas_labels.csv',
                   f'{PP}/probability_equal_count_bins_vs_atlas_labels.csv',
                   f'{DP}/website/proteome_*.tsv.gz'),
    '6_training_grid': files_in(f'{DP}/training/grid') + files_in(f'{DP}/training/grid_h512'),
    '7_case_studies_and_summaries': files_in(f'{DP}/case_studies') + files_in(f'{PP}/case_study_annotation')
        + files_in(f'{DP}/benchmark_summary') + files_in(f'{DP}/literature'),
}

# Summary: files and size of each group
all_files = [p for files in GROUPS.values() for p in files]
assert len(all_files) == len(set(all_files)), 'a file is listed in more than one group'
for name, files in GROUPS.items():
    print(f'{name:30s} {len(files):6d} files  {sum(map(os.path.getsize, files)) / 1e6:9.1f} MB')
print(f'{"total":30s} {len(all_files):6d} files  {sum(map(os.path.getsize, all_files)) / 1e6:9.1f} MB')

# Files taken by folder rather than by name, listed for review
REVIEW = {'tool folders, top level': tool_top_level,
          'reference proteomes, top level': files_in(f'{PP}/reference_proteomes', recursive=False)}
for title, files in REVIEW.items():
    print(f'\nFor review, {title}:')
    for p in files:
        print(f'{os.path.getsize(p) / 1e6:10.2f} MB  {p}')

# %%
import hashlib
import shutil
import zipfile
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Build the record: one zip per group (paths inside the zips are the project paths),
# a manifest with the SHA-256 of every file, and the SHA-256 of every zip.
# Built on the local disk of the Colab VM, then copied to OUT_DIR on Drive.
# ---------------------------------------------------------------------------
def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


LOCAL = f'/content/{os.path.basename(OUT_DIR)}'
shutil.rmtree(LOCAL, ignore_errors=True)
os.makedirs(LOCAL)

manifest = []
for name, files in GROUPS.items():
    with zipfile.ZipFile(f'{LOCAL}/{name}.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in tqdm(files, desc=name, unit='file'):
            manifest.append((name, p, os.path.getsize(p), sha256(p)))
            z.write(p, arcname=p)
    with zipfile.ZipFile(f'{LOCAL}/{name}.zip') as z:   # integrity check of the archive just written
        assert z.testzip() is None and len(z.namelist()) == len(files), name

with open(f'{LOCAL}/MANIFEST.tsv', 'w', encoding='utf-8') as f:
    f.write('archive\tpath\tbytes\tsha256\n')
    f.writelines(f'{a}.zip\t{p}\t{n}\t{h}\n' for a, p, n, h in manifest)

with open(f'{LOCAL}/SHA256SUMS.txt', 'w', encoding='utf-8') as f:
    for name in GROUPS:
        f.write(f'{sha256(f"{LOCAL}/{name}.zip")}  {name}.zip\n')

# Files used but not redistributed: the supplementary files of the five external studies
with open(f'{LOCAL}/external_study_files_sha256.txt', 'w', encoding='utf-8') as f:
    for p in files_in('data/external_validation'):
        f.write(f'{sha256(p)}  {p}\n')

os.makedirs(OUT_DIR, exist_ok=True)
for fname in sorted(os.listdir(LOCAL)):
    shutil.copy2(f'{LOCAL}/{fname}', f'{OUT_DIR}/{fname}')
    print(f'{os.path.getsize(f"{OUT_DIR}/{fname}") / 1e6:10.1f} MB  {OUT_DIR}/{fname}')
print(f'\nfiles in the archives: {len(manifest)}')
print('SHA-256 of data/msu7_all.pep (not redistributed):', sha256('data/msu7_all.pep'))

# %%
README = """# O-GlcNAcFinder: dataset, model outputs and proteome predictions

Version 0.26.1119. Data of the article that describes O-GlcNAcFinder, submitted for publication (Arcanjo et al.).

Authors: Adenilson Arcanjo, Leonardo Henrique da Silva, Fernanda Coura Pena de Sousa, Raquel C. de Melo-Minardi.

Related records:
- O-GlcNAcFinder (tool: Python package, command-line tool, trained model): https://doi.org/10.5281/zenodo.24378501,
  code at https://github.com/LBS-UFMG/oglcnacfinder
- Analysis code of the article, which produced every file below: https://doi.org/10.5281/zenodo.24378941,
  code at https://github.com/LBS-UFMG/oglcnacfinder-article
- Web server and proteome browser: https://bioinfo.dcc.ufmg.br/oglcnacfinder/

## Contents

Each archive keeps the paths of the analysis code (`data/...`, `data_processed/...`), so the scripts of the
article repository find the files when the archives are extracted in its working folder.

| Archive | Contents |
|---|---|
| `1_atlas_files.zip` | O-GlcNAcAtlas 5.0 site files (release 5.0, files dated 8 December 2025) used for training and evaluation, and the O-GlcNAcAtlas 4.0 site files used to filter the test set for tools trained on earlier Atlas versions. |
| `2_processed_dataset.zip` | Processed dataset (proteins, sequences and labels), homology-aware split into training, validation and test sets (GraphPart, 30% identity) with its diagnostics, all-versus-all MMseqs2 alignments, record counts of each processing step, and the O-GlcNAc sites of the five external studies mapped to UniProt. |
| `3_model_outputs.zip` | Trained model (`final_model.pt`), scores of every serine and threonine of the validation, test and external sets and of the Atlas proteins, metrics, top-k hit rates, and the alignments used to remove proteins similar to the training data of each tool. |
| `4_other_tools_outputs.zip` | Scores of the compared tools (LM-OGlcNAc-Site, Stack-OglyPred-PLM, O-GlcNAcPRED-DL, the two RNN models of Seber and Braatz, and the earlier model of Arcanjo et al. 2025) on the same residues, their installation checks, metrics and provenance (repository, commit, settings). |
| `5_proteome_predictions.zip` | Predictions for every serine and threonine of the UniProt reference proteomes of *Homo sapiens*, *Mus musculus*, *Arabidopsis thaliana*, *Oryza sativa* and *Drosophila melanogaster* (Swiss-Prot and TrEMBL), the sequences scored, UniProt annotations, candidate new sites, the human and mouse Swiss-Prot canonical subset, and the same predictions as tab-separated text (`data_processed/website/proteome_*.tsv.gz`). |
| `6_training_grid.zip` | The 672 runs of the model-selection grid: weights, training history and metrics of each run. |
| `7_case_studies_and_summaries.zip` | Case-study scores and annotations (YEATS2, LDB3), the UniProt entries retrieved for the candidate shortlist, and summary tables of the comparisons. |
| `MANIFEST.tsv` | Every file in the archives, with its archive, size and SHA-256. |
| `SHA256SUMS.txt` | SHA-256 of each archive. Check with `sha256sum -c SHA256SUMS.txt`. |
| `external_study_files_sha256.txt` | SHA-256 of the supplementary files of the five external studies, which are not redistributed (see below). |

## Not included

- Protein language model embeddings and classifier features of our model and of the compared tools (about
  170 GB). The analysis code computes them again.
- Files of the external studies and of the compared tools that are not ours to redistribute: the supplementary
  files of the five external studies (download them from the articles; `external_study_files_sha256.txt` lists the
  versions used), the code, models and training data of the compared tools (the analysis code downloads each
  tool from its repository at a fixed commit, or from its web page), and the rice proteins of the Rice Genome Annotation Project,
  release 7 (`msu7_all.pep`, from https://rice.uga.edu/pub/data/Eukaryotic_Projects/o_sativa/annotation_dbs/,
  SHA-256 MSU7_SHA256).

## Sources and licences

This record is released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0). It
contains or derives from:

- O-GlcNAcAtlas (https://oglcnac.org/), CC BY 4.0. Hou C, Li W, Li Y, Ma J (2025). O-GlcNAcAtlas 4.0: an
  updated protein O-GlcNAcylation database with site-specific quantification. Journal of Molecular Biology 437.
  https://doi.org/10.1016/j.jmb.2025.169033. Release 5.0 has no separate article.
- UniProt (https://www.uniprot.org/), CC BY 4.0. The UniProt Consortium (2025). UniProt: the Universal Protein
  Knowledgebase in 2025. Nucleic Acids Research 53. https://doi.org/10.1093/nar/gkae1010
- The O-GlcNAc sites of five studies, mapped to UniProt from their supplementary tables:
  - Lei C et al. (2025). Bioorganic & Medicinal Chemistry 124. https://doi.org/10.1016/j.bmc.2025.118191 (PMID 40245499)
  - Hou C et al. (2025). Molecular & Cellular Proteomics. https://doi.org/10.1016/j.mcpro.2025.101064 (PMID 40885482)
  - Griffin ME et al. (2026). Nature Chemical Biology 22. https://doi.org/10.1038/s41589-025-02108-7 (PMID 41634442)
  - Hou C et al. (2026). Journal of Proteome Research 25. https://doi.org/10.1021/acs.jproteome.5c00939 (PMID 41811088)
  - Fu L et al. (2026). Analytical Chemistry 98. https://doi.org/10.1021/acs.analchem.6c00972 (PMID 42153620)

## Funding

This work was supported by the Fundação de Amparo à Pesquisa do Estado de Minas Gerais (FAPEMIG) and the
Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq). This study was financed in part by the
Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES) - Finance Code 001.
"""

README = README.replace('MSU7_SHA256', sha256('data/msu7_all.pep'))
for folder in (LOCAL, OUT_DIR):
    with open(f'{folder}/README.md', 'w', encoding='utf-8') as f:
        f.write(README)
print(open(f'{OUT_DIR}/README.md', encoding='utf-8').read())
print(sorted(os.listdir(OUT_DIR)))

# %%
# MD5 of the record files on Drive, to compare with the MD5 shown by Zenodo after upload
for fname in sorted(os.listdir(OUT_DIR)):
    h = hashlib.md5()
    with open(f'{OUT_DIR}/{fname}', 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    print(h.hexdigest(), fname)
