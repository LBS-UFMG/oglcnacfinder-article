# O-GlcNAcFinder: analysis code of the article

Code of every step of the study that built and evaluated O-GlcNAcFinder, a predictor of O-GlcNAc
sites on serines and threonines from ESM-2 embeddings, from the download of the data to the figures
and tables of the article. Laboratory of Bioinformatics and Systems (LBS), UFMG.

- Tool (Python package, command-line tool, trained model, web-server worker):
  [LBS-UFMG/oglcnacfinder](https://github.com/LBS-UFMG/oglcnacfinder)
- Web server: <https://bioinfo.dcc.ufmg.br/oglcnacfinder/>
- Article: *to be added at publication*
- Data (Zenodo): <https://doi.org/10.5281/zenodo.24379893>
- This code (Zenodo): <https://doi.org/10.5281/zenodo.24378941>; the tool: <https://doi.org/10.5281/zenodo.24378501>

## Contents

| Folder | Content |
|---|---|
| `analysis/` | the 19 analysis scripts, run in order (below), and `ldb3_conservation/` (alignment of the LDB3 case study) |
| `scripts/` | two scripts run outside Colab, on a GPU or CPU server, for the long runs of two predictors |
| `other_tools/` | search for published O-GlcNAc predictors and builders of Supplementary Tables S3, S4, S6 and S7 |
| `figures/` | sources of Figure 1 (draw.io) and of the Figure 12 screenshots |
| `results/` | small result tables (compute times, grid ranking, paired tests) |
| `third_party/` | `assemble.py`, which builds the scripts that use code without a license (below) |

The analysis scripts were written and run as Google Colab notebooks and converted to Python scripts in
the percent format: every `# %%` line starts a notebook cell, so the scripts can be run cell by cell
in VS Code, Spyder or Jupyter (with jupytext), or as plain scripts. The code is the code that was run;
only Colab-specific lines were removed (mounting Google Drive, flushing it and releasing the runtime).

## Setup

1. Download the data record from Zenodo (DOI above) into a folder, and set the environment variable
   `OGLCNACFINDER_DATA` to that folder. Each script starts with
   `os.chdir(os.environ.get('OGLCNACFINDER_DATA', '.'))` and reads and writes paths relative to it
   (`data/`, `data_processed/`, `results/`).
2. Third-party inputs that are not redistributed: the supplementary data of the five external studies
   (`data/external_validation/`). Their sources and SHA-256 checksums are listed in the data record;
   download each file from its publisher and check the checksum. Other inputs (UniProt, NCBI, RGAP 7,
   MMseqs2, the predictors' repositories) are downloaded by the scripts themselves.
3. Assemble the scripts that use code of LM-OGlcNAc-Site and Stack-OglyPred-PLM (needs git):

   ```bash
   python third_party/assemble.py
   ```

4. The scripts use `/content` as local scratch space (cloned repositories, the MMseqs2 binary,
   temporary files), as on Colab. On another machine, create a writable `/content` folder
   (e.g. `sudo mkdir /content && sudo chown $USER /content`) or change those paths.

## Run order

Times and hardware are those recorded by the scripts (Supplementary Table S9).

| Script | Step | Hardware |
|---|---|---|
| `01_data_pipeline.py` | Atlas 5.0 sites matched to protein sequences, labels (positive, negative, masked), MMseqs2 + GraphPart split into training, validation and test sets, external studies mapped | CPU |
| `02_extract_embeddings.py` | ESM-2 (8M, 35M, 150M, 650M) per-residue embeddings and classifier features | GPU |
| `03_model_training.py` | grid of 672 runs (ESM-2 size, window, hidden units, dropout, learning rate) | GPU |
| `04_model_selection.py` | model selection on the validation set (paired tests), grid figures | CPU |
| `05_benchmark_own_model.py` | final model, Platt scaling and cut-off on the validation set, scores on the test set and the external studies | GPU |
| `06_benchmark_LM_OGlcNAc_Site.py` | LM-OGlcNAc-Site on the test set (assembled, see below) | GPU |
| `07_benchmark_O-GlcNAcPRED-DL.py` | O-GlcNAcPRED-DL on the test set and external studies | GPU |
| `08_benchmark_Seber.py` | RNN-225 and RNN-[600,75] on the test set and external studies | GPU |
| `09_benchmark_YinOYang.py` | YinOYang 1.2: submission files for its web server (the submissions failed, so YinOYang was not compared; Supplementary Table S3) | CPU |
| `10_benchmark_Stack-OglyPred-PLM.py` | Stack-OglyPred-PLM on the test set and external studies (assembled) | GPU |
| `11_benchmark_old_Arcanjo_et_al.py` | our earlier model (Arcanjo et al. 2025) | GPU |
| `12_external_validation.py` | evaluation on the five external studies, top-*k* hit rate | CPU |
| `13_case_studies.py` | case studies (YEATS2, LDB3 and others) | GPU |
| `14_proteome_predictions.py` | scores for five UniProt reference proteomes, candidate new sites | GPU |
| `15_benchmark_summary.py` | comparison with the published predictors, figures | CPU |
| `16_tool_reproducibility.py` | check of the O-GlcNAcFinder package against the proteome predictions | GPU (the package) |
| `17_website_data.py` | data files of the web server's proteome browser | CPU |
| `18_literature_search.py` | PubMed records per year (Supplementary Figure S2) | CPU |
| `19_zenodo_record.py` | builds the Zenodo data record: one zip per group of files, manifest and checksums, README | CPU |

`scripts/lm_oglcnac_site_external.py` (LM-OGlcNAc-Site on the external studies, 38.7 h on an A100)
and `scripts/stack_oglypred_plm_long_proteins.py` (Stack-OglyPred-PLM on proteins too long for the
GPU) are run on a server, as described in their docstrings, with inputs prepared by scripts 06 and 10.

## Environment

The scripts ran on Google Colab (Python 3, A100 GPU for the GPU steps) between October and November
2026. There is no single environment file, because the predictors need incompatible library
versions (for example, TensorFlow with Keras 2 for LM-OGlcNAc-Site, and transformers 4.57.6 for Ankh).
Each script installs the extra packages it needs at fixed versions and prints the versions it runs
with; `01_data_pipeline.py` writes the versions of the main libraries to
`data_processed/software_versions.txt` (included in the data record).

## Third-party code

LM-OGlcNAc-Site and Stack-OglyPred-PLM have no license, so their code is not stored here. The four
scripts that run their code with our changes (`analysis/06`, `analysis/10` and the two in `scripts/`)
are kept as templates (`*.template.py`): every line taken from those repositories is replaced by a
reference to its file and line, and our changes are tagged `[modified]`, `[added]` or `[removed]`.
`third_party/assemble.py` clones both repositories at the commits used in the study, fills in the
references and checks that each assembled script is identical (SHA-256) to the script that was run.
The other predictors are cloned and run by the scripts; see `THIRD_PARTY_NOTICES`.

## Funding

This work was supported by the Fundação de Amparo à Pesquisa do Estado de Minas Gerais (FAPEMIG) and the
Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq). This study was financed in part by the
Coordenação de Aperfeiçoamento de Pessoal de Nível Superior – Brasil (CAPES) – Finance Code 001.

## License

MIT license, following the LBS Software Policy (<https://bioinfo.dcc.ufmg.br/software-policy>); see
`LICENSE`.

## Citation

*To be added at publication.*
