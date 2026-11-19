# Conservation of the LDB3 sites T399 and T400

Best local alignment (Biopython `PairwiseAligner`, BLOSUM62, gap open -10, extend -1) of human LDB3
residues 378 to 414 (UniProt O75112, canonical isoform) against every LDB3 sequence in UniProt of several
vertebrates, keeping the best-scoring sequence of each species. Used for the conservation sentence of the
LDB3 case study (Results).

Inputs, downloaded from UniProt on 2026-11-17 (CC BY 4.0):

- `O75112.fasta`: `https://rest.uniprot.org/uniprotkb/O75112.fasta`
- `orth_<taxon>.fasta`: `https://rest.uniprot.org/uniprotkb/stream?query=(gene:ldb3+OR+gene:ldb3a+OR+gene:ldb3b)+AND+organism_id:<taxon>&format=fasta`
  for mouse (10090), cow (9913), pig (9823), dog (9615), platypus (9258), chicken (9031),
  *Xenopus tropicalis* (8364) and zebrafish (7955).

Run `python motif_conservation.py` in this folder; `motif_conservation_output_2026-11-19.txt` is its output.
