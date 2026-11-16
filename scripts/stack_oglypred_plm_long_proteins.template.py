"""Stack-OglyPred-PLM on the external proteins too long to embed on the GPU, on the CPU.

Stack-OglyPred-PLM (Pakhrin lab; Bioinformatics 2024, doi:10.1093/bioinformatics/btae643;
github.com/PakhrinLab/Stack-OglyPred-PLM, commit f9934d2), nucleocytoplasmic model. NB10 embeds each
external protein at full length on the GPU, as their scripts do; proteins longer than the longest test
protein (5,115 residues) need more memory than the GPU has (ProtT5's attention tables grow with the
square of the length) and are embedded here, on the CPU, as in their embedding jobs (SLURM jobs with
CPUs only, 200 GB of memory; analyze_Prot_GalNAc_T_833_Proteins.sh, analyze_ankh_OGalNAc_154.sh).

Embedding functions: their two scripts as functions, as in NB10 (cell 2), with every change tagged
[modified] / [added] / [removed]; here both pLMs run on the CPU. Score: the stacked model's second output;
call: their rule (second output > 0.5). TensorFlow's oneDNN operations are turned off, as in the
TensorFlow version of their jobs (2.6.2, where they were off by default).

Before the long proteins, three proteins already scored on the GPU in NB10 are scored again here; the
scores must agree. Each protein is saved when done and skipped on a restart.

Timing (compute statement): wall-clock time of each long protein (both pLMs and the stacked model),
saved next to its scores as proteins/{seq_id}.json; the sum over proteins is written, in the format of
the notebooks' compute records, to NB10_external_cpu_part.json (copied to results/compute/ on Drive)
once every protein is timed. If scores_long_part.parquet from an earlier run is in input-dir, the new
scores are compared with it.

Usage:
    python stack_oglypred_plm_long_proteins.py --input-dir input --out-dir output [--threads 48]
input-dir: long_proteins.parquet and scores_gpu_part.parquet (NB10), external_proteins.parquet (NB01),
           Model_Nucleus_and_Cytoplasm_Stacked_2687032Model.h5 (their Google Drive folder, NB10 cell 0);
           optionally scores_long_part.parquet from an earlier run.
out-dir:   a new folder for a timed run (proteins already in out-dir/proteins are skipped).
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''      # CPU only (the GPU is left to other jobs)
os.environ['TF_USE_LEGACY_KERAS'] = '1'      # tf.keras = Keras 2 (tf-keras): their .h5 model
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import argparse, hashlib, json, platform, resource, sys, time
from datetime import date
from importlib.metadata import version

parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
parser.add_argument('--input-dir', required=True)
parser.add_argument('--out-dir', required=True)
parser.add_argument('--threads', type=int, default=48, help='CPU threads for TensorFlow and PyTorch')
args = parser.parse_args()
for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[var] = str(args.threads)

import numpy as np
import pandas as pd
import torch
import torch._dynamo   # PyTorch and its compiler (triton) before TensorFlow: loading them after TensorFlow crashes Python
torch.set_num_threads(args.threads)
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(args.threads)
tf.config.threading.set_inter_op_parallelism_threads(2)
import transformers, ankh
from transformers import TFT5EncoderModel, T5Tokenizer

MODEL_FILE = 'Model_Nucleus_and_Cytoplasm_Stacked_2687032Model.h5'
MODEL_SHA256_PREFIX = '83ef58f0aa617db9'   # NB10 cell 0 (first 16 hex digits)
CHECK_TOL = 1e-4
os.makedirs(f'{args.out_dir}/proteins', exist_ok=True)

with open(f'{args.input_dir}/{MODEL_FILE}', 'rb') as f:
    assert hashlib.sha256(f.read()).hexdigest().startswith(MODEL_SHA256_PREFIX), f'{MODEL_FILE} differs'
stacked_model = tf.keras.models.load_model(f'{args.input_dir}/{MODEL_FILE}')   # as in their notebook (cell 11)

# ---------------------------------------------------------------------------
# Embedding functions: their two scripts (analyze_Prot_GalNAc_T_833_Proteins.py,
# analyze_ankh_OGalNAc_154.py), as functions, as in NB10 cell 2; both pLMs on the CPU.
# ---------------------------------------------------------------------------
#@ Stack-OglyPred-PLM analyze_Prot_GalNAc_T_833_Proteins.py 13-15 shift=0
def prott5_embedding(seq):  # [added] one sequence in, per-residue array out
    # [removed] FASTA reading: for seq_record in SeqIO.parse(name,"fasta"): placeholder = seq_record.id;
    #           seq = str(seq_record.seq); sequence_of_amino_acid_of_protein = seq; placeholder = placeholder.split("|")[1]
#@ Stack-OglyPred-PLM analyze_Prot_GalNAc_T_833_Proteins.py 21-21 shift=-4
#@ Stack-OglyPred-PLM analyze_Prot_GalNAc_T_833_Proteins.py 25-43 shift=-4
    # [removed] the per-residue CSV writing (filename = placeholder+"_Prot_Trans_"+".csv" ... fp.close())
    assert features.shape == (length_of_protein, 1024)  # [added] one row per residue
    return features  # [added] the per-residue array

ankh_model, ankh_tokenizer = ankh.load_large_model()  # [modified] original: {{Stack-OglyPred-PLM analyze_ankh_OGalNAc_154.py 32}}
ankh_model.eval()  # [modified] original: {{Stack-OglyPred-PLM analyze_ankh_OGalNAc_154.py 33}}

def ankh_embedding(sequence):  # [added] one sequence in, per-residue array out
    protein_sequences = [sequence]  # [modified] original: protein_sequences.append(Sequence) (read from a FASTA file)
    protein_sequences = [list(seq) for seq in protein_sequences]
    outputs = ankh_tokenizer.batch_encode_plus(protein_sequences,  # [modified] original: tokenizer.batch_encode_plus(
#@ Stack-OglyPred-PLM analyze_ankh_OGalNAc_154.py 37-41 shift=4
        embeddings = ankh_model(input_ids=outputs['input_ids'], attention_mask=outputs['attention_mask'])  # [modified] original: model(input_ids=outputs['input_ids'], attention_mask=outputs['attention_mask'])
    exact_embedding = embeddings[0][0][:-1]
    # [removed] the per-residue CSV writing (filename = str(placeholder)+"_ANKH_"+".csv" ... fp.close())
    exact_embedding = exact_embedding.numpy()  # [added] NumPy array
    assert exact_embedding.shape == (len(sequence), 1536)  # [added] one row per residue
    return exact_embedding  # [added] the per-residue array


def score_protein(seq):
    """Scores and calls of every S/T of one protein (1-based positions)."""
    pos = np.array([i + 1 for i, aa in enumerate(seq) if aa in 'ST'])
    X_t5 = prott5_embedding(seq)[pos - 1].astype(np.float32)
    X_ankh = ankh_embedding(seq)[pos - 1].astype(np.float32)
    assert np.isfinite(X_ankh).all() and np.isfinite(X_t5).all()
    probs = stacked_model.predict([X_ankh, X_t5], batch_size=4096, verbose=0)
    called = np.array([np.argmax(y, axis=None, out=None) for y in (probs > 0.5)]) == 1
    return pd.DataFrame({'position': pos, 'residue': [seq[p - 1] for p in pos], 'score': probs[:, 1], 'called': called})

peak_gb = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # kB on Linux

def cpu_model():
    """CPU model name from /proc/cpuinfo (Linux), or platform.processor() elsewhere."""
    try:
        with open('/proc/cpuinfo') as f:
            return next(line.split(':', 1)[1].strip() for line in f if line.startswith('model name'))
    except (OSError, StopIteration):
        return platform.processor() or None

provenance = {'tensorflow': tf.__version__, 'transformers': transformers.__version__, 'torch': torch.__version__,
              'ankh': version('ankh'), 'python': sys.version.split()[0], 'threads': args.threads, 'device': 'CPU'}
print('Versions: ' + '   '.join(f'{k} {v}' for k, v in provenance.items()), flush=True)

# ---------------------------------------------------------------------------
# Check: three proteins scored on the GPU in NB10 (300-1,000 residues, first by seq_id)
# ---------------------------------------------------------------------------
external = pd.read_parquet(f'{args.input_dir}/external_proteins.parquet', columns=['seq_id', 'sequence'])
gpu_scores = pd.read_parquet(f'{args.input_dir}/scores_gpu_part.parquet')
seq_of = dict(zip(external['seq_id'], external['sequence']))
candidates = sorted(sid for sid in gpu_scores['seq_id'].unique() if 300 <= len(seq_of[sid]) <= 1000)[:3]
diffs = []
for sid in candidates:
    t0 = time.time()
    ours = score_protein(seq_of[sid])
    gpu = gpu_scores[gpu_scores['seq_id'] == sid].sort_values('position')
    assert ours['position'].tolist() == gpu['position'].tolist()
    diffs.append(float(np.abs(ours['score'].to_numpy() - gpu['score'].to_numpy()).max()))
    calls = int((ours['called'].to_numpy() != gpu['called'].to_numpy()).sum())
    print(f'Check {sid} ({len(seq_of[sid]):,} residues, {len(ours)} S/T): max |score diff| vs GPU {diffs[-1]:.2e}, '
          f'calls differing {calls}, {time.time() - t0:.0f} s', flush=True)
assert max(diffs) < CHECK_TOL
provenance['check vs NB10 GPU scores'] = dict(zip(candidates, diffs))

# ---------------------------------------------------------------------------
# Long proteins, shortest first; each saved when done
# ---------------------------------------------------------------------------
long_proteins = pd.read_parquet(f'{args.input_dir}/long_proteins.parquet').sort_values(['length', 'seq_id'])
print(f'\nLong proteins: {len(long_proteins)}, {long_proteins["length"].min():,}-{long_proteins["length"].max():,} residues, '
      f'{long_proteins["n_st"].sum():,} S/T', flush=True)
t_start = time.time()
for k, (sid, seq) in enumerate(zip(long_proteins['seq_id'], long_proteins['sequence']), start=1):
    path = f'{args.out_dir}/proteins/{sid}.parquet'
    if os.path.exists(path):
        print(f'[{k}/{len(long_proteins)}] {sid}: done earlier', flush=True)
        continue
    print(f'[{k}/{len(long_proteins)}] {sid}: {len(seq):,} residues, started {time.strftime("%H:%M:%S")}', flush=True)
    t0 = time.perf_counter()
    scores = score_protein(seq)
    seconds = time.perf_counter() - t0
    scores.insert(0, 'seq_id', sid)
    scores.to_parquet(path + '.part', index=False)
    with open(f'{args.out_dir}/proteins/{sid}.json', 'w') as f:   # written before the scores are complete
        json.dump({'seconds': round(seconds, 1), 'residues': len(seq), 'st_sites': len(scores)}, f)
    os.replace(path + '.part', path)
    print(f'[{k}/{len(long_proteins)}] {sid}: {len(scores):,} S/T scored in {seconds / 60:.1f} min '
          f'(peak memory so far {peak_gb():.0f} GB; elapsed {(time.time() - t_start) / 60:.0f} min)', flush=True)

long_scores = pd.concat([pd.read_parquet(f'{args.out_dir}/proteins/{sid}.parquet') for sid in long_proteins['seq_id']],
                        ignore_index=True)
assert len(long_scores) == long_proteins['n_st'].sum() and not long_scores.duplicated(['seq_id', 'position']).any()
previous_file = f'{args.input_dir}/scores_long_part.parquet'
if os.path.exists(previous_file):
    previous = pd.read_parquet(previous_file).sort_values(['seq_id', 'position']).reset_index(drop=True)
    current = long_scores.sort_values(['seq_id', 'position']).reset_index(drop=True)
    assert previous[['seq_id', 'position']].equals(current[['seq_id', 'position']])
    provenance['vs earlier run'] = {'max |score diff|': float((previous['score'] - current['score']).abs().max()),
                                    'calls changed': int((previous['called'] != current['called']).sum())}
    print(f'Earlier run vs this run: {provenance["vs earlier run"]}', flush=True)
long_scores.to_parquet(f'{args.out_dir}/scores_long_part.parquet', index=False)

# Compute record: sum over proteins, in the format of the notebooks' compute records
timings = [f'{args.out_dir}/proteins/{sid}.json' for sid in long_proteins['seq_id']]
if all(os.path.exists(t) for t in timings):
    record = {'notebook': 'NB10', 'step': 'external_cpu_part',
              'seconds': round(sum(json.load(open(t))['seconds'] for t in timings), 1),
              'proteins': len(long_proteins), 'st_sites': int(long_proteins['n_st'].sum()),
              'gpu': None, 'cpu_cores': os.cpu_count(), 'threads': args.threads, 'cpu_model': cpu_model(),
              'host': platform.node(), 'date': date.today().isoformat()}
    with open(f'{args.out_dir}/NB10_external_cpu_part.json', 'w') as f:
        json.dump(record, f, indent=1)
    print(f'Compute record: {record["seconds"] / 3600:.2f} h on {record["cpu_model"]} ({args.threads} threads): '
          f'{args.out_dir}/NB10_external_cpu_part.json', flush=True)
else:
    print(f'{sum(not os.path.exists(t) for t in timings)} proteins have no timing (scored before timing was added): '
          'no compute record written', flush=True)
provenance['peak memory GB'] = peak_gb()
provenance['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
with open(f'{args.out_dir}/provenance.json', 'w') as f:
    json.dump(provenance, f, indent=1)
print(f'\nSaved {len(long_scores):,} S/T scores ({int(long_scores["called"].sum()):,} called positive): '
      f'{args.out_dir}/scores_long_part.parquet', flush=True)
