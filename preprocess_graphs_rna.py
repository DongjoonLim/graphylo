"""Preprocess genomic alignments for RNA-binding prediction.

Similar to preprocess_graphs.py but tailored for RNA-binding tasks:
- Labels encode strand orientation (0/1 = forward, 2/3 = reverse complement).
- Forward or reverse-complement sequence is selected based on the label.
- Output labels are mapped to their absolute values (binary binding signal).

Usage:
    python preprocess_graphs_rna.py <input_path> <chrom> <output_X> <output_y> [alignment_dir]

Arguments:
    input_path     Path to the whitespace-delimited input file (cols: chr, pos, ..., label).
    chrom          Chromosome number (e.g. 1, 2, ..., 22).
    output_X       Output path for the feature matrix (.npy).
    output_y       Output path for the label vector (.npy).
    alignment_dir  (Optional) Directory containing seqDictPad_chr*.pkl files.
                   Defaults to the current working directory.
"""

import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Encode nucleotides + gap as integers 0-5
le = LabelEncoder()
le.fit(['A', 'C', 'G', 'T', 'N', '-'])

# --- Parse command-line arguments ---
input_path = sys.argv[1]
chrom = sys.argv[2]
output_X = sys.argv[3]
output_y = sys.argv[4]
# Optional 5th argument: directory containing alignment pickle files
alignment_dir = sys.argv[5] if len(sys.argv) > 5 else '.'

# Number of bases on each side of the target position
context = 100

# Load the pre-built multi-species alignment for this chromosome
alignment_file = os.path.join(alignment_dir, f'seqDictPad_chr{chrom}.pkl')
alignment = pd.read_pickle(alignment_file)

# Read input positions and filter to the requested chromosome
input = pd.read_csv(input_path, delimiter=r"\s+")
input = input.loc[input.iloc[:, 0] == f'chr{chrom}']
indices = input.iloc[:, 1]     # genomic positions
y_true = input.iloc[:, 3]      # labels (signed: sign encodes strand)
print(indices, y_true)


def reverse_complement(dna):
    """Return the reverse complement of a DNA sequence (list of characters)."""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N', '-': '-'}
    return [complement[base] for base in reversed(dna)]


# Species and ancestral-node names in the 115-row phylogenetic graph
names = [
    'hg38', 'panTro4', 'gorGor3', 'ponAbe2', 'nomLeu3', 'rheMac3',
    'macFas5', 'papAnu2', 'chlSab2', 'calJac3', 'saiBol1', 'otoGar3', 'tupChi1',
    'speTri2', 'jacJac1', 'micOch1', 'criGri1', 'mesAur1', 'mm10',
    'rn6', 'hetGla2', 'cavPor3', 'chiLan1', 'octDeg1',
    'oryCun2', 'ochPri3', 'susScr3', 'vicPac2', 'camFer1', 'turTru2',
    'orcOrc1', 'panHod1', 'bosTau8', 'oviAri3', 'capHir1', 'equCab2',
    'cerSim1', 'felCat8', 'canFam3',
    'musFur1', 'ailMel1', 'odoRosDiv1', 'lepWed1', 'pteAle1', 'pteVam1',
    'eptFus1', 'myoDav1', 'myoLuc2', 'eriEur2',
    'sorAra2', 'conCri1', 'loxAfr3', 'eleEdw1', 'triMan1', 'chrAsi1',
    'echTel2', 'oryAfe1', 'dasNov3',
    # Internal / ancestral nodes in the phylogenetic tree
    '_HP', '_HPG', '_HPGP', '_HPGPN', '_RM', '_RMP', '_RMPC', '_HPGPNRMPC',
    '_CS', '_HPGPNRMPCCS', '_HPGPNRMPCCSO', '_HPGPNRMPCCSOT',
    '_CM', '_MR', '_MCM', '_MCMMR', '_JMCMMR', '_SJMCMMR', '_CO', '_CCO',
    '_HCCO', '_SJMCMMRHCCO', '_OO', '_SJMCMMRHCCOOO',
    '_HPGPNRMPCCSOTSJMCMMRHCCOOO',
    '_VC', '_TO', '_OC', '_BOC', '_PBOC', '_TOPBOC', '_VCTOPBOC', '_SVCTOPBOC',
    '_EC', '_OL', '_AOL', '_MAOL', '_CMAOL', '_FCMAOL', '_ECFCMAOL',
    '_PP', '_MM', '_EMM', '_PPEMM', '_ECFCMAOLPPEMM', '_SVCTOPBOCECFCMAOLPPEMM',
    '_SC', '_ESC', '_SVCTOPBOCECFCMAOLPPEMMESC',
    '_HPGPNRMPCCSOTSJMCMMRHCCOOOSVCTOPBOCECFCMAOLPPEMMESC',
    '_LE', '_LET', '_CE', '_LETCE', '_LETCEO', '_LETCEOD',
    '_HPGPNRMPCCSOTSJMCMMRHCCOOOSVCTOPBOCECFCMAOLPPEMMESCLETCEOD'
]

# Build feature arrays: each example is (115 species, context*2+1) uint8
# For RNA binding, we select forward or reverse complement based on the label
examples = []
targets = []
for i, target in tqdm(zip(indices, y_true)):
    i = int(i)
    try:
        example = []
        # Skip positions where the human reference is ambiguous
        if alignment['hg38'][i] == 'N':
            print('contains N')
            continue
        for key in names:
            sequence_raw = alignment[key]
            # Labels 0 or 1 -> forward strand; otherwise -> reverse complement
            if (int(target) == 0) or (int(target) == 1):
                sequence = le.transform(sequence_raw[i - context: i + context + 1])
            else:
                sequence = le.transform(
                    reverse_complement(sequence_raw[i - context: i + context + 1])
                )
            example.append(sequence)
        example = np.array(example).astype('uint8')
        assert example.shape == (115, context * 2 + 1)
        examples.append(example)
        # Store absolute value of label (binary binding signal)
        targets.append(int(abs(int(target))))
    except Exception as e:
        print(e)
        continue

examples = np.array(examples)
targets = np.array(targets)
print(examples.shape, targets.shape)

# Save processed arrays
np.save(f'{output_X}', examples)
np.save(f'{output_y}', targets)
