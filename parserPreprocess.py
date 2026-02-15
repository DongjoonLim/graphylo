"""Parse a MAF (Multiple Alignment Format) file and build a per-base alignment dictionary.

This script reads an ancestral-sequence MAF file for a given chromosome,
reconstructs a gapless, position-indexed alignment dictionary for 58 extant
species and 57 internal/ancestral nodes (115 entries total), and saves the
result as a pickle file.

Steps:
    1. Read and parse the MAF file, extracting aligned blocks.
    2. Assemble a position-indexed dictionary (seqDict) that maps each species
       or ancestral node to a list of characters covering the full chromosome.
       Gaps between alignment blocks are filled with 'N'.
    3. Remove columns where the human reference (hg38) has a gap character,
       producing a coordinate system that matches the reference genome.
    4. Save the final dictionary as ``seqDictPad_chr<chrom>.pkl``.

Usage:
    python parserPreprocess.py <chromosome>

Arguments:
    chromosome  Chromosome identifier (e.g. 1, 2, ..., 22, X).
"""

import numpy as np
from tqdm import tqdm
import pickle
import sys

chromosome = sys.argv[1]

# Full list of 58 extant species + 57 ancestral/internal nodes = 115 entries
ancSeq = [
    'hg38', 'panTro4', 'gorGor3', 'ponAbe2', 'nomLeu3', 'rheMac3',
    'macFas5', 'papAnu2', 'chlSab2', 'calJac3', 'saiBol1', 'otoGar3', 'tupChi1',
    'speTri2', 'jacJac1', 'micOch1', 'criGri1', 'mesAur1', 'mm10',
    'rn6', 'hetGla2', 'cavPor3', 'chiLan1', 'octDeg1',
    'oryCun2', 'ochPri3',
    'susScr3', 'vicPac2', 'camFer1', 'turTru2', 'orcOrc1', 'panHod1',
    'bosTau8', 'oviAri3', 'capHir1',
    'equCab2', 'cerSim1', 'felCat8', 'canFam3',
    'musFur1', 'ailMel1', 'odoRosDiv1', 'lepWed1', 'pteAle1', 'pteVam1',
    'eptFus1', 'myoDav1', 'myoLuc2', 'eriEur2', 'sorAra2', 'conCri1',
    'loxAfr3', 'eleEdw1', 'triMan1', 'chrAsi1', 'echTel2', 'oryAfe1', 'dasNov3',
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
print(len(ancSeq))


def ungap(anc, des):
    """Remove columns where both ancestor and descendant have a gap.

    Args:
        anc: Ancestor sequence string.
        des: Descendant sequence string.

    Returns:
        Tuple of (ungapped_ancestor, ungapped_descendant) strings.
    """
    a = ''
    d = ''
    for i in range(len(anc)):
        if anc[i] == '-' and des[i] == '-':
            continue
        else:
            a = a + anc[i]
            d = d + des[i]
    return a, d


def getAlign(inputList, ancSeq):
    """Build a position-indexed alignment dictionary from parsed MAF blocks.

    For each alignment block the sequences are appended to the running
    dictionary.  Gaps between consecutive blocks (positions not covered by
    any block) are filled with 'N'.  After processing all blocks, shorter
    sequences are padded with '-' so every entry has the same length.

    Args:
        inputList: List of alignment blocks.  Each block is a list of
                   [species_name, start, length, sequence] entries.
        ancSeq:    Ordered list of species / node names.

    Returns:
        dict mapping each name in *ancSeq* to a list of characters covering
        the full chromosome.
    """
    nucSet = set(['A', 'C', 'G', 'T', '-'])
    seqDict = {}
    seqTemp = {}

    # Initialise each species with 'N' padding up to the first block start
    for a in tqdm(ancSeq):
        seqDict[a] = ['N'] * inputList[0][0][1]
    full = []
    idList = []

    # (First pass was a no-op in the original; kept for fidelity)
    for i in tqdm(range(len(inputList))):
        idList = []
        lengths = []

    # Main pass: append aligned bases and fill inter-block gaps with 'N'
    for i in tqdm(range(len(inputList))):
        for j in range(len(inputList[i])):
            # Extract species name (strip chromosome suffix after '.')
            item = str(inputList[i][j][0]).split('.')[0]
            seqDict[item].extend(list(inputList[i][j][3].upper()))

        # Fill inter-block gaps for every species
        for item in ancSeq:
            if i != 0 and (inputList[i][0][1] != inputList[i-1][0][1] + inputList[i-1][0][2]):
                seqDict[item].extend(
                    ['N'] * (inputList[i][0][1] - inputList[i-1][0][1] - inputList[i-1][0][2])
                )

        # Pad shorter sequences with '-' to equalise lengths across species
        lengths = []
        for item in ancSeq:
            lengths.append(len(seqDict[item]))
        maximum = max(lengths)
        minimum = min(lengths)
        if maximum == minimum:
            continue
        else:
            for item in ancSeq:
                if len(seqDict[item]) == maximum:
                    continue
                else:
                    seqDict[item].extend(['-'] * (maximum - len(seqDict[item])))

    print(len(seqDict['hg38']))
    return seqDict


# --- Read and parse the MAF file ---
file1 = open("../research/data/chr{}.anc.maf".format(chromosome), "rb")
Lines = file1.readlines()
count = 0
seqList = []   # list of alignment blocks
tempList = []  # current block being built
for line in Lines:
    line = str(line, 'utf-8')
    # Blank line signals end of an alignment block
    if len(line.split()) == 0:
        if len(tempList) != 0:
            seqList.append(tempList)
            tempList = []
    elif line.split()[0] == "s":
        # 's' lines contain aligned sequence data
        try:
            tempList.append([
                line.split()[1],          # species.chrom
                int(line.split()[2]),      # start position
                int(line.split()[3]),      # alignment length
                line.split()[6]            # aligned sequence
            ])
        except:
            print(line.split())
            continue
print(seqList[:1000])
file1.close()

# Build the full alignment dictionary
seqDictRaw = getAlign(seqList, ancSeq)

# Remove columns where the human reference has a gap ('-'),
# so that indices correspond to reference-genome coordinates
indicies = []
for i in tqdm(range(len(seqDictRaw['hg38']))):
    if seqDictRaw['hg38'][i] != '-':
        indicies.append(i)

# Subset every species to the non-gap reference positions
for key in tqdm(seqDictRaw.keys()):
    temp = [seqDictRaw[key][i] for i in indicies]
    seqDictRaw[key] = temp

print(len(indicies))

# Save the final alignment dictionary as a pickle
with open('seqDictPad_chr{}.pkl'.format(chromosome), 'wb') as handle:
    pickle.dump(seqDictRaw, handle)
