"""Concatenate per-chromosome graph features and append reverse complements.

Loads per-chromosome NumPy arrays produced by preprocess_graphs.py (or a
similar pipeline), concatenates them across training chromosomes, computes
the element-wise reverse complement of the encoded sequences, and saves the
combined forward + reverse-complement feature matrix along with labels.

Usage:
    python preprocessRevComp.py <transcriptionFactor> <celltype>

Arguments:
    transcriptionFactor  Name of the transcription factor (used in file paths).
    celltype             Cell type identifier (used in file paths).
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

transcriptionFactor = sys.argv[1]
celltype = sys.argv[2]

transcriptionFactors = [transcriptionFactor]
celltypes = [celltype]


def reverseComplement(input):
    """Map a single encoded nucleotide to its complement.

    Encoding: 0='-', 1='A', 2='C', 3='G', 4='N', 5='T'
    Complements: A<->T (1<->5), C<->G (2<->3); others unchanged.

    Args:
        input: Integer-encoded nucleotide value.

    Returns:
        Integer-encoded complement value.
    """
    if int(input) == 1:
        return 5
    elif int(input) == 5:
        return 1
    elif int(input) == 2:
        return 3
    elif (input) == 3:
        return 2
    else:
        return input


def makeReverse(input):
    """Compute element-wise reverse complement of a 3-D encoded array.

    Applies reverseComplement to every element and reverses the sequence
    axis (axis=2).

    Args:
        input: 3-D NumPy array of shape (samples, species, positions).

    Returns:
        Reverse-complemented array of the same shape.
    """
    output = np.zeros(input.shape)
    a, b, c = input.shape
    for i in tqdm(range(a)):
        for j in range(b):
            for k in range(c):
                output[i, j, k] = reverseComplement(input[i, j, k])
    return np.flip(output, axis=2)


def get_one_hot(targets, nb_classes):
    """Convert integer labels to one-hot encoded arrays.

    Args:
        targets:    NumPy array of integer class labels.
        nb_classes: Number of classes.

    Returns:
        One-hot encoded uint8 array with an extra trailing dimension.
    """
    res = np.eye(nb_classes, dtype=np.uint8)[np.array(targets).reshape(-1)].astype('uint8')
    return res.reshape(list(targets.shape) + [nb_classes])


# Sequence lengths to process (context window * 2 + 1 per side, doubled for rev comp)
lengths = [1001]

for transcriptionFactor in transcriptionFactors:
    for celltype in celltypes:
        for length in lengths:
            encode_list = ['-', 'A', 'C', 'G', 'N', 'T']

            # Training chromosomes (chr1, chr8, chr21 held out for val/test)
            chromosomes_train = [3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22]
            chromosomes_val = []

            # Start with chr2 as the base array
            X_train = np.load(
                'graphs/{}/dataset_{}_chr{}_{}_{}_X_train.npy'.format(
                    transcriptionFactor, length, 2, transcriptionFactor, celltype
                )
            ).astype(np.uint8)
            y_train = np.load(
                'graphs/{}/dataset_{}_chr{}_{}_{}_y_train.npy'.format(
                    transcriptionFactor, length, 2, transcriptionFactor, celltype
                )
            ).astype(np.uint8)

            # Concatenate remaining training chromosomes
            for chrom in tqdm(chromosomes_train):
                X_train = np.concatenate((
                    X_train,
                    np.load('graphs/{}/dataset_{}_chr{}_{}_{}_X_train.npy'.format(
                        transcriptionFactor, length, chrom, transcriptionFactor, celltype
                    )).astype(np.uint8)
                ), axis=0)
                y_train = np.concatenate((
                    y_train,
                    np.load('graphs/{}/dataset_{}_chr{}_{}_{}_y_train.npy'.format(
                        transcriptionFactor, length, chrom, transcriptionFactor, celltype
                    )).astype(np.uint8)
                ), axis=0)

            # Compute reverse complement using vectorised function (much faster than loop)
            myfunc_vec = np.vectorize(reverseComplement)
            result = myfunc_vec(X_train)
            print(transcriptionFactor, celltype)
            print(result.shape)
            print(type(result[0, 0, 0]))
            print(X_train.shape)
            print(type(X_train[0, 0, 0]))

            # Reverse the sequence axis so the complement reads 3'->5'
            result = np.flip(result, axis=2)

            # Concatenate forward and reverse-complement along sequence axis
            X_train_revComp = np.concatenate((X_train, result), axis=2).astype(np.uint8)
            y_train = y_train.astype(np.uint8)
            print(X_train_revComp.shape, y_train.shape)

            # Save concatenated feature matrix and labels
            np.save('graphs/{}/X_revCompConcatenatedTrue{}_{}.npy'.format(
                transcriptionFactor, length, celltype), X_train_revComp)
            np.save('graphs/{}/y_revCompConcatenatedTrue{}_{}.npy'.format(
                transcriptionFactor, length, celltype), y_train)
            np.save('graphs/{}/y_revCompConcatenated{}_{}.npy'.format(
                transcriptionFactor, length, celltype), y_train)

            # Free memory
            X_train_revComp = []
            y_train_revComp = []
            X_train = []
            y_train = []
