# Task 1: Promoter vs. Non-Promoter Classification
from itertools import product
from collections import Counter
import pandas as pd

def train_test():
    # Split dataset
    task1_data = pd.read_csv("data/task1_promoter.csv")

    train = task1_data[task1_data["split"] == "train"].copy()
    test = task1_data[task1_data["split"] == "test"].copy()

    train_label = train["label"]
    train_seq = train["sequence"]

    test_label = test["label"]
    test_seq = test["sequence"]

    return train_seq, train_label, test_seq, test_label


# Feature Engineering
def extract_kmer_features(sequence, k):
    """
       Extract normalized frequencies of all possible
       overlapping k-mers from a DNA sequence.
    """

    bases = ['A', 'C', 'G', 'T']

    # Generate all possible k-mers
    all_kmers = ["".join(p) for p in product(bases, repeat=k)]

    # Extract overlapping k-mers
    seq_kmers = [sequence[i:i+k] for i in range(len(sequence)-k+1)]

    # count occurrence
    counts = Counter(seq_kmers)

    # Calculate frequencies
    total = len(seq_kmers)

    features = {
        f"kmer_{k}_{kmer}": counts[kmer] / total
        for kmer in all_kmers
    }

    return features

def extract_composition_features(sequence):
    """
    Calculate nucleotide composition and GC content.
    """
    length = len(sequence)
    counts = Counter(sequence)

    features = {
        "A_freq": counts["A"]/length,
        "C_freq": counts["C"]/length,
        "G_freq": counts["G"]/length,
        "T_freq": counts["T"]/length,
        "GC_content": (counts["G"] + counts["C"]) / length,
    }

    return features

def extract_additional_features(sequence):

    # Divide the sequence into TSS-relative regions
    counts_negative60_36 = Counter(sequence[0:25])
    counts_negative35_16 = Counter(sequence[25:45])
    counts_negative15_1 = Counter(sequence[45:60])
    counts_positive1_21 = Counter(sequence[60:81])

    counts = Counter(sequence)

    g = counts["G"]
    c = counts["C"]

    # GC skew
    gc_skew = (g - c) / (g + c) if (g + c) > 0 else 0

    # Regional nucleotide composition based on TSS
    features = {}

    regions = {
        "negative60_36": (counts_negative60_36, 25),
        "negative35_16": (counts_negative35_16, 20),
        "negative15_1": (counts_negative15_1, 15),
        "positive1_21": (counts_positive1_21, 21)
    }

    for region_name, (region_counts, region_length) in regions.items():
        for base in "ATCG":
            features[f"{base}_{region_name}_freq"] = (
                region_counts[base] / region_length
            )

    features["GC_skew"] = gc_skew

    return features

def extract_all_features(sequence, k):

    features = {}

    # K-mer frequencies
    features.update(
        extract_kmer_features(sequence, k)
    )

    # Nucleotide composition
    features.update(
        extract_composition_features(sequence)
    )

    # Additional features
    features.update(
        extract_additional_features(sequence)
    )

    return features

def create_feature_matrix(sequences, k):

    feature_list = [
        extract_all_features(seq, k)
        for seq in sequences
    ]

    return pd.DataFrame(feature_list, index=sequences.index)



