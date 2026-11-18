import pandas as pd
import os

def create_meme_file(folder, output_file):

    nucleotides = ["A", "C", "G", "T"]
    pseudocount = 0.5

    with open(output_file, "w") as f:

        f.write("MEME version 4\n\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("strands: + -\n\n")
        f.write("Background letter frequencies\n")
        f.write("A 0.25 C 0.25 G 0.25 T 0.25\n\n")

        for n in range(1, 33):

            filename = f"filter_{n}_PFM.csv"
            filepath = os.path.join(folder, filename)

            pfm = pd.read_csv(filepath)

            # Ensure nucleotide column order
            pfm = pfm[nucleotides]

            counts = pfm.to_numpy(dtype=float)

            # Add pseudocounts
            adjusted_counts = counts + pseudocount

            # Convert counts into probabilities
            probabilities = (
                adjusted_counts /
                adjusted_counts.sum(axis=1, keepdims=True)
            )

            width = probabilities.shape[0]

            f.write(f"MOTIF Filter_{n}\n")

            f.write(
                f"letter-probability matrix: "
                f"alength= 4 w= {width} nsites= {int(counts.sum())}\n"
            )

            for row in probabilities:
                f.write(
                    " ".join(f"{value:.6f}" for value in row) + "\n"
                )

            f.write("\n")

create_meme_file(
    folder="task3_filter_results/PFM",
    output_file="task3_filter_results/cnn_filter_motifs_bonus.meme"
)

with open("task3_filter_results/cnn_filter_motifs_bonus.meme", "r") as f:
    print(f.read()[:1500])