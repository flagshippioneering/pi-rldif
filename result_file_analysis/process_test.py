import pandas as pd
import numpy as np
import ast

#Put the location of the file you want to analyze here
test_file_path = "../result_files/RLDIF_100K_CATH_results.csv"
scTM_threshold = 0.7
df = pd.read_csv(test_file_path)


def accuracy(x, y):
    num = 0
    for i, j in zip(x, y):
        if i == j:
            num += 1
    return num / len(x)


def diversity(sequences, to_sum = False, return_raw = False):
    sequences = sequences.values

    pairwise_similarity = []
    n = len(sequences)
    for i in range(n):
        for j in range(i+1, n):
            seq_i = sequences[i]
            seq_j = sequences[j]
            res = sum([1 for x, z in zip(seq_i, seq_j) if x == z]) / len(seq_i)
            pairwise_similarity.append(1 - res)
    if return_raw:
        return pairwise_similarity
    if to_sum:
        return np.sum(pairwise_similarity)
    return np.mean(pairwise_similarity)

if 'TS500' in test_file_path:
    to_scan = ['ts500']
elif 'TS50' in test_file_path:
    to_scan = ['ts50']
elif 'CASP15' in test_file_path or 'Casp15' in test_file_path:
    to_scan = ['casp15']
else:
    to_scan = ["all", "single_chain", "short"]

for i in to_scan:
    print(f"Stats for {i}")
    sub_df = df[df["split_name"] == i]
    res = sub_df.apply(lambda x: accuracy(x["pred"], x["real"]), axis=1)
    print(f"Sequence Recovery: {res.mean()}")
    res = sub_df.groupby("name").apply(lambda x: diversity(x["pred"])).mean()
    print(f"Diversity: {res.mean()}")
    res = sub_df["tm_score"].mean()
    print(f"TM-Score Output: {res}")
    res = (
        sub_df[sub_df["tm_score"] > scTM_threshold]
        .groupby("name")
        .apply(lambda x: diversity(x["pred"], to_sum = True))
    )
    print(f"Foldable diversity (Diversity for TM-Score > scTM_Threshold): {(2*res.sum())/((sub_df.shape[0]//4)*4*3)}")
    print("\n")

print("Mean out of all 4 sampled")
for i in to_scan:
    print(f"Stats for {i}")
    sub_df = df[df["split_name"] == i]
    res = sub_df.apply(
        lambda x: pd.Series(
            {"name": x["name"], "accuracy": accuracy(x["pred"], x["real"])}
        ),
        axis=1,
    )
    print(f"Sequence Recovery: {res.groupby('name').mean().mean().item()}")

    res = sub_df["tm_score"].mean()
    print(
        f"TM-Score Output: {sub_df[['name', 'tm_score']].groupby('name').mean().mean()['tm_score'].item()}"
    )
    print("\n")



