import pandas as pd
import numpy as np
import ast

#Put the location of the file you want to analyze here
test_file_path = "../result_files/RLDIF_CATH4.2_results.csv"

df = pd.read_csv(test_file_path)


def accuracy(x, y):
    num = 0
    for i, j in zip(x, y):
        if i == j:
            num += 1
    return num / len(x)


def diversity(sequences):
    if len(set(sequences)) == 1:
        return 0

    pairwise_similarity = []
    for i in sequences:
        for j in sequences:
            if i != j:
                res = sum([1 for x, z in zip(i, j) if x == z]) / len(i)
                pairwise_similarity.append(1 - res)

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
    print(f"NGCC TM-Score Output: {res}")
    res = (
        sub_df[sub_df["tm_score"] > 0.7]
        .groupby("name")
        .apply(lambda x: diversity(x["pred"]))
    )
    print(f"Effective Diversity (Diversity for NGCC TM-Score > 0.7): {res.mean()}\t{res.std()}")
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



