from scipy.stats import ttest_ind
import pandas as pd
import numpy as np

def get_distribution(df, type = 'effective_diversity', filter = None):
    sub_df = pd.read_csv(df)
    if filter:
        sub_df = sub_df[sub_df["split_name"] == filter]
    if type == 'effective_diversity':
        return (
            sub_df[sub_df["tm_score"] > 0.7]
            .groupby("name")
            .apply(lambda x: diversity(x["pred"]))
        )
    else:
        return sub_df.groupby("name")["tm_score"].mean()

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

run_cath = False
run_ts_50 = True
run_ts_500 = False
run_casp15 = False

if run_cath:
    our_model = ["../result_files/RLDIF_CATH4.2_results.csv"]
    CATH4_2_files = ["../result_files/protein_mpnn_CATH4.2_results.csv",
    "../result_files/PiFold_CATH4.2_results.csv",
    "../result_files/KWDesign_CATH4.2_results.csv"
    ]

    to_scan = ["all", "single_chain", "short"]
    for w in to_scan:
        print(f"CATH4.2 Results EFFECTIVE DIVERSITY for {w}")
        for i in our_model:
            for j in CATH4_2_files:
                if i != j:
                    print(f"Stats for {i} vs {j}")
                    model_one = get_distribution(i, filter = w)
                    model_two = get_distribution(j, filter = w)

                    # Assuming mpnn and pifold are your two samples
                    t_statistic, p_value = ttest_ind(model_one, model_two)
                    print(f"Direction: {np.mean(model_one) - np.mean(model_two)}")
                    print(f"{np.mean(model_one)} vs {np.mean(model_two)}")
                    print(f'T-statistic: {t_statistic}')
                    print(f'P-value: {p_value}')
                    print("\n")

        print("CATH4.2 Results STRUCTURAL SIMILARITY")
        for i in our_model:
            for j in CATH4_2_files:
                if i != j:
                    print(f"Stats for {i} vs {j}")
                    model_one = get_distribution(i, None, filter = w)
                    model_two = get_distribution(j, None, filter = w)
                    # Assuming mpnn and pifold are your two samples
                    t_statistic, p_value = ttest_ind(model_one, model_two)
                    print(f"Direction: {np.mean(model_one) - np.mean(model_two)}")
                    print(f"{np.mean(model_one)} vs {np.mean(model_two)}")
                    print(f'T-statistic: {t_statistic}')
                    print(f'P-value: {p_value}')
                    print("\n")

if run_ts_50:
    our_model_ts50 = ["../result_files/RLDIF_TS50_results.csv"]
    TS50_files = ["../result_files/protein_mpnn_TS50_results.csv",
    "../result_files/PiFold_TS50_results.csv",
    ]

    print(f"TS50 Results EFFECTIVE DIVERSITY")
    for i in our_model_ts50:
        for j in TS50_files:
            if i != j:
                print(f"Stats for {i} vs {j}")
                model_one = get_distribution(i)
                model_two = get_distribution(j)

                # Assuming mpnn and pifold are your two samples
                t_statistic, p_value = ttest_ind(model_one, model_two)
                print(f"Direction: {np.mean(model_one) - np.mean(model_two)}")
                print(f"{np.mean(model_one)} vs {np.mean(model_two)}")
                print(f'T-statistic: {t_statistic}')
                print(f'P-value: {p_value}')
                print("\n")

    print("TS50 Results STRUCTURAL SIMILARITY")
    for i in our_model_ts50:
        for j in TS50_files:
            if i != j:
                print(f"Stats for {i} vs {j}")
                model_one = get_distribution(i, None)
                model_two = get_distribution(j, None)
                # Assuming mpnn and pifold are your two samples
                t_statistic, p_value = ttest_ind(model_one, model_two)
                print(f"Direction: {np.mean(model_one) - np.mean(model_two)}")
                print(f"{np.mean(model_one)} vs {np.mean(model_two)}")
                print(f'T-statistic: {t_statistic}')
                print(f'P-value: {p_value}')
                print("\n")

if run_ts_500:
    our_model_ts500 = ["../result_files/RLDIF_TS500_results.csv"]
    TS500_files = ["../result_files/protein_mpnn_TS500_results.csv",
    "../result_files/PiFold_TS500_results.csv",
    ]

    print(f"TS500 Results EFFECTIVE DIVERSITY")
    for i in our_model_ts500:
        for j in TS500_files:
            if i != j:
                print(f"Stats for {i} vs {j}")
                model_one = get_distribution(i)
                model_two = get_distribution(j)

                # Assuming mpnn and pifold are your two samples
                t_statistic, p_value = ttest_ind(model_one, model_two)
                print(f"Direction: {np.mean(model_one) - np.mean(model_two)}")
                print(f"{np.mean(model_one)} vs {np.mean(model_two)}")
                print(f'T-statistic: {t_statistic}')
                print(f'P-value: {p_value}')
                print("\n")

    print("TS500 Results STRUCTURAL SIMILARITY")
    for i in our_model_ts500:
        for j in TS500_files:
            if i != j:
                print(f"Stats for {i} vs {j}")
                model_one = get_distribution(i, None)
                model_two = get_distribution(j, None)
                # Assuming mpnn and pifold are your two samples
                t_statistic, p_value = ttest_ind(model_one, model_two)
                print(f"Direction: {np.mean(model_one) - np.mean(model_two)}")
                print(f"{np.mean(model_one)} vs {np.mean(model_two)}")
                print(f'T-statistic: {t_statistic}')
                print(f'P-value: {p_value}')
                print("\n") 

if run_casp15:
    our_model_casp15 = ["../result_files/RLDIF_CASP15_results.csv"]
    CASP15_files = ["../result_files/protein_mpnn_CASP15_results.csv",
    "../result_files/PiFold_CASP15_results.csv",
    "../result_files/KWDesign_CASP15_results.csv"
    ]

    print(f"CASP15 Results EFFECTIVE DIVERSITY")
    for i in our_model_casp15:
        for j in CASP15_files:
            if i != j:
                print(f"Stats for {i} vs {j}")
                model_one = get_distribution(i)
                model_two = get_distribution(j)

                # Assuming mpnn and pifold are your two samples
                t_statistic, p_value = ttest_ind(model_one, model_two)
                print(f"Direction: {np.mean(model_one) - np.mean(model_two)}")
                print(f"{np.mean(model_one)} vs {np.mean(model_two)}")
                print(f'T-statistic: {t_statistic}')
                print(f'P-value: {p_value}')
                print("\n")

    print("CASP15 Results STRUCTURAL SIMILARITY")
    for i in our_model_casp15:
        for j in CASP15_files:
            if i != j:
                print(f"Stats for {i} vs {j}")
                model_one = get_distribution(i, None)
                model_two = get_distribution(j, None)
                # Assuming mpnn and pifold are your two samples
                t_statistic, p_value = ttest_ind(model_one, model_two)
                print(f"Direction: {np.mean(model_one) - np.mean(model_two)}")
                print(f"{np.mean(model_one)} vs {np.mean(model_two)}")
                print(f'T-statistic: {t_statistic}')
                print(f'P-value: {p_value}')
                print("\n") 
