# # #!/usr/bin/env python3

# # """
# # Clean intersectional fairness evaluation for FairVLMed10k.

# # For each method × demographic intersection:
# #     - enumerate all Cartesian subgroup combinations
# #     - identify EMPTY / SPARSE / SINGLE-CLASS cells
# #     - compute subgroup AUC only for valid cells
# #     - mark the whole intersection as non-evaluable if ANY expected cell
# #       is empty, sparse, or single-class
# #     - generate a publication-ready heatmap

# # Outputs:
# #     intersectional_results.csv
# #     intersectional_cell_details.csv
# #     intersectional_empty_report.csv
# #     fig_intersectional_heatmap_clean.png
# # """

# # import os
# # import glob
# # import pickle
# # import numpy as np
# # import pandas as pd
# # import scipy.special

# # from sklearn.metrics import roc_auc_score

# # import matplotlib.pyplot as plt
# # import seaborn as sns


# # # ============================================================
# # # 1. CONFIGURATION
# # # ============================================================

# # PROJECT_ROOT = (
# #     "/home/alaa/papersubmission/"
# #     "The-Impact-of-Adaptation-on-Fairness-in-Medical-Multimodal-"
# #     "Foundation-Models"
# # )

# # TEST_CSV = os.path.join(
# #     PROJECT_ROOT,
# #     "Harvard FairVLMed (Harvard-FairVLMed）",
# #     "test.csv"
# # )

# # RESULTS_DIR = os.path.join(
# #     PROJECT_ROOT,
# #     "FairVLMed10k_clip_results"
# # )

# # OUTPUT_DIR = PROJECT_ROOT


# # # Minimum number of samples required in a subgroup
# # MIN_N = 10


# # # Age bins used in the intersectional analysis
# # # IMPORTANT: these MUST be the same bins used in your original analysis.
# # AGE_BINS = [51.2, 63.9, 73.3]


# # # Methods in paper order
# # METHODS = [
# #     "clip-zs",
# #     "lp",
# #     "clip-adapt",
# #     "lora-lp",
# #     "lora-text",
# #     "full-ft",
# #     "full-ft-text",
# # ]


# # # Your actual filesystem directory names
# # METHOD_DIR_MAP = {
# #     "clip-zs": "clip-zs_lr1e-2_seed1",
# #     "lp": "lp_lr1e-2_seed1",
# #     "clip-adapt": "clip-adapt_lr1e-2_seed1",
# #     "lora-lp": "lora-lp_lr1e-2_seed1",
# #     "lora-text": "lora-zs_lr1e-2_seed1",
# #     "full-ft": "full-ft_lr5e-3_seed1",
# #     "full-ft-text": "full-ft-zs_lr5e-3_seed1",
# # }


# # # All 15 pairwise intersections
# # INTERSECTIONS = [
# #     ("Age", "Marital"),
# #     ("Race", "Language"),
# #     ("Gender", "Marital"),
# #     ("Language", "Marital"),
# #     ("Age", "Language"),
# #     ("Ethnicity", "Marital"),
# #     ("Race", "Marital"),
# #     ("Age", "Ethnicity"),
# #     ("Race", "Ethnicity"),
# #     ("Age", "Race"),
# #     ("Language", "Ethnicity"),
# #     ("Gender", "Language"),
# #     ("Gender", "Race"),
# #     ("Gender", "Ethnicity"),
# #     ("Gender", "Age"),
# # ]


# # # ============================================================
# # # 2. LOAD TEST METADATA
# # # ============================================================

# # print("=" * 100)
# # print("LOADING TEST METADATA")
# # print("=" * 100)

# # df = pd.read_csv(TEST_CSV)

# # print(f"Test CSV: {TEST_CSV}")
# # print(f"Test set size: {len(df)}")
# # print("\nColumns:")
# # print(df.columns.tolist())


# # # ------------------------------------------------------------
# # # Find columns robustly
# # # ------------------------------------------------------------

# # lower_cols = {c.lower(): c for c in df.columns}


# # def find_column(*names):
# #     for name in names:
# #         if name.lower() in lower_cols:
# #             return lower_cols[name.lower()]
# #     raise KeyError(
# #         f"Could not find any of these columns: {names}\n"
# #         f"Available columns: {df.columns.tolist()}"
# #     )


# # gender_col = find_column("Gender", "Sex")
# # age_col = find_column("Age")
# # race_col = find_column("Race")
# # language_col = find_column("Language")
# # ethnicity_col = find_column("Ethnicity")
# # marital_col = find_column(
# #     "MaritalStatus",
# #     "Marital Status",
# #     "Marital"
# # )


# # # ============================================================
# # # 3. CREATE AGE GROUPS
# # # ============================================================

# # df["Age_Group"] = pd.cut(
# #     df[age_col],
# #     bins=[-np.inf] + AGE_BINS + [np.inf],
# #     labels=[0, 1, 2, 3]
# # ).astype("Int64")


# # ATTR_COLS = {
# #     "Gender": gender_col,
# #     "Age": "Age_Group",
# #     "Race": race_col,
# #     "Language": language_col,
# #     "Ethnicity": ethnicity_col,
# #     "Marital": marital_col,
# # }


# # print("\nDemographic columns used:")
# # for k, v in ATTR_COLS.items():
# #     print(f"  {k:15s} -> {v}")


# # print("\nAge-group distribution:")
# # print(df["Age_Group"].value_counts(dropna=False).sort_index())


# # # ============================================================
# # # 4. FIND PREDICTION FILES
# # # ============================================================

# # def find_prediction_file(method):
# #     """
# #     Find prediction.pkl for the method.

# #     If multiple files exist, print them and use the newest one.
# #     """

# #     folder = METHOD_DIR_MAP[method]

# #     pattern = os.path.join(
# #         RESULTS_DIR,
# #         folder,
# #         "**",
# #         "predictions.pkl"
# #     )

# #     matches = glob.glob(pattern, recursive=True)

# #     if not matches:
# #         return None

# #     matches = sorted(
# #         matches,
# #         key=os.path.getmtime,
# #         reverse=True
# #     )

# #     if len(matches) > 1:
# #         print(f"\n[{method}] Multiple prediction files found:")
# #         for p in matches:
# #             print("   ", p)
# #         print("Using newest:", matches[0])

# #     return matches[0]


# # # ============================================================
# # # 5. LOAD PREDICTIONS
# # # ============================================================

# # def extract_array(data, primary, secondary):
# #     """
# #     Safely extract NumPy arrays without using:
# #         a or b
# #     because NumPy arrays do not have a simple truth value.
# #     """

# #     if primary in data:
# #         return np.asarray(data[primary])

# #     if secondary in data:
# #         return np.asarray(data[secondary])

# #     raise KeyError(
# #         f"Neither '{primary}' nor '{secondary}' found.\n"
# #         f"Available keys: {list(data.keys())}"
# #     )


# # def load_predictions(method):

# #     path = find_prediction_file(method)

# #     if path is None:
# #         print(f"\n[{method}] [MISSING] predictions.pkl")
# #         return None, None, None

# #     print(f"\n[{method}]")
# #     print("Prediction file:")
# #     print(path)

# #     with open(path, "rb") as f:
# #         data = pickle.load(f)

# #     logits = extract_array(
# #         data,
# #         "logits",
# #         "predictions"
# #     )

# #     labels = extract_array(
# #         data,
# #         "label",
# #         "labels"
# #     ).astype(int).ravel()

# #     print("  logits shape:", logits.shape)
# #     print("  labels shape:", labels.shape)

# #     # --------------------------------------------------------
# #     # Convert to positive-class probability
# #     # --------------------------------------------------------

# #     if logits.ndim == 2 and logits.shape[1] >= 2:

# #         probs = scipy.special.softmax(
# #             logits,
# #             axis=-1
# #         )[:, 1]

# #     elif logits.ndim == 1:

# #         probs = scipy.special.expit(
# #             logits
# #         )

# #     else:

# #         raise ValueError(
# #             f"Unexpected logits shape: {logits.shape}"
# #         )

# #     probs = np.asarray(probs).ravel()

# #     # --------------------------------------------------------
# #     # IMPORTANT: verify prediction length
# #     # --------------------------------------------------------

# #     if len(probs) != len(df):

# #         raise ValueError(
# #             f"{method}: prediction length ({len(probs)}) "
# #             f"does not match test CSV ({len(df)})."
# #         )

# #     # --------------------------------------------------------
# #     # Verify labels against CSV if possible
# #     # --------------------------------------------------------

# #     # We cannot blindly assume a metadata label column name,
# #     # so try common possibilities.
# #     label_candidates = [
# #         "label",
# #         "Label",
# #         "target",
# #         "Target",
# #         "y",
# #         "Y",
# #     ]

# #     csv_label_col = None

# #     for c in label_candidates:
# #         if c in df.columns:
# #             csv_label_col = c
# #             break

# #     if csv_label_col is not None:

# #         csv_labels = df[csv_label_col].astype(int).to_numpy()

# #         if np.array_equal(labels, csv_labels):

# #             print("  ✓ Prediction labels match test.csv exactly.")

# #         else:

# #             mismatch = np.sum(labels != csv_labels)

# #             print(
# #                 f"  WARNING: {mismatch} label positions "
# #                 f"do not match test.csv!"
# #             )

# #     print(
# #         f"  n={len(labels)}, "
# #         f"positive rate={labels.mean():.4f}"
# #     )

# #     return probs, labels, path


# # # ============================================================
# # # 6. INTERSECTION ANALYSIS
# # # ============================================================

# # def analyze_intersection(
# #     probs,
# #     labels,
# #     a1,
# #     a2,
# #     min_n=10
# # ):

# #     cats1 = sorted(
# #         pd.Series(a1).dropna().unique().tolist()
# #     )

# #     cats2 = sorted(
# #         pd.Series(a2).dropna().unique().tolist()
# #     )

# #     cells = []

# #     valid_aucs = []

# #     has_problem = False

# #     for c1 in cats1:

# #         for c2 in cats2:

# #             mask = (
# #                 (a1 == c1) &
# #                 (a2 == c2)
# #             )

# #             n = int(mask.sum())

# #             if n == 0:

# #                 status = "EMPTY"
# #                 auc = np.nan
# #                 has_problem = True

# #             elif n < min_n:

# #                 status = "SPARSE"
# #                 auc = np.nan
# #                 has_problem = True

# #             else:

# #                 y_sub = labels[mask]

# #                 if len(np.unique(y_sub)) < 2:

# #                     status = "SINGLE_CLASS"
# #                     auc = np.nan
# #                     has_problem = True

# #                 else:

# #                     auc = roc_auc_score(
# #                         y_sub,
# #                         probs[mask]
# #                     )

# #                     status = "VALID"

# #                     valid_aucs.append(auc)

# #             cells.append({
# #                 "Attribute 1": a1.name,
# #                 "Attribute 2": a2.name,
# #                 "Group 1": c1,
# #                 "Group 2": c2,
# #                 "N": n,
# #                 "Status": status,
# #                 "AUC": auc,
# #             })

# #     # --------------------------------------------------------
# #     # Strict interpretation:
# #     # if ANY expected subgroup is non-evaluable,
# #     # don't report Δ-AUC for the intersection.
# #     # --------------------------------------------------------

# #     if has_problem:

# #         delta_auc = np.nan

# #     elif len(valid_aucs) >= 2:

# #         delta_auc = (
# #             max(valid_aucs) -
# #             min(valid_aucs)
# #         )

# #     else:

# #         delta_auc = np.nan

# #     return delta_auc, cells


# # # ============================================================
# # # 7. RUN ALL METHODS × INTERSECTIONS
# # # ============================================================

# # print("\n")
# # print("=" * 100)
# # print("INTERSECTIONAL ANALYSIS")
# # print("=" * 100)

# # result_records = []
# # cell_records = []


# # for method in METHODS:

# #     probs, labels, pred_path = load_predictions(method)

# #     if probs is None:
# #         continue

# #     print("\n" + "-" * 100)
# #     print(f"METHOD: {method}")
# #     print("-" * 100)

# #     for a1_name, a2_name in INTERSECTIONS:

# #         a1 = df[ATTR_COLS[a1_name]]
# #         a2 = df[ATTR_COLS[a2_name]]

# #         a1 = a1.rename(a1_name)
# #         a2 = a2.rename(a2_name)

# #         delta_auc, cells = analyze_intersection(
# #             probs,
# #             labels,
# #             a1,
# #             a2,
# #             min_n=MIN_N
# #         )

# #         # ----------------------------------------------------
# #         # Save cell-level information
# #         # ----------------------------------------------------

# #         for cell in cells:

# #             cell["Method"] = method
# #             cell_records.append(cell)

# #         # ----------------------------------------------------
# #         # Determine problematic cells
# #         # ----------------------------------------------------

# #         problems = [
# #             c for c in cells
# #             if c["Status"] != "VALID"
# #         ]

# #         empty = [
# #             c for c in cells
# #             if c["Status"] == "EMPTY"
# #         ]

# #         sparse = [
# #             c for c in cells
# #             if c["Status"] == "SPARSE"
# #         ]

# #         single_class = [
# #             c for c in cells
# #             if c["Status"] == "SINGLE_CLASS"
# #         ]

# #         if np.isnan(delta_auc):

# #             status = "NOT_EVALUABLE"

# #         else:

# #             status = "VALID"

# #         result_records.append({
# #             "Method": method,
# #             "Intersection": f"{a1_name} × {a2_name}",
# #             "Delta_AUC": delta_auc,
# #             "Status": status,
# #             "N_problematic_cells": len(problems),
# #             "N_empty": len(empty),
# #             "N_sparse": len(sparse),
# #             "N_single_class": len(single_class),
# #             "N_valid_cells": sum(
# #                 c["Status"] == "VALID"
# #                 for c in cells
# #             ),
# #             "N_total_cells": len(cells),
# #         })

# #         # ----------------------------------------------------
# #         # Print useful diagnostic
# #         # ----------------------------------------------------

# #         print(
# #             f"{a1_name:12s} × {a2_name:12s} | "
# #             f"Δ-AUC = "
# #             f"{'—' if np.isnan(delta_auc) else f'{delta_auc:.4f}'} | "
# #             f"{status}"
# #         )

# #         if problems:

# #             for c in problems:

# #                 print(
# #                     f"    {c['Group 1']} × "
# #                     f"{c['Group 2']} : "
# #                     f"N={c['N']} "
# #                     f"[{c['Status']}]"
# #                 )


# # # ============================================================
# # # 8. SAVE RESULTS
# # # ============================================================

# # results_df = pd.DataFrame(result_records)

# # cells_df = pd.DataFrame(cell_records)


# # results_path = os.path.join(
# #     OUTPUT_DIR,
# #     "intersectional_results.csv"
# # )

# # cells_path = os.path.join(
# #     OUTPUT_DIR,
# #     "intersectional_cell_details.csv"
# # )

# # results_df.to_csv(
# #     results_path,
# #     index=False
# # )

# # cells_df.to_csv(
# #     cells_path,
# #     index=False
# # )


# # # ============================================================
# # # 9. HUMAN-READABLE PROBLEM REPORT
# # # ============================================================

# # problem_rows = []

# # for _, row in cells_df.iterrows():

# #     if row["Status"] != "VALID":

# #         problem_rows.append(row)


# # problem_df = pd.DataFrame(problem_rows)

# # problem_path = os.path.join(
# #     OUTPUT_DIR,
# #     "intersectional_empty_report.csv"
# # )

# # problem_df.to_csv(
# #     problem_path,
# #     index=False
# # )


# # print("\n")
# # print("=" * 100)
# # print("PROBLEMATIC / DEGENERATE CELLS")
# # print("=" * 100)

# # if len(problem_df) == 0:

# #     print("✓ No empty, sparse, or single-class cells found.")

# # else:

# #     display_cols = [
# #         "Method",
# #         "Attribute 1",
# #         "Attribute 2",
# #         "Group 1",
# #         "Group 2",
# #         "N",
# #         "Status",
# #     ]

# #     print(
# #         problem_df[
# #             display_cols
# #         ].to_string(index=False)
# #     )


# # # ============================================================
# # # 10. CREATE PIVOT TABLE
# # # ============================================================

# # row_order = [
# #     f"{a1} × {a2}"
# #     for a1, a2 in INTERSECTIONS
# # ]

# # pivot_df = results_df.pivot(
# #     index="Intersection",
# #     columns="Method",
# #     values="Delta_AUC"
# # )

# # pivot_df = pivot_df.reindex(
# #     index=row_order,
# #     columns=METHODS
# # )


# # print("\n")
# # print("=" * 100)
# # print("FINAL INTERSECTIONAL Δ-AUC MATRIX")
# # print("=" * 100)

# # print(
# #     pivot_df.to_string(
# #         float_format=lambda x: f"{x:.4f}"
# #     )
# # )


# # # ============================================================
# # # 11. IDENTIFY WINNERS
# # # ============================================================

# # # Lower Δ-AUC = better fairness.

# # winner_mask = pd.DataFrame(
# #     False,
# #     index=pivot_df.index,
# #     columns=pivot_df.columns
# # )

# # for idx in pivot_df.index:

# #     row = pivot_df.loc[idx]

# #     valid = row.dropna()

# #     if len(valid) > 0:

# #         minimum = valid.min()

# #         for method in valid.index:

# #             if np.isclose(
# #                 valid[method],
# #                 minimum,
# #                 atol=1e-10
# #             ):

# #                 winner_mask.loc[
# #                     idx,
# #                     method
# #                 ] = True


# # # ============================================================
# # # 12. PLOT HEATMAP
# # # ============================================================

# # sns.set_theme(style="white")

# # fig, ax = plt.subplots(
# #     figsize=(14, 10)
# # )


# # # ------------------------------------------------------------
# # # Base heatmap
# # # ------------------------------------------------------------

# # sns.heatmap(
# #     pivot_df,
# #     mask=pivot_df.isna(),
# #     cmap="Reds",
# #     annot=True,
# #     fmt=".2f",
# #     linewidths=0.7,
# #     linecolor="white",
# #     cbar_kws={
# #         "label": r"$\Delta$AUC (Max − Min)"
# #     },
# #     ax=ax
# # )


# # # ------------------------------------------------------------
# # # Highlight winners
# # # ------------------------------------------------------------

# # winner_values = pivot_df.where(
# #     winner_mask
# # )

# # sns.heatmap(
# #     winner_values,
# #     mask=~winner_mask,
# #     cmap="Greens",
# #     annot=True,
# #     fmt=".2f",
# #     linewidths=0.7,
# #     linecolor="white",
# #     cbar=False,
# #     ax=ax
# # )


# # # ------------------------------------------------------------
# # # Mark non-evaluable cells
# # # ------------------------------------------------------------

# # for y in range(pivot_df.shape[0]):

# #     for x in range(pivot_df.shape[1]):

# #         if pd.isna(
# #             pivot_df.iloc[y, x]
# #         ):

# #             ax.text(
# #                 x + 0.5,
# #                 y + 0.5,
# #                 "—",
# #                 ha="center",
# #                 va="center",
# #                 fontsize=15,
# #                 fontweight="bold"
# #             )


# # # ------------------------------------------------------------
# # # Formatting
# # # ------------------------------------------------------------

# # ax.set_title(
# #     "Intersectional Fairness: Subgroup AUC Gap\n"
# #     "Green = Lowest Disparity; — = Non-evaluable / Degenerate",
# #     fontsize=15,
# #     pad=18,
# #     fontweight="bold"
# # )

# # ax.set_xlabel(
# #     "Adaptation Strategy",
# #     fontsize=13,
# #     labelpad=10
# # )

# # ax.set_ylabel(
# #     "Demographic Intersection",
# #     fontsize=13,
# #     labelpad=10
# # )

# # plt.xticks(
# #     rotation=35,
# #     ha="right",
# #     fontsize=11
# # )

# # plt.yticks(
# #     rotation=0,
# #     fontsize=11
# # )

# # plt.tight_layout()


# # output_img = os.path.join(
# #     OUTPUT_DIR,
# #     "fig_intersectional_heatmap_clean.png"
# # )

# # plt.savefig(
# #     output_img,
# #     dpi=300,
# #     bbox_inches="tight"
# # )

# # plt.show()


# # # ============================================================
# # # 13. FINAL SUMMARY
# # # ============================================================

# # print("\n")
# # print("=" * 100)
# # print("FILES SAVED")
# # print("=" * 100)

# # print(results_path)
# # print(cells_path)
# # print(problem_path)
# # print(output_img)

# # print("\n✓ Intersectional analysis complete.")

# #!/usr/bin/env python3
# #!/usr/bin/env python3#!/usr/bin/env python3
# #!/usr/bin/env python3
# #!/usr/bin/env python3
# #!/usr/bin/env python3

# """
# Clean intersectional fairness evaluation for FairVLMed10k.

# Outputs:
#     intersectional_results.csv
#     intersectional_cell_details.csv
#     intersectional_empty_report.csv
#     fig_intersectional_heatmap_robust.png
# """

# import os
# import glob
# import pickle
# import numpy as np
# import pandas as pd
# import scipy.special
# import matplotlib.colors as mcolors

# from sklearn.metrics import roc_auc_score
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ============================================================
# # 1. CONFIGURATION
# # ============================================================

# PROJECT_ROOT = (
#     "/home/alaa/papersubmission/"
#     "The-Impact-of-Adaptation-on-Fairness-in-Medical-Multimodal-"
#     "Foundation-Models"
# )

# TEST_CSV = os.path.join(PROJECT_ROOT, "Harvard FairVLMed (Harvard-FairVLMed）", "test.csv")
# RESULTS_DIR = os.path.join(PROJECT_ROOT, "FairVLMed10k_clip_results")
# OUTPUT_DIR = PROJECT_ROOT

# MIN_N = 10
# AGE_BINS = [51.2, 63.9, 73.3]

# METHODS = [
#     "clip-zs", "lp", "clip-adapt", "lora-lp", "lora-text", "full-ft", "full-ft-text"
# ]

# METHOD_DIR_MAP = {
#     "clip-zs": "clip-zs_lr1e-2_seed1",
#     "lp": "lp_lr1e-2_seed1",
#     "clip-adapt": "clip-adapt_lr1e-2_seed1",
#     "lora-lp": "lora-lp_lr1e-2_seed1",
#     "lora-text": "lora-zs_lr1e-2_seed1",
#     "full-ft": "full-ft_lr5e-3_seed1",
#     "full-ft-text": "full-ft-zs_lr5e-3_seed1",
# }

# INTERSECTIONS = [
#     ("Age", "Marital"), ("Race", "Language"), ("Gender", "Marital"),
#     ("Language", "Marital"), ("Age", "Language"), ("Ethnicity", "Marital"),
#     ("Race", "Marital"), ("Age", "Ethnicity"), ("Race", "Ethnicity"),
#     ("Age", "Race"), ("Language", "Ethnicity"), ("Gender", "Language"),
#     ("Gender", "Race"), ("Gender", "Ethnicity"), ("Gender", "Age"),
# ]

# # ============================================================
# # 2-5. LOAD METADATA AND PREDICTIONS
# # ============================================================

# df = pd.read_csv(TEST_CSV)
# lower_cols = {c.lower(): c for c in df.columns}

# def find_column(*names):
#     for name in names:
#         if name.lower() in lower_cols:
#             return lower_cols[name.lower()]
#     raise KeyError(f"Could not find columns: {names}")

# gender_col = find_column("Gender", "Sex")
# age_col = find_column("Age")
# race_col = find_column("Race")
# language_col = find_column("Language")
# ethnicity_col = find_column("Ethnicity")
# marital_col = find_column("MaritalStatus", "Marital Status", "Marital")

# df["Age_Group"] = pd.cut(df[age_col], bins=[-np.inf] + AGE_BINS + [np.inf], labels=[0, 1, 2, 3]).astype("Int64")

# ATTR_COLS = {
#     "Gender": gender_col, "Age": "Age_Group", "Race": race_col,
#     "Language": language_col, "Ethnicity": ethnicity_col, "Marital": marital_col,
# }

# def find_prediction_file(method):
#     pattern = os.path.join(RESULTS_DIR, METHOD_DIR_MAP[method], "**", "predictions.pkl")
#     matches = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime, reverse=True)
#     return matches[0] if matches else None

# def extract_array(data, primary, secondary):
#     if primary in data: return np.asarray(data[primary])
#     if secondary in data: return np.asarray(data[secondary])
#     raise KeyError("Data missing")

# def load_predictions(method):
#     path = find_prediction_file(method)
#     if not path: return None, None, None
#     with open(path, "rb") as f: data = pickle.load(f)
    
#     logits = extract_array(data, "logits", "predictions")
#     labels = extract_array(data, "label", "labels").astype(int).ravel()
    
#     if logits.ndim == 2 and logits.shape[1] >= 2:
#         probs = scipy.special.softmax(logits, axis=-1)[:, 1]
#     else:
#         probs = scipy.special.expit(logits)
        
#     return np.asarray(probs).ravel(), labels, path

# # ============================================================
# # 6. INTERSECTION ANALYSIS
# # ============================================================

# def analyze_intersection(probs, labels, a1, a2, min_n=10):
#     cats1 = sorted(pd.Series(a1).dropna().unique().tolist())
#     cats2 = sorted(pd.Series(a2).dropna().unique().tolist())

#     cells = []
#     valid_aucs = []
#     has_dropped_cells = False 

#     for c1 in cats1:
#         for c2 in cats2:
#             mask = ((a1 == c1) & (a2 == c2))
#             n = int(mask.sum())

#             if n == 0:
#                 status, auc = "EMPTY", np.nan
#                 has_dropped_cells = True
#             elif n < min_n:
#                 status, auc = "SPARSE", np.nan
#                 has_dropped_cells = True
#             else:
#                 y_sub = labels[mask]
#                 if len(np.unique(y_sub)) < 2:
#                     status, auc = "SINGLE_CLASS", np.nan
#                     has_dropped_cells = True
#                 else:
#                     auc = roc_auc_score(y_sub, probs[mask])
#                     status = "VALID"
#                     valid_aucs.append(auc)

#             cells.append({
#                 "Attribute 1": a1.name, "Attribute 2": a2.name,
#                 "Group 1": c1, "Group 2": c2,
#                 "N": n, "Status": status, "AUC": auc,
#             })

#     delta_auc = (max(valid_aucs) - min(valid_aucs)) if len(valid_aucs) >= 2 else np.nan
#     return delta_auc, cells, has_dropped_cells

# # ============================================================
# # 7. RUN ALL METHODS × INTERSECTIONS
# # ============================================================

# result_records = []
# cell_records = []
# dropped_intersections = set() 

# for method in METHODS:
#     probs, labels, _ = load_predictions(method)
#     if probs is None: continue

#     for a1_name, a2_name in INTERSECTIONS:
#         a1, a2 = df[ATTR_COLS[a1_name]].rename(a1_name), df[ATTR_COLS[a2_name]].rename(a2_name)
        
#         delta_auc, cells, has_dropped_cells = analyze_intersection(probs, labels, a1, a2, min_n=MIN_N)
        
#         intersection_name = f"{a1_name} × {a2_name}"
#         if has_dropped_cells:
#             dropped_intersections.add(intersection_name)

#         for cell in cells:
#             cell["Method"] = method
#             cell_records.append(cell)

#         result_records.append({
#             "Method": method,
#             "Intersection_Base": intersection_name,
#             "Delta_AUC": delta_auc,
#         })

# results_df = pd.DataFrame(result_records)
# results_df["Intersection"] = results_df["Intersection_Base"].apply(
#     lambda x: f"{x} *" if x in dropped_intersections else x
# )

# # ============================================================
# # 8. CREATE PIVOT TABLES AND CUSTOM ANNOTATIONS
# # ============================================================

# row_order = [f"{a1} × {a2} *" if f"{a1} × {a2}" in dropped_intersections else f"{a1} × {a2}" for a1, a2 in INTERSECTIONS]

# pivot_df = results_df.pivot(index="Intersection", columns="Method", values="Delta_AUC").reindex(index=row_order, columns=METHODS)

# annot_matrix = np.empty(pivot_df.shape, dtype=object)
# for i in range(pivot_df.shape[0]):
#     for j in range(pivot_df.shape[1]):
#         val = pivot_df.iloc[i, j]
#         annot_matrix[i, j] = "" if pd.isna(val) else f"{val:.2f}"

# # ============================================================
# # 9. PLOT HEATMAP
# # ============================================================

# sns.set_theme(style="white")
# fig, ax = plt.subplots(figsize=(14, 10))

# # Identify winners
# winner_mask = pd.DataFrame(False, index=pivot_df.index, columns=pivot_df.columns)
# for idx in pivot_df.index:
#     valid = pivot_df.loc[idx].dropna()
#     if len(valid) > 0:
#         minimum = valid.min()
#         for method in valid.index:
#             if np.isclose(valid[method], minimum, atol=1e-10):
#                 winner_mask.loc[idx, method] = True

# sns.heatmap(
#     pivot_df, mask=pivot_df.isna(), cmap="Reds", 
#     annot=annot_matrix, fmt="", linewidths=0.7, linecolor="white",
#     cbar_kws={"label": r"$\Delta$AUC (Max − Min)"}, ax=ax
# )

# solid_green = mcolors.ListedColormap(['#2ca02c'])
# sns.heatmap(
#     pivot_df.where(winner_mask), mask=~winner_mask, cmap=solid_green, 
#     annot=annot_matrix, fmt="", linewidths=0.7, linecolor="white", cbar=False, ax=ax
# )

# for y in range(pivot_df.shape[0]):
#     for x in range(pivot_df.shape[1]):
#         if pd.isna(pivot_df.iloc[y, x]):
#             ax.text(x + 0.5, y + 0.5, "—", ha="center", va="center", fontsize=15, fontweight="bold", color="gray")

# ax.set_title("Intersectional Fairness: Subgroup AUC Gap\nGreen = Lowest Disparity", fontsize=16, pad=18, fontweight="bold")
# ax.set_xlabel("Adaptation Strategy", fontsize=13, labelpad=10)
# ax.set_ylabel("Demographic Intersection", fontsize=13, labelpad=10)
# plt.xticks(rotation=35, ha="right", fontsize=11)
# plt.yticks(rotation=0, fontsize=11)
# plt.tight_layout()

# output_img = os.path.join(OUTPUT_DIR, "fig_intersectional_heatmap_robust.png")
# plt.savefig(output_img, dpi=300, bbox_inches="tight")
# plt.show()

# # ============================================================
# # 10. TERMINAL HEATMAP OUTPUT
# # ============================================================
# print("\n" + "=" * 120)
# print("FINAL ROBUST INTERSECTIONAL Δ-AUC MATRIX (TERMINAL HEATMAP)")
# print("=" * 120)
# print("Legend: [ 0.XXXX ] = Best Strategy (Lowest Gap) | '   —   ' = Degenerate/Non-evaluable")
# print("-" * 120)

# # Build a dataframe specifically formatted for terminal printing
# terminal_df = pd.DataFrame(index=pivot_df.index, columns=pivot_df.columns)
# for col in pivot_df.columns:
#     for idx in pivot_df.index:
#         val = pivot_df.loc[idx, col]
#         if pd.isna(val):
#             terminal_df.loc[idx, col] = "   —   "
#         else:
#             if winner_mask.loc[idx, col]:
#                 terminal_df.loc[idx, col] = f"[{val:.4f}]"  # Bracket the winners!
#             else:
#                 terminal_df.loc[idx, col] = f" {val:.4f} "

# print(terminal_df.to_string())
# print("=" * 120 + "\n")
# print(f"✓ Saved updated heatmap with Y-axis asterisks and solid green to {output_img}")


import pandas as pd
import numpy as np

# Load the cell details CSV generated by your main script
df = pd.read_csv("intersectional_cell_details.csv")

# Cell statuses (Valid/Empty/Sparse) are the same across methods because 
# the dataset and ground-truth labels don't change. We just grab the first method.
first_method = df["Method"].iloc[0]
df = df[df["Method"] == first_method].copy()

# Format the intersection name for LaTeX
df["Intersection"] = df["Attribute 1"] + r" $\times$ " + df["Attribute 2"]

# Start building the LaTeX table
lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{Intersectional subgroup cell details for FairVLMed10k. We evaluate all Cartesian combinations of subgroups for each pair of attributes. Cells were excluded if they were empty, contained fewer than 10 samples (sparse), or contained only a single diagnostic class (making AUC undefined).}",
    r"\label{tab:supp_intersections}",
    r"\renewcommand{\arraystretch}{1.1}",
    r"\small",
    r"\begin{tabular}{lcccc}",
    r"\toprule",
    r"\textbf{Intersection} & \textbf{Total Cells} & \textbf{Valid Cells} & \textbf{Excluded} & \textbf{Avg $N$ (Valid)} \\",
    r"\midrule"
]

# Calculate stats for each intersection
for intersection, group in df.groupby("Intersection", sort=False):
    total = len(group)
    valid = sum(group["Status"] == "VALID")
    excluded = total - valid
    
    # Calculate average N of the VALID cells
    valid_cells = group[group["Status"] == "VALID"]
    avg_n = int(round(valid_cells["N"].mean())) if len(valid_cells) > 0 else 0
    
    lines.append(f"{intersection} & {total} & {valid} & {excluded} & {avg_n} \\\\")

lines.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}"
])

# Save to a text file
output_file = "latex_intersectional_table.txt"
with open(output_file, "w") as f:
    f.write("\n".join(lines))

print(f"✓ LaTeX table generated and saved to {output_file}!")
print("Open this file and paste the contents into Section 4 of your supp.tex.")