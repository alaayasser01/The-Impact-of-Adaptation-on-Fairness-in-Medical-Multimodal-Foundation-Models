# # # import os, re, glob, json
# # # import numpy as np
# # # import pandas as pd
# # # import matplotlib.pyplot as plt

# # # SWEEP_DIR = "./lr_sweep"
# # # TEXT_SWEEP_DIR = "./text_sweep"
# # # BASELINE_DIR = "./baseline/clip_zs"
# # # OUT_DIR = "./poster_final_results"
# # # os.makedirs(OUT_DIR, exist_ok=True)

# # # # ------------------------------------------------------------
# # # # 1. Helper functions
# # # # ------------------------------------------------------------
# # # def find_history_log(exp_folder):
# # #     """Find the most recent history*.log file inside exp_folder."""
# # #     logs = glob.glob(os.path.join(exp_folder, "**", "history*.log"), recursive=True)
# # #     if not logs:
# # #         return None
# # #     return max(logs, key=os.path.getmtime)

# # # def extract_test_metrics(log_path):
# # #     """Extract overall-auc, male/female AUC, and EoD from the final test evaluation block."""
# # #     with open(log_path, 'r') as f:
# # #         content = f.read()
# # #     # Overall AUC (last occurrence)
# # #     overall_auc = re.findall(r"overall-auc: (0\.\d+)", content)
# # #     if not overall_auc:
# # #         return None
# # #     test_auc = float(overall_auc[-1])
# # #     # Subgroup AUCs (last block with "subgroup metrics: auc: [...]")
# # #     sub_blocks = list(re.finditer(r"subgroup metrics: auc: \[([0-9.,\s]+)\]", content))
# # #     if sub_blocks:
# # #         last_sub = sub_blocks[-1].group(1)
# # #         vals = [float(x.strip()) for x in last_sub.split(",")]
# # #         if len(vals) >= 2:
# # #             male_auc, female_auc = vals[0], vals[1]
# # #         else:
# # #             male_auc = female_auc = np.nan
# # #     else:
# # #         male_auc = female_auc = np.nan
# # #     # EoD
# # #     eod_match = re.search(r"eod: (0\.\d+)", content)
# # #     eod_val = float(eod_match.group(1)) if eod_match else np.nan
# # #     return {"test_auc": test_auc, "male_auc": male_auc, "female_auc": female_auc, "eod": eod_val}

# # # # ------------------------------------------------------------
# # # # 2. Load sweep summary and parameter counts
# # # # ------------------------------------------------------------
# # # sweep_csv = os.path.join(SWEEP_DIR, "sweep_summary_corrected.csv")
# # # df_sweep = pd.read_csv(sweep_csv)
# # # df_sweep = df_sweep[df_sweep["best_val_auc"].apply(
# # #     lambda x: isinstance(x, str) and x.replace('.','',1).isdigit()
# # # )].copy()
# # # df_sweep["best_val_auc"] = df_sweep["best_val_auc"].astype(float)
# # # df_sweep["best_epoch"] = df_sweep["best_epoch"].astype(int)

# # # param_counts = {
# # #     "lp": 1026,
# # #     "clip-adapt": 1024,
# # #     "full-ft": 151278339,
# # #     "full-ft-zs": 151277313,
# # #     "lora-lp": 1475586,
# # #     "lora-zs": 1474560,
# # # }
# # # df_sweep["trainable_params"] = df_sweep["usage"].map(param_counts)
# # # df_sweep["param_epochs"] = df_sweep["trainable_params"] * df_sweep["best_epoch"]

# # # best_per_usage = df_sweep.loc[df_sweep.groupby("usage")["best_val_auc"].idxmax()]

# # # # ------------------------------------------------------------
# # # # 3. Extract test metrics for template champions (from sweep logs)
# # # # ------------------------------------------------------------
# # # test_metrics_template = {}
# # # for _, row in best_per_usage.iterrows():
# # #     usage = row["usage"]
# # #     lr = row["lr"]
# # #     exp_dir = os.path.join(SWEEP_DIR, f"{usage}_lr{lr}_seed0")
# # #     log_file = find_history_log(exp_dir)
# # #     if log_file:
# # #         metrics = extract_test_metrics(log_file)
# # #         if metrics:
# # #             metrics["best_lr"] = lr
# # #             metrics["best_val_auc"] = row["best_val_auc"]
# # #             metrics["best_epoch"] = row["best_epoch"]
# # #             metrics["trainable_params"] = row["trainable_params"]
# # #             metrics["param_epochs"] = row["param_epochs"]
# # #             test_metrics_template[usage] = metrics

# # # # ------------------------------------------------------------
# # # # 3b. Extract baseline (clip‑zs) metrics
# # # # ------------------------------------------------------------
# # # baseline_log = find_history_log(BASELINE_DIR)
# # # clipzs_metrics = None
# # # if baseline_log:
# # #     clipzs_metrics = extract_test_metrics(baseline_log)
# # # if clipzs_metrics:
# # #     clipzs_metrics["best_lr"] = "–"
# # #     clipzs_metrics["best_val_auc"] = "–"
# # #     clipzs_metrics["best_epoch"] = "–"
# # #     clipzs_metrics["trainable_params"] = "–"
# # #     clipzs_metrics["param_epochs"] = "–"

# # # # ------------------------------------------------------------
# # # # 4. Extract test metrics for clustered champions (clip-adapt, full-ft-zs, lora-zs)
# # # # ------------------------------------------------------------
# # # clustered_usages = ["clip-adapt", "full-ft-zs", "lora-zs"]
# # # test_metrics_clustered = {}
# # # for usage in clustered_usages:
# # #     exp_dir = os.path.join(TEXT_SWEEP_DIR, f"{usage}_clustered")
# # #     log_file = find_history_log(exp_dir)
# # #     if log_file:
# # #         metrics = extract_test_metrics(log_file)
# # #         if metrics:
# # #             test_metrics_clustered[usage] = metrics

# # # # ------------------------------------------------------------
# # # # 5. Table 1: Usage Comparison (template) – includes baseline
# # # # ------------------------------------------------------------
# # # table1_rows = []
# # # # Baseline first
# # # if clipzs_metrics:
# # #     table1_rows.append({
# # #         "Usage": "clip‑zs (zero‑shot)",
# # #         "Best LR": "–",
# # #         "Val AUC": "–",
# # #         "Best Epoch": "–",
# # #         "Param-Epochs (M)": "–",
# # #         "Test AUC": f'{clipzs_metrics["test_auc"]:.4f}',
# # #         "Male AUC": f'{clipzs_metrics["male_auc"]:.4f}',
# # #         "Female AUC": f'{clipzs_metrics["female_auc"]:.4f}',
# # #         "EoD": f'{clipzs_metrics["eod"]:.4f}',
# # #     })
# # # for usage, m in test_metrics_template.items():
# # #     table1_rows.append({
# # #         "Usage": usage,
# # #         "Best LR": m["best_lr"],
# # #         "Val AUC": f'{m["best_val_auc"]:.4f}',
# # #         "Best Epoch": m["best_epoch"],
# # #         "Param-Epochs (M)": f'{m["param_epochs"]/1e6:.1f}',
# # #         "Test AUC": f'{m["test_auc"]:.4f}',
# # #         "Male AUC": f'{m["male_auc"]:.4f}',
# # #         "Female AUC": f'{m["female_auc"]:.4f}',
# # #         "EoD": f'{m["eod"]:.4f}',
# # #     })
# # # df_table1 = pd.DataFrame(table1_rows)
# # # df_table1.to_csv(os.path.join(OUT_DIR, "table1_usage_comparison.csv"), index=False)

# # # # ------------------------------------------------------------
# # # # 6. Table 2: Text-Source Impact (template vs clustered)
# # # # ------------------------------------------------------------
# # # table2_rows = []
# # # for usage in clustered_usages:
# # #     if usage in test_metrics_template:
# # #         tm = test_metrics_template[usage]
# # #         table2_rows.append({
# # #             "Usage": usage,
# # #             "Source": "template",
# # #             "Test AUC": f'{tm["test_auc"]:.4f}',
# # #             "AUC-gap": f'{abs(tm["male_auc"] - tm["female_auc"]):.4f}',
# # #             "EoD": f'{tm["eod"]:.4f}',
# # #         })
# # #     if usage in test_metrics_clustered:
# # #         cm = test_metrics_clustered[usage]
# # #         table2_rows.append({
# # #             "Usage": usage,
# # #             "Source": "clustered",
# # #             "Test AUC": f'{cm["test_auc"]:.4f}',
# # #             "AUC-gap": f'{abs(cm["male_auc"] - cm["female_auc"]):.4f}',
# # #             "EoD": f'{cm["eod"]:.4f}',
# # #         })
# # # df_table2 = pd.DataFrame(table2_rows)
# # # df_table2.to_csv(os.path.join(OUT_DIR, "table2_text_source_impact.csv"), index=False)

# # # # ------------------------------------------------------------
# # # # 7. Table 3: Prototype Discriminability
# # # # ------------------------------------------------------------
# # # prototype_rows = [
# # #     {"Source": "template", "Similarity": "0.9667"},
# # #     {"Source": "gpt4_avg", "Similarity": "0.9991"},
# # #     {"Source": "clustered (max)", "Similarity": "0.9979"},
# # #     {"Source": "clustered (median)", "Similarity": "0.9205"},
# # #     {"Source": "clustered (min)", "Similarity": "0.8783"},
# # # ]
# # # df_proto = pd.DataFrame(prototype_rows)
# # # df_proto.to_csv(os.path.join(OUT_DIR, "table_prototype_similarity.csv"), index=False)

# # # # ------------------------------------------------------------
# # # # 8. Intersectional Table (manual, from your saved results)
# # # # ------------------------------------------------------------
# # # intersectional_rows = [
# # #     {
# # #         "Method": "clip‑zs",
# # #         "gender×race gap": "0.1282",
# # #         "maritalstatus×race gap": "0.4576",
# # #         "language×race gap": "1.0000* (small subgroups)",
# # #     },
# # #     {
# # #         "Method": "full-ft-zs (template)",
# # #         "gender×race gap": "0.1366",
# # #         "maritalstatus×race gap": "0.6275",
# # #         "language×race gap": "0.4584",
# # #     },
# # # ]
# # # pd.DataFrame(intersectional_rows).to_csv(
# # #     os.path.join(OUT_DIR, "table_intersectional.csv"), index=False
# # # )

# # # # ------------------------------------------------------------
# # # # 9. Learning Curves (validation AUC vs epoch) for the best LR of each usage
# # # # ------------------------------------------------------------
# # # plt.figure(figsize=(10,6))
# # # for _, row in best_per_usage.iterrows():
# # #     usage = row["usage"]
# # #     lr = row["lr"]
# # #     exp_dir = os.path.join(SWEEP_DIR, f"{usage}_lr{lr}_seed0")
# # #     log_file = find_history_log(exp_dir)
# # #     if not log_file:
# # #         continue
# # #     with open(log_file, 'r') as f:
# # #         content = f.read()
# # #     matches = re.findall(r"epoch (\d+): val AUC = (0\.\d+)", content)
# # #     if not matches:
# # #         continue
# # #     epochs, aucs = zip(*[(int(e), float(a)) for e, a in matches])
# # #     plt.plot(epochs, aucs, label=f"{usage} (LR={lr})")
# # # plt.xlabel("Epoch")
# # # plt.ylabel("Validation AUC")
# # # plt.title("Learning Curves (Validation AUC, best LR per usage)")
# # # plt.legend()
# # # plt.grid(True)
# # # plt.tight_layout()
# # # plt.savefig(os.path.join(OUT_DIR, "learning_curves.png"), dpi=150)
# # # plt.close()

# # # # ------------------------------------------------------------
# # # # 10. AUC vs EoD scatter plot (template methods + baseline)
# # # # ------------------------------------------------------------
# # # plot_aucs = []
# # # plot_eods = []
# # # labels = []
# # # if clipzs_metrics:
# # #     plot_aucs.append(clipzs_metrics["test_auc"])
# # #     plot_eods.append(clipzs_metrics["eod"])
# # #     labels.append("clip‑zs")
# # # for usage, m in test_metrics_template.items():
# # #     plot_aucs.append(m["test_auc"])
# # #     plot_eods.append(m["eod"])
# # #     labels.append(usage)

# # # plt.figure(figsize=(8,6))
# # # plt.scatter(plot_aucs, plot_eods, s=100)
# # # for i, lab in enumerate(labels):
# # #     plt.annotate(lab, (plot_aucs[i], plot_eods[i]),
# # #                  textcoords="offset points", xytext=(5,5))
# # # plt.xlabel("Test AUC")
# # # plt.ylabel("EoD (higher is better)")
# # # plt.title("Performance-Fairness Trade‑off (Template Source)")
# # # plt.grid(True)
# # # plt.tight_layout()
# # # plt.savefig(os.path.join(OUT_DIR, "auc_vs_eod.png"), dpi=150)
# # # plt.close()

# # # # ------------------------------------------------------------
# # # # 11. Cost bar chart (Parameter-Epochs) – excluding baseline
# # # # ------------------------------------------------------------
# # # usages_cost = list(test_metrics_template.keys())
# # # costs = [test_metrics_template[u]["param_epochs"] for u in usages_cost]
# # # plt.figure(figsize=(10,5))
# # # plt.bar(usages_cost, costs, color='skyblue')
# # # plt.ylabel("Parameter‑Epochs")
# # # plt.title("Training Cost (Sustainability) – Lower is Better")
# # # plt.xticks(rotation=45)
# # # plt.tight_layout()
# # # plt.savefig(os.path.join(OUT_DIR, "cost_bar.png"), dpi=150)
# # # plt.close()

# # # print("✅ All outputs saved to ./poster_final_results/")
# # # print("   - table1_usage_comparison.csv")
# # # print("   - table2_text_source_impact.csv")
# # # print("   - table_prototype_similarity.csv")
# # # print("   - table_intersectional.csv")
# # # print("   - learning_curves.png")
# # # print("   - auc_vs_eod.png")
# # # print("   - cost_bar.png")

# # import os, re, glob, json
# # import pandas as pd
# # import numpy as np

# # SWEEP_DIR = "./lr_sweep"
# # BASELINE_DIR = "./baseline/clip_zs"
# # OUT_DIR = "./poster_data"
# # os.makedirs(OUT_DIR, exist_ok=True)

# # # Attributes evaluated in --sensitive_name All
# # ATTRIBUTES = ["Sex", "Age", "Race", "Language", "Ethnicity", "MaritalStatus"]

# # # Usages we want in the summary (template source)
# # USAGES = ["lp", "clip-adapt", "full-ft", "full-ft-zs", "lora-lp", "lora-zs"]

# # # Best LRs (from sweep)
# # BEST_LRS = {
# #     "lp": "1e-2",
# #     "clip-adapt": "1e-2",
# #     "full-ft": "1e-2",
# #     "full-ft-zs": "5e-3",
# #     "lora-lp": "1e-2",
# #     "lora-zs": "1e-2",
# # }

# # # Parameter counts (from previous run)
# # PARAM_COUNTS = {
# #     "lp": 1026,
# #     "clip-adapt": 1024,
# #     "full-ft": 151278339,
# #     "full-ft-zs": 151277313,
# #     "lora-lp": 1475586,
# #     "lora-zs": 1474560,
# # }

# # # Best epochs (from sweep CSV, we can extract from logs as well but we know them)
# # BEST_EPOCHS = {
# #     "lp": 88,
# #     "clip-adapt": 87,
# #     "full-ft": 8,
# #     "full-ft-zs": 13,
# #     "lora-lp": 25,
# #     "lora-zs": 28,
# # }

# # # Helper: find the most recent history*.log inside a folder
# # def find_latest_log(folder):
# #     logs = glob.glob(os.path.join(folder, "**", "history*.log"), recursive=True)
# #     return max(logs, key=os.path.getmtime) if logs else None

# # # Helper: extract overall test AUC and EoD from a log for a specific attribute evaluation
# # def extract_metrics_for_attribute(log_text, attr):
# #     # Find the test evaluation block for this attribute.
# #     # The log contains sections like "--- Evaluating Sex ---" followed by "overall-auc: ..."
# #     # We'll find the block that starts with "--- Evaluating {attr} ---" and then capture the metrics.
# #     pattern = re.compile(
# #         rf"--- Evaluating {attr} ---.*?"
# #         r"overall-auc:\s*([\d.]+).*?"
# #         r"eod:\s*([\d.]+)",
# #         re.DOTALL
# #     )
# #     m = pattern.search(log_text)
# #     if m:
# #         return float(m.group(1)), float(m.group(2))
# #     return None, None

# # # ------------------------------------------------------------------
# # # 1. Extract metrics for all attributes from each champion (template)
# # # ------------------------------------------------------------------
# # def get_all_attr_metrics(exp_folder):
# #     log_file = find_latest_log(exp_folder)
# #     if not log_file:
# #         return None
# #     with open(log_file, 'r') as f:
# #         content = f.read()
# #     metrics = {}
# #     for attr in ATTRIBUTES:
# #         auc, eod = extract_metrics_for_attribute(content, attr)
# #         if auc is not None:
# #             metrics[attr] = {"AUC": auc, "EoD": eod}
# #     return metrics

# # # Baseline (clip‑zs) – from ./baseline/clip_zs
# # baseline_metrics = get_all_attr_metrics(BASELINE_DIR)

# # # For each usage, get the metrics from its sweep folder (the eval_only All output is stored there)
# # usage_metrics = {}
# # for usage in USAGES:
# #     lr = BEST_LRS[usage]
# #     exp_dir = os.path.join(SWEEP_DIR, f"{usage}_lr{lr}_seed0")
# #     usage_metrics[usage] = get_all_attr_metrics(exp_dir)

# # # ------------------------------------------------------------------
# # # 2. Build the Summary Table CSV (for Table 1 / Executive Briefing)
# # # ------------------------------------------------------------------
# # summary_rows = []
# # # Add baseline
# # if baseline_metrics:
# #     baseline_auc = np.mean([v["AUC"] for v in baseline_metrics.values()])
# #     baseline_eod = np.mean([v["EoD"] for v in baseline_metrics.values()])
# #     summary_rows.append({
# #         "Usage": "clip‑zs",
# #         "Best LR": "–",
# #         "Param-Epochs (M)": 0,
# #         "Overall Test AUC": baseline_auc,
# #         "Mean EoD": baseline_eod,
# #     })

# # for usage in USAGES:
# #     if usage in usage_metrics and usage_metrics[usage]:
# #         metrics = usage_metrics[usage]
# #         overall_auc = np.mean([v["AUC"] for v in metrics.values()])
# #         mean_eod = np.mean([v["EoD"] for v in metrics.values()])
# #         best_epoch = BEST_EPOCHS[usage]
# #         param_epochs = (PARAM_COUNTS[usage] * best_epoch) / 1e6   # in millions
# #         summary_rows.append({
# #             "Usage": usage,
# #             "Best LR": BEST_LRS[usage],
# #             "Param-Epochs (M)": f"{param_epochs:.1f}",
# #             "Overall Test AUC": f"{overall_auc:.4f}",
# #             "Mean EoD": f"{mean_eod:.4f}",
# #         })

# # df_summary = pd.DataFrame(summary_rows)
# # df_summary.to_csv(os.path.join(OUT_DIR, "summary_table.csv"), index=False)

# # # ------------------------------------------------------------------
# # # 3. Build the Radar Data CSV
# # #    We'll produce a table with columns: Method, Attribute, AUC (or EoD)
# # #    Choose three methods: clip‑zs, full‑ft‑zs, lora‑zs (template)
# # # ------------------------------------------------------------------
# # radar_rows = []
# # methods_for_radar = {
# #     "clip‑zs": baseline_metrics,
# #     "full‑ft‑zs": usage_metrics.get("full-ft-zs"),
# #     "lora‑zs": usage_metrics.get("lora-zs"),
# # }
# # for method, metrics in methods_for_radar.items():
# #     if metrics:
# #         for attr in ATTRIBUTES:
# #             if attr in metrics:
# #                 radar_rows.append({
# #                     "Method": method,
# #                     "Attribute": attr,
# #                     "AUC": metrics[attr]["AUC"],
# #                     "EoD": metrics[attr]["EoD"]
# #                 })

# # df_radar = pd.DataFrame(radar_rows)
# # df_radar.to_csv(os.path.join(OUT_DIR, "radar_data.csv"), index=False)

# # # ------------------------------------------------------------------
# # # 4. Build the Scatter Data CSV (FESTA Trade-off)
# # #    Usage, Param-Epochs, Overall Test AUC, Mean EoD
# # # ------------------------------------------------------------------
# # scatter_rows = []
# # for usage in USAGES:
# #     if usage in usage_metrics and usage_metrics[usage]:
# #         metrics = usage_metrics[usage]
# #         overall_auc = np.mean([v["AUC"] for v in metrics.values()])
# #         mean_eod = np.mean([v["EoD"] for v in metrics.values()])
# #         param_epochs = (PARAM_COUNTS[usage] * BEST_EPOCHS[usage]) / 1e6
# #         scatter_rows.append({
# #             "Usage": usage,
# #             "Param-Epochs (M)": param_epochs,
# #             "Test AUC": overall_auc,
# #             "Mean EoD": mean_eod,
# #         })
# # # Add baseline if desired (cost = 0)
# # if baseline_metrics:
# #     baseline_auc = np.mean([v["AUC"] for v in baseline_metrics.values()])
# #     baseline_eod = np.mean([v["EoD"] for v in baseline_metrics.values()])
# #     scatter_rows.append({
# #         "Usage": "clip‑zs",
# #         "Param-Epochs (M)": 0,
# #         "Test AUC": baseline_auc,
# #         "Mean EoD": baseline_eod,
# #     })

# # df_scatter = pd.DataFrame(scatter_rows)
# # df_scatter.to_csv(os.path.join(OUT_DIR, "scatter_data.csv"), index=False)

# # print(f"✅ CSVs saved to {OUT_DIR}/")

# import os, re, glob, json, pickle
# import pandas as pd
# import numpy as np
# import torch
# from sklearn.metrics import roc_auc_score

# SWEEP_DIR = "./lr_sweep"
# TEXT_SWEEP_DIR = "./text_sweep"
# BASELINE_DIR = "./baseline/clip_zs"
# OUT_DIR = "./poster_data"
# os.makedirs(OUT_DIR, exist_ok=True)

# ATTRIBUTES = ["Sex", "Age", "Race", "Language", "Ethnicity", "MaritalStatus"]
# USAGES = ["lp", "clip-adapt", "full-ft", "full-ft-zs", "lora-lp", "lora-zs"]
# BEST_LRS = {
#     "lp": "1e-2", "clip-adapt": "1e-2", "full-ft": "1e-2",
#     "full-ft-zs": "5e-3", "lora-lp": "1e-2", "lora-zs": "1e-2",
# }
# PARAM_COUNTS = {
#     "lp": 1026, "clip-adapt": 1024, "full-ft": 151278339,
#     "full-ft-zs": 151277313, "lora-lp": 1475586, "lora-zs": 1474560,
# }
# BEST_EPOCHS = {
#     "lp": 88, "clip-adapt": 87, "full-ft": 8,
#     "full-ft-zs": 13, "lora-lp": 25, "lora-zs": 28,
# }

# # ------------------------------------------------------------------
# def find_latest_log(folder):
#     logs = glob.glob(os.path.join(folder, "**", "history*.log"), recursive=True)
#     return max(logs, key=os.path.getmtime) if logs else None

# def extract_metrics_for_attribute(log_text, attr):
#     pattern = re.compile(
#         rf"--- Evaluating {attr} ---.*?"
#         r"overall-auc:\s*([\d.]+).*?"
#         r"eod:\s*([\d.]+)",
#         re.DOTALL
#     )
#     m = pattern.search(log_text)
#     return (float(m.group(1)), float(m.group(2))) if m else (None, None)

# def extract_sex_subgroup_aucs(log_text):
#     sex_block = re.search(r"--- Evaluating Sex ---(.*?)(?=--- Evaluating|---{10,}|\Z)", log_text, re.DOTALL)
#     if not sex_block: return None, None
#     block = sex_block.group(0)
#     male = re.search(r"Male AUC:\s*([\d.]+)", block)
#     female = re.search(r"Female AUC:\s*([\d.]+)", block)
#     return (float(male.group(1)) if male else None,
#             float(female.group(1)) if female else None)

# def get_all_attr_metrics(exp_folder):
#     log_file = find_latest_log(exp_folder)
#     if not log_file: return None, None, None
#     with open(log_file) as f:
#         content = f.read()
#     metrics = {}
#     for attr in ATTRIBUTES:
#         auc, eod = extract_metrics_for_attribute(content, attr)
#         if auc is not None:
#             metrics[attr] = {"AUC": auc, "EoD": eod}
#     male_auc, female_auc = extract_sex_subgroup_aucs(content)
#     return metrics, male_auc, female_auc

# # ------------------------------------------------------------------
# baseline_metrics, base_male, base_female = get_all_attr_metrics(BASELINE_DIR)
# usage_metrics, usage_sex = {}, {}
# for usage in USAGES:
#     lr = BEST_LRS[usage]
#     exp_dir = os.path.join(SWEEP_DIR, f"{usage}_lr{lr}_seed0")
#     metrics, male, female = get_all_attr_metrics(exp_dir)
#     usage_metrics[usage] = metrics
#     usage_sex[usage] = (male, female)

# # ------------------------------------------------------------------
# # 1. Summary Table (with Worst EoD Attribute)
# # ------------------------------------------------------------------
# def worst_eod_attr(metrics):
#     if not metrics: return None, None
#     worst = min(metrics.items(), key=lambda x: x[1]["EoD"])
#     return worst[1]["EoD"], worst[0]

# rows = []
# if baseline_metrics:
#     avg_auc = np.mean([v["AUC"] for v in baseline_metrics.values()])
#     avg_eod = np.mean([v["EoD"] for v in baseline_metrics.values()])
#     worst_val, worst_attr = worst_eod_attr(baseline_metrics)
#     rows.append({"Usage":"clip‑zs","Best LR":"–","Conv. Epoch":"–","Params (M)":0,"Param-Epochs (M)":0,
#                   "Test AUC":avg_auc,"Mean EoD":avg_eod,"Worst EoD":worst_val,
#                   "Worst EoD Attr":worst_attr,"Male AUC":base_male,"Female AUC":base_female})
# for usage in USAGES:
#     if usage in usage_metrics and usage_metrics[usage]:
#         m = usage_metrics[usage]
#         avg_auc = np.mean([v["AUC"] for v in m.values()])
#         avg_eod = np.mean([v["EoD"] for v in m.values()])
#         worst_val, worst_attr = worst_eod_attr(m)
#         male, female = usage_sex.get(usage, (None,None))
#         ep = BEST_EPOCHS[usage]
#         param_ep = PARAM_COUNTS[usage] * ep / 1e6
#         params_m = PARAM_COUNTS[usage] / 1e6
#         rows.append({"Usage":usage,"Best LR":BEST_LRS[usage],"Conv. Epoch":ep,
#                      "Params (M)":f"{params_m:.3f}","Param-Epochs (M)":f"{param_ep:.1f}",
#                      "Test AUC":avg_auc,"Mean EoD":avg_eod,"Worst EoD":worst_val,
#                      "Worst EoD Attr":worst_attr,"Male AUC":male,"Female AUC":female})
# df_summary = pd.DataFrame(rows)
# df_summary.to_csv(os.path.join(OUT_DIR, "summary_table.csv"), index=False)

# # ------------------------------------------------------------------
# # 2. Radar Data (AUC & EoD per attribute, for clip‑zs, full‑ft‑zs, lora‑zs)
# # ------------------------------------------------------------------
# radar = []
# for method, met in [("clip‑zs",baseline_metrics),("full‑ft‑zs",usage_metrics.get("full-ft-zs")),
#                     ("lora‑zs",usage_metrics.get("lora-zs"))]:
#     if met:
#         for attr in ATTRIBUTES:
#             if attr in met:
#                 radar.append({"Method":method,"Attribute":attr,"AUC":met[attr]["AUC"],"EoD":met[attr]["EoD"]})
# df_radar = pd.DataFrame(radar)
# df_radar.to_csv(os.path.join(OUT_DIR, "radar_data.csv"), index=False)

# # ------------------------------------------------------------------
# # 3. Scatter Data (all usages + family)
# # ------------------------------------------------------------------
# def family(u):
#     if u=="clip‑zs": return "Frozen"
#     if u=="clip-adapt": return "Text‑Only"
#     if u in ["lp","lora-lp","lora-zs"]: return "Lightweight Visual"
#     return "Fully Unfrozen"

# scat = []
# if baseline_metrics:
#     avg_auc = np.mean([v["AUC"] for v in baseline_metrics.values()])
#     avg_eod = np.mean([v["EoD"] for v in baseline_metrics.values()])
#     scat.append({"Usage":"clip‑zs","Family":family("clip‑zs"),"Param-Epochs (M)":0,
#                   "Test AUC":avg_auc,"Mean EoD":avg_eod})
# for usage in USAGES:
#     if usage in usage_metrics and usage_metrics[usage]:
#         m = usage_metrics[usage]
#         avg_auc = np.mean([v["AUC"] for v in m.values()])
#         avg_eod = np.mean([v["EoD"] for v in m.values()])
#         param_ep = PARAM_COUNTS[usage] * BEST_EPOCHS[usage] / 1e6
#         scat.append({"Usage":usage,"Family":family(usage),"Param-Epochs (M)":param_ep,
#                       "Test AUC":avg_auc,"Mean EoD":avg_eod})
# df_scatter = pd.DataFrame(scat)
# df_scatter.to_csv(os.path.join(OUT_DIR, "scatter_data.csv"), index=False)

# # ------------------------------------------------------------------
# # 4. Subgroup AUCs for focal method (full‑ft‑zs) – all protected attributes
# # ------------------------------------------------------------------
# def subgroup_aucs(pred_pkl, test_csv, attr_col):
#     meta = pd.read_csv(test_csv)
#     with open(pred_pkl,"rb") as f:
#         data = pickle.load(f)
#     probs = torch.softmax(torch.tensor(data["logits"]),1)[:,1].numpy()
#     meta["prob"] = probs
#     meta["label"] = data["label"]
#     aucs = {}
#     for val, grp in meta.groupby(attr_col):
#         if len(grp)>=5:
#             aucs[val] = roc_auc_score(grp["label"], grp["prob"])
#     return aucs

# focal_dir = os.path.join(SWEEP_DIR, "full-ft-zs_lr5e-3_seed0")
# pred_files = glob.glob(os.path.join(focal_dir, "**/full-ft-zs_eval_Sex/predictions.pkl"), recursive=True)
# test_csv = "/home/hnn/Desktop/rhome/aym/FairMedFM/data/Harvard FairVLMed (Harvard-FairVLMed）/test.csv"
# sub_rows = []
# if pred_files:
#     for col in ["gender","race","language","ethnicity","maritalstatus"]:
#         aucs = subgroup_aucs(pred_files[0], test_csv, col)
#         for sub, auc_val in aucs.items():
#             sub_rows.append({"Attribute":col,"Subgroup":str(sub),"AUC":auc_val})
# df_subgroup = pd.DataFrame(sub_rows)
# df_subgroup.to_csv(os.path.join(OUT_DIR, "subgroup_aucs_focus.csv"), index=False)

# # ------------------------------------------------------------------
# # 5. Text‑source comparison (template vs clustered)
# # ------------------------------------------------------------------
# ts_rows = []
# for usage in ["clip-adapt","full-ft-zs","lora-zs"]:
#     # Template
#     if usage in usage_metrics and usage_metrics[usage]:
#         m = usage_metrics[usage]
#         avg_auc = np.mean([v["AUC"] for v in m.values()])
#         avg_eod = np.mean([v["EoD"] for v in m.values()])
#         male, female = usage_sex.get(usage, (None,None))
#         gap = abs(male-female) if male and female else np.nan
#         ts_rows.append({"Usage":usage,"Source":"template","Test AUC":avg_auc,"AUC-gap (Sex)":gap,"Mean EoD":avg_eod})
#     # Clustered
#     exp_dir = os.path.join(TEXT_SWEEP_DIR, f"{usage}_clustered")
#     log_file = find_latest_log(exp_dir)
#     if log_file:
#         with open(log_file) as f:
#             content = f.read()
#         auc_match = re.search(r"overall-auc:\s*([\d.]+)", content)
#         eod_match = re.search(r"eod:\s*([\d.]+)", content)
#         male_match = re.search(r"Male AUC:\s*([\d.]+)", content)
#         female_match = re.search(r"Female AUC:\s*([\d.]+)", content)
#         if auc_match and eod_match:
#             avg_auc = float(auc_match.group(1))
#             avg_eod = float(eod_match.group(1))
#             male = float(male_match.group(1)) if male_match else np.nan
#             female = float(female_match.group(1)) if female_match else np.nan
#             gap = abs(male-female)
#             ts_rows.append({"Usage":usage,"Source":"clustered","Test AUC":avg_auc,"AUC-gap (Sex)":gap,"Mean EoD":avg_eod})
# df_ts = pd.DataFrame(ts_rows)
# df_ts.to_csv(os.path.join(OUT_DIR, "text_source_compare.csv"), index=False)

# # ------------------------------------------------------------------
# # 6. Intersectional (hardcoded from previous run)
# # ------------------------------------------------------------------
# inter = [
#     {"Intersection":"gender × race","Zero‑shot gap":0.1282,"Fine‑tuned gap":0.1366,"Δ":0.0084},
#     {"Intersection":"maritalstatus × race","Zero‑shot gap":0.4576,"Fine‑tuned gap":0.6275,"Δ":0.1699},
#     {"Intersection":"language × race","Zero‑shot gap":0.4584,"Fine‑tuned gap":0.4584,"Δ":0.0000},
#     {"Intersection":"ethnicity × language","Zero‑shot gap":0.3500,"Fine‑tuned gap":0.3651,"Δ":0.0151},
# ]
# pd.DataFrame(inter).to_csv(os.path.join(OUT_DIR, "intersectional_focus.csv"), index=False)

# # ------------------------------------------------------------------
# # 7. Per‑attribute EoD for the three representative methods
# # ------------------------------------------------------------------
# eod_rows = []
# for method, met in [("clip‑zs",baseline_metrics),("full‑ft‑zs",usage_metrics.get("full-ft-zs")),
#                     ("lora‑zs",usage_metrics.get("lora-zs"))]:
#     if met:
#         for attr in ATTRIBUTES:
#             if attr in met:
#                 eod_rows.append({"Method":method,"Attribute":attr,"EoD":met[attr]["EoD"]})
# df_eod = pd.DataFrame(eod_rows)
# df_eod.to_csv(os.path.join(OUT_DIR, "per_attribute_eod_focus.csv"), index=False)

# print("✅ All 7 CSVs saved to ./poster_data/")
# import os, re, glob, pickle
# import pandas as pd
# import numpy as np
# import torch
# from sklearn.metrics import roc_auc_score

# SWEEP_DIR = "./lr_sweep"
# BASELINE_DIR = "./baseline/clip_zs"
# OUT_DIR = "./poster_data"
# os.makedirs(OUT_DIR, exist_ok=True)

# ATTRIBUTES = ["Sex", "Age", "Race", "Language", "Ethnicity", "MaritalStatus"]
# FOCAL_METHODS = ["clip‑zs", "full‑ft‑zs", "lora‑zs"]
# TRACK_USAGES = ["lp", "clip-adapt", "full-ft", "full-ft-zs", "lora-lp", "lora-zs"]
# BEST_LRS = {
#     "lp": "1e-2", "clip-adapt": "1e-2", "full-ft": "1e-2",
#     "full-ft-zs": "5e-3", "lora-lp": "1e-2", "lora-zs": "1e-2",
# }

# # ------------------------------------------------------------------
# def find_training_log(exp_folder):
#     """Find the history log that contains training/validation epochs."""
#     logs = glob.glob(os.path.join(exp_folder, "**", "history*.log"), recursive=True)
#     best_log = None
#     best_count = -1
#     for log in logs:
#         with open(log, 'r') as f:
#             content = f.read()
#         count = len(re.findall(r"epoch \d+: val AUC =", content))
#         if count > best_count:
#             best_count = count
#             best_log = log
#     return best_log if best_log else None

# def find_latest_log(folder):
#     logs = glob.glob(os.path.join(folder, "**", "history*.log"), recursive=True)
#     return max(logs, key=os.path.getmtime) if logs else None

# # ------------------------------------------------------------------
# # 1. PER‑ATTRIBUTE SUBGROUP TABLES (from test set predictions)
# # ------------------------------------------------------------------
# def compute_subgroup_aucs(pred_pkl, test_csv, attr_col):
#     meta = pd.read_csv(test_csv)
#     with open(pred_pkl, "rb") as f:
#         data = pickle.load(f)
#     probs = torch.softmax(torch.tensor(data["logits"]), dim=1)[:, 1].numpy()
#     meta["prob"] = probs
#     meta["label"] = data["label"]
#     aucs = {}
#     for val, grp in meta.groupby(attr_col):
#         if len(grp) >= 5:
#             aucs[val] = roc_auc_score(grp["label"], grp["prob"])
#     return aucs

# test_csv = "/home/hnn/Desktop/rhome/aym/FairMedFM/data/Harvard FairVLMed (Harvard-FairVLMed）/test.csv"

# subgroup_rows = []
# for method in FOCAL_METHODS:
#     if method == "clip‑zs":
#         pred_files = glob.glob(os.path.join(BASELINE_DIR, "**/clip_zs_final_Sex/predictions.pkl"), recursive=True)
#     elif method == "full‑ft‑zs":
#         pred_files = glob.glob(os.path.join(SWEEP_DIR, "full-ft-zs_lr5e-3_seed0", "**/full-ft-zs_eval_Sex/predictions.pkl"), recursive=True)
#     elif method == "lora‑zs":
#         pred_files = glob.glob(os.path.join(SWEEP_DIR, "lora-zs_lr1e-2_seed0", "**/lora-zs_eval_Sex/predictions.pkl"), recursive=True)
#     else:
#         continue

#     if pred_files:
#         pred_pkl = pred_files[0]
#         for attr_col in ["gender", "race", "language", "ethnicity", "maritalstatus"]:
#             aucs = compute_subgroup_aucs(pred_pkl, test_csv, attr_col)
#             for subgroup, auc_val in aucs.items():
#                 subgroup_rows.append({
#                     "Method": method,
#                     "Attribute": attr_col,
#                     "Subgroup": str(subgroup),
#                     "AUC": auc_val
#                 })

# df_subgroup = pd.DataFrame(subgroup_rows)
# df_subgroup.to_csv(os.path.join(OUT_DIR, "per_attribute_subgroup_aucs.csv"), index=False)
# print("✅ Subgroup AUCs saved.")

# # ------------------------------------------------------------------
# # 2. PER‑ATTRIBUTE SUMMARY (Overall AUC & EoD from logs)
# # ------------------------------------------------------------------
# def extract_attr_metrics_from_log(exp_folder):
#     log_file = find_latest_log(exp_folder)
#     if not log_file:
#         return {}
#     with open(log_file) as f:
#         content = f.read()
#     metrics = {}
#     for attr in ATTRIBUTES:
#         m = re.search(rf"--- Evaluating {attr} ---.*?overall-auc:\s*([\d.]+).*?eod:\s*([\d.]+)", content, re.DOTALL)
#         if m:
#             metrics[attr] = {"Overall AUC": float(m.group(1)), "EoD": float(m.group(2))}
#     return metrics

# attr_summary_rows = []
# for method in FOCAL_METHODS:
#     exp_dir = (BASELINE_DIR if method == "clip‑zs" else
#                os.path.join(SWEEP_DIR, "full-ft-zs_lr5e-3_seed0") if method == "full‑ft‑zs" else
#                os.path.join(SWEEP_DIR, "lora-zs_lr1e-2_seed0"))
#     metrics = extract_attr_metrics_from_log(exp_dir)
#     for attr, vals in metrics.items():
#         attr_summary_rows.append({
#             "Method": method,
#             "Attribute": attr,
#             "Overall AUC": vals["Overall AUC"],
#             "EoD": vals["EoD"]
#         })

# df_attr_summary = pd.DataFrame(attr_summary_rows)
# df_attr_summary.to_csv(os.path.join(OUT_DIR, "per_attribute_summary.csv"), index=False)
# print("✅ Per-attribute summary saved.")

# # ------------------------------------------------------------------
# # 3. SEX BIAS TRACKING ON VALIDATION (per epoch male/female AUC)
# # ------------------------------------------------------------------
# tracking_rows = []
# for usage in TRACK_USAGES:
#     lr = BEST_LRS[usage]
#     exp_dir = os.path.join(SWEEP_DIR, f"{usage}_lr{lr}_seed0")
#     log_file = find_training_log(exp_dir)   # <-- use the training log, not the latest
#     if not log_file:
#         continue
#     with open(log_file) as f:
#         content = f.read()

#     # Match each epoch with its subgroup aucs
#     matches = re.findall(r"epoch (\d+): val AUC = ([\d.]+).*?subgroup metrics: auc: \[([^\]]+)\]", content, re.DOTALL)
#     for match in matches:
#         epoch = int(match[0])
#         val_auc = float(match[1])
#         aucs = [float(x.strip()) for x in match[2].split(",")]
#         if len(aucs) >= 2:
#             male_auc, female_auc = aucs[0], aucs[1]
#         else:
#             male_auc = female_auc = np.nan
#         tracking_rows.append({
#             "Usage": usage,
#             "Epoch": epoch,
#             "Val AUC": val_auc,
#             "Male AUC": male_auc,
#             "Female AUC": female_auc
#         })

# df_tracking = pd.DataFrame(tracking_rows)
# df_tracking.to_csv(os.path.join(OUT_DIR, "sex_bias_tracking_val.csv"), index=False)
# print(f"✅ Sex bias tracking saved – {len(tracking_rows)} rows.")

import os, re, glob, pickle
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

SWEEP_DIR = "./lr_sweep"
BASELINE_DIR = "./baseline/clip_zs"
OUT_DIR = "/home/aym/Desktop/rhome/aym/FairMedFM/poster_data"  # <-- updated path
os.makedirs(OUT_DIR, exist_ok=True)

ATTRIBUTES = ["Sex", "Age", "Race", "Language", "Ethnicity", "MaritalStatus"]
FOCAL_METHODS = ["clip‑zs", "full‑ft‑zs", "lora‑zs"]
TRACK_USAGES = ["lp", "clip-adapt", "full-ft", "full-ft-zs", "lora-lp", "lora-zs"]
BEST_LRS = {
    "lp": "1e-2", "clip-adapt": "1e-2", "full-ft": "1e-2",
    "full-ft-zs": "5e-3", "lora-lp": "1e-2", "lora-zs": "1e-2",
}

# ------------------------------------------------------------------
def find_training_log(exp_folder):
    """Find the history log that contains training/validation epochs."""
    logs = glob.glob(os.path.join(exp_folder, "**", "history*.log"), recursive=True)
    best_log = None
    best_count = -1
    for log in logs:
        with open(log, 'r') as f:
            content = f.read()
        count = len(re.findall(r"epoch \d+: val AUC =", content))
        if count > best_count:
            best_count = count
            best_log = log
    return best_log if best_log else None

def find_latest_log(folder):
    logs = glob.glob(os.path.join(folder, "**", "history*.log"), recursive=True)
    return max(logs, key=os.path.getmtime) if logs else None

# ------------------------------------------------------------------
# 1. PER‑ATTRIBUTE SUBGROUP TABLES (from test set predictions)
# ------------------------------------------------------------------
def compute_subgroup_aucs(pred_pkl, test_csv, attr_col):
    meta = pd.read_csv(test_csv)
    with open(pred_pkl, "rb") as f:
        data = pickle.load(f)
    probs = torch.softmax(torch.tensor(data["logits"]), dim=1)[:, 1].numpy()
    meta["prob"] = probs
    meta["label"] = data["label"]
    aucs = {}
    for val, grp in meta.groupby(attr_col):
        if len(grp) >= 5:
            aucs[val] = roc_auc_score(grp["label"], grp["prob"])
    return aucs

test_csv = "/home/hnn/Desktop/rhome/aym/FairMedFM/data/Harvard FairVLMed (Harvard-FairVLMed）/test.csv"

subgroup_rows = []
for method in FOCAL_METHODS:
    if method == "clip‑zs":
        pred_files = glob.glob(os.path.join(BASELINE_DIR, "**/clip_zs_final_Sex/predictions.pkl"), recursive=True)
    elif method == "full‑ft‑zs":
        pred_files = glob.glob(os.path.join(SWEEP_DIR, "full-ft-zs_lr5e-3_seed0", "**/full-ft-zs_eval_Sex/predictions.pkl"), recursive=True)
    elif method == "lora‑zs":
        pred_files = glob.glob(os.path.join(SWEEP_DIR, "lora-zs_lr1e-2_seed0", "**/lora-zs_eval_Sex/predictions.pkl"), recursive=True)
    else:
        continue

    if pred_files:
        pred_pkl = pred_files[0]
        # Added "age_group" to the list
        for attr_col in ["gender", "age_group", "race", "language", "ethnicity", "maritalstatus"]:
            aucs = compute_subgroup_aucs(pred_pkl, test_csv, attr_col)
            for subgroup, auc_val in aucs.items():
                subgroup_rows.append({
                    "Method": method,
                    "Attribute": attr_col,
                    "Subgroup": str(subgroup),
                    "AUC": auc_val
                })

df_subgroup = pd.DataFrame(subgroup_rows)
df_subgroup.to_csv(os.path.join(OUT_DIR, "per_attribute_subgroup_aucs.csv"), index=False)
print("✅ Subgroup AUCs saved.")

# ------------------------------------------------------------------
# 2. PER‑ATTRIBUTE SUMMARY (Overall AUC & EoD from logs) – unchanged
# ------------------------------------------------------------------
def extract_attr_metrics_from_log(exp_folder):
    log_file = find_latest_log(exp_folder)
    if not log_file:
        return {}
    with open(log_file) as f:
        content = f.read()
    metrics = {}
    for attr in ATTRIBUTES:
        m = re.search(rf"--- Evaluating {attr} ---.*?overall-auc:\s*([\d.]+).*?eod:\s*([\d.]+)", content, re.DOTALL)
        if m:
            metrics[attr] = {"Overall AUC": float(m.group(1)), "EoD": float(m.group(2))}
    return metrics

attr_summary_rows = []
for method in FOCAL_METHODS:
    exp_dir = (BASELINE_DIR if method == "clip‑zs" else
               os.path.join(SWEEP_DIR, "full-ft-zs_lr5e-3_seed0") if method == "full‑ft‑zs" else
               os.path.join(SWEEP_DIR, "lora-zs_lr1e-2_seed0"))
    metrics = extract_attr_metrics_from_log(exp_dir)
    for attr, vals in metrics.items():
        attr_summary_rows.append({
            "Method": method,
            "Attribute": attr,
            "Overall AUC": vals["Overall AUC"],
            "EoD": vals["EoD"]
        })

df_attr_summary = pd.DataFrame(attr_summary_rows)
df_attr_summary.to_csv(os.path.join(OUT_DIR, "per_attribute_summary.csv"), index=False)
print("✅ Per-attribute summary saved.")

# ------------------------------------------------------------------
# 3. SEX BIAS TRACKING ON VALIDATION (per epoch male/female AUC) – unchanged
# ------------------------------------------------------------------
tracking_rows = []
for usage in TRACK_USAGES:
    lr = BEST_LRS[usage]
    exp_dir = os.path.join(SWEEP_DIR, f"{usage}_lr{lr}_seed0")
    log_file = find_training_log(exp_dir)
    if not log_file:
        continue
    with open(log_file) as f:
        content = f.read()

    matches = re.findall(r"epoch (\d+): val AUC = ([\d.]+).*?subgroup metrics: auc: \[([^\]]+)\]", content, re.DOTALL)
    for match in matches:
        epoch = int(match[0])
        val_auc = float(match[1])
        aucs = [float(x.strip()) for x in match[2].split(",")]
        if len(aucs) >= 2:
            male_auc, female_auc = aucs[0], aucs[1]
        else:
            male_auc = female_auc = np.nan
        tracking_rows.append({
            "Usage": usage,
            "Epoch": epoch,
            "Val AUC": val_auc,
            "Male AUC": male_auc,
            "Female AUC": female_auc
        })

df_tracking = pd.DataFrame(tracking_rows)
df_tracking.to_csv(os.path.join(OUT_DIR, "sex_bias_tracking_val.csv"), index=False)
print(f"✅ Sex bias tracking saved – {len(tracking_rows)} rows.")