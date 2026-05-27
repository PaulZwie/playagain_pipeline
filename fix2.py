import sys
with open("/Users/paul/Coding_Projects/Master/Dataprocessing/playground_scripts/generate_plots.py", "r") as f:
    lines = f.readlines()
new_lines = []
for line in lines:
    new_lines.append(line)
    if 'df_thresh_p  = pd.read_csv("results/table_6_11_threshold_pooled.csv")' in line:
        new_lines.append("""
# extract subject from fold_id
def extract_subject(fold_id):
    parts = fold_id.split("__")
    return parts[1] if len(parts) > 1 else "unknown"
df_results["subject"] = df_results["fold_id"].apply(extract_subject)
# filter out VP_04 and KinderUni globally for relevant subject-level data (for image 5 and onwards)
df_results = df_results[~df_results["subject"].isin(["VP_04", "KinderUni"])]
df_thresh_ps = df_thresh_ps[~df_thresh_ps["subject_id"].isin(["VP_04", "KinderUni"])]
""")
with open("/Users/paul/Coding_Projects/Master/Dataprocessing/playground_scripts/generate_plots.py", "w") as f:
    f.writelines(new_lines)
