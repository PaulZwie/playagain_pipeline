import re

with open("generate_plots.py", "r") as f:
    text = f.read()

# Pattern to rip out the incorrectly placed block
broken_pattern = r"""# filter out VP_04 and KinderUni globally for relevant subject-level data \(for image 5 and onwards\)
df_results = df_results\[~df_results\["subject"\].isin\(\["VP_04", "KinderUni"\]\)\]
df_thresh_ps = df_thresh_ps\[~df_thresh_ps\["subject_id"\].isin\(\["VP_04", "KinderUni"\]\)\]

# extract subject from fold_id
def extract_subject\(fold_id\):
    parts = fold_id\.split\("__"\)
    return parts\[1\] if len\(parts\) > 1 else "unknown"

df_results\["subject"\] = df_results\["fold_id"\].apply\(extract_subject\)
"""
import re
text = re.sub(broken_pattern, "", text)

# Ensure this text is placed directly after all read_csv calls
insert_marker = 'df_thresh_p  = pd.read_csv("results/table_6_11_threshold_pooled.csv")'

if "# extract subject from fold_id" not in text:
    text = text.replace(insert_marker, insert_marker + """

# extract subject from fold_id
def extract_subject(fold_id):
    parts = fold_id.split("__")
    return parts[1] if len(parts) > 1 else "unknown"

df_results["subject"] = df_results["fold_id"].apply(extract_subject)

# filter out VP_04 and KinderUni globally for relevant subject-level data (for image 5 and onwards)
df_results = df_results[~df_results["subject"].isin(["VP_04", "KinderUni"])]
df_thresh_ps = df_thresh_ps[~df_thresh_ps["subject_id"].isin(["VP_04", "KinderUni"])]
""")

with open("generate_plots.py", "w") as f:
    f.write(text)
