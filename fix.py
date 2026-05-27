import sys
with open("/Users/paul/Coding_Projects/Master/Dataprocessing/playground_scripts/generate_plots.py", "r") as f:
    lines = f.readlines()
new_lines = []
skip = False
for line in lines:
    if "# filter out VP_04 and KinderUni globally for relevant subject-level data" in line:
        skip = True
    if skip and 'df_thresh_ps = df_thresh_ps[' in line:
        skip = False
        continue
    if skip:
        continue
    new_lines.append(line)
with open("/Users/paul/Coding_Projects/Master/Dataprocessing/playground_scripts/generate_plots.py", "w") as f:
    f.writelines(new_lines)
