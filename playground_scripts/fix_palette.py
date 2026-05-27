import sys
with open("/Users/paul/Coding_Projects/Master/Dataprocessing/playground_scripts/generate_plots.py", "r") as f:
    code = f.read()
# Make CLASS_COLORS based on the blend
code = code.replace(
    'CLASS_COLORS = ["#2a7b8c", "#3a7c52", "#c65d3a", "#6b4c8a"]',
    'CLASS_COLORS = ["#2f586e", "#4e7084", "#6d879b", "#8c9fb1"]'
)
with open("/Users/paul/Coding_Projects/Master/Dataprocessing/playground_scripts/generate_plots.py", "w") as f:
    f.write(code)
