with open("/Users/paul/Coding_Projects/Master/Dataprocessing/playground_scripts/generate_plots.py", "r") as f:
    lines = f.readlines()
with open("/Users/paul/Coding_Projects/Master/Dataprocessing/playground_scripts/generate_plots.py", "w") as f:
    for i, line in enumerate(lines):
        if 94 <= i <= 101:
            continue
        f.write(line)
