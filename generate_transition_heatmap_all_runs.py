import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 👇 MODIFY THIS TO MATCH YOUR RUN DIRECTORY
RUN_DIR = "runs/2025-04-01 13:00:00"

# 🧠 Step 1: Aggregate all transitions
transition_counts = defaultdict(lambda: defaultdict(int))
agent_set = set()

for game_dir in os.listdir(RUN_DIR):
    game_path = os.path.join(RUN_DIR, game_dir)
    trans_file = os.path.join(game_path, "transition_log.json")
    if os.path.isfile(trans_file):
        with open(trans_file, "r") as f:
            transitions = json.load(f)
        for t in transitions:
            from_agent = t["from"]
            to_agent = t["to"]
            transition_counts[from_agent][to_agent] += 1
            agent_set.update([from_agent, to_agent])

# 📦 Step 2: Build transition matrix
agents = sorted(agent_set)
idx = {name: i for i, name in enumerate(agents)}
n = len(agents)
matrix = np.zeros((n, n), dtype=int)

for from_agent, targets in transition_counts.items():
    for to_agent, count in targets.items():
        i, j = idx[from_agent], idx[to_agent]
        matrix[i][j] = count

# 🎨 Step 3: Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(matrix, annot=True, fmt="d", cmap="YlOrRd", xticklabels=agents, yticklabels=agents, cbar=True)
plt.title("Agent Transition Frequency Heatmap (All Games)")
plt.xlabel("To Agent")
plt.ylabel("From Agent")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

# 📁 Step 4: Save heatmap image
output_path = os.path.join(RUN_DIR, "agent_transition_heatmap_all_games.png")
plt.savefig(output_path)
plt.close()

print(f"✅ Saved transition frequency heatmap to:\n{output_path}")
