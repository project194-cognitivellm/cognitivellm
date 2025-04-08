import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 👇 Set your run directory
RUN_DIR = "runs/2025-04-08 00:40:57"
# work on runs after_2025-04-07_10:56:11

# 📄 Load allowed transitions from clean JSON file
with open(os.path.join(RUN_DIR, "agents_info.json"), "r") as f:
    agents_info = json.load(f)

allowed_transitions_dict = {
    agent_name: set(info.get("Allowed Transitions", []))
    for agent_name, info in agents_info.items()
}

# 🧠 Step 1: Aggregate transitions
transition_counts = defaultdict(lambda: defaultdict(int))
agent_set = set()

for game_dir in os.listdir(RUN_DIR):
    game_path = os.path.join(RUN_DIR, game_dir)
    trans_file = os.path.join(game_path, "transition_log.json")
    if os.path.isfile(trans_file):
        with open(trans_file, "r") as f:
            transitions = json.load(f)
        for t in transitions:
            from_agent = t["from"].strip()
            to_agent = t["to"].strip()
            transition_counts[from_agent][to_agent] += 1
            agent_set.update([from_agent, to_agent])

# 📦 Step 2: Build normalized matrix
agents = sorted(agent_set)
idx = {name: i for i, name in enumerate(agents)}
n = len(agents)
matrix = np.full((n, n), -1.0)  # default disallowed

for from_agent in agents:
    allowed_targets = allowed_transitions_dict.get(from_agent, set())
    total_allowed = sum(
        transition_counts[from_agent].get(to_agent, 0)
        for to_agent in agents if to_agent in allowed_targets
    )

    for to_agent in agents:
        i, j = idx[from_agent], idx[to_agent]
        if to_agent in allowed_targets:
            count = transition_counts[from_agent].get(to_agent, 0)
            matrix[i][j] = count / total_allowed if total_allowed > 0 else 0.0

# 🎨 Step 3: Plot heatmap
mask = (matrix == -1)
masked_matrix = np.ma.masked_where(mask, matrix)

plt.figure(figsize=(12, 10))
sns.heatmap(
    masked_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    xticklabels=agents,
    yticklabels=agents,
    cbar=True,
    linewidths=0.5,
    linecolor='gray'
)

plt.title("Normalized Agent Transition Probability Heatmap")
plt.xlabel("To Agent")
plt.ylabel("From Agent")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

# 📁 Save
output_path = os.path.join(RUN_DIR, "normalized_transition_heatmap.png")
plt.savefig(output_path)
plt.close()

print(f"✅ Saved normalized transition heatmap to:\n{output_path}")
