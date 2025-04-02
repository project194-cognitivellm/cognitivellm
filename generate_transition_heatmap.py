import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Load transitions
with open("path/to/transition_log.json", "r") as f:
    transitions = json.load(f)

# Extract all agent names
all_agents = sorted({t["from"] for t in transitions} | {t["to"] for t in transitions})
index_map = {agent: i for i, agent in enumerate(all_agents)}
n = len(all_agents)
matrix = np.zeros((n, n), dtype=int)

# Count transitions
for t in transitions:
    i = index_map[t["from"]]
    j = index_map[t["to"]]
    matrix[i][j] += 1

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(matrix, annot=True, fmt="d", cmap="YlOrRd", xticklabels=all_agents, yticklabels=all_agents, cbar=True)
plt.title("Agent Transition Frequency Heatmap")
plt.xlabel("To Agent")
plt.ylabel("From Agent")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("agent_transition_frequency_heatmap.png")
plt.show()
