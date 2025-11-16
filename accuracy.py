# accuracy_graph.py
import json
import matplotlib.pyplot as plt

# Load saved accuracy results
with open("models/results.json", "r") as f:
    results = json.load(f)

# Plot graph
plt.figure(figsize=(12, 6))
plt.bar(results.keys(), results.values())

plt.xticks(rotation=45)
plt.ylabel("Accuracy")
plt.title("Air Quality Models — Accuracy Comparison")

plt.tight_layout()
plt.show()
