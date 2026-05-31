import matplotlib.pyplot as plt
import numpy as np
import os

# Data
models = ['Entropy', 'Hybrid LR', 'Hybrid SVM']
auroc_before = [0.970, 0.976, 0.978]
auroc_after = [0.971, 0.969, 0.981]

x = np.arange(len(models))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - width/2, auroc_before, width, label='Before Features', color='#1f77b4')
rects2 = ax.bar(x + width/2, auroc_after, width, label='After Features', color='#ff7f0e')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('AUROC Score')
ax.set_title('Impact of Expanded Uncertainty Features on AUROC')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0.95, 0.99) # Set y-limits to zoom in on the differences
ax.legend()

ax.bar_label(rects1, padding=3, fmt='%.3f')
ax.bar_label(rects2, padding=3, fmt='%.3f')

fig.tight_layout()

# Save the plot
output_dir = 'docs/images'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'feature_expansion_impact.png')
plt.savefig(output_path, dpi=300)
print(f"Plot saved successfully to {output_path}")
