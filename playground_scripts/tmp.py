import numpy as np
import matplotlib.pyplot as plt

# Create the data matrix
mapping = np.array([
    np.arange(17, 33),
    np.arange(1, 17)
])

# Create the figure
fig, ax = plt.subplots(figsize=(12, 2.5))

# Plot the matrix
cax = ax.matshow(mapping, cmap='Blues', alpha=0.3)

# Add text for each cell
for (i, j), val in np.ndenumerate(mapping):
    ax.text(j, i, f'{val:02d}', ha='center', va='center', fontsize=12, fontweight='bold', color='black')

# Remove axes
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])

# Remove borders
for spine in ax.spines.values():
    spine.set_visible(False)

# Add a title
plt.title('EMG Bracelet Channel Mapping', pad=20, fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/paul/Coding_Projects/Master/Dataprocessing/playground_scripts/channel_mapping.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved mapping visualization to channel_mapping.png")
