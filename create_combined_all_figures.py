"""
Create a combined figure showing all 7 figures in the correct manuscript order.
Each figure is placed in its own row at full width for clarity.
Note: Repo Figure4=MSFT (manuscript Fig 5), Repo Figure5=StatSig (manuscript Fig 4).
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Manuscript order: map manuscript figure number to repo filename
figures = [
    ('Figure 1', 'Figure1_Model_Performance_Comparison.png'),
    ('Figure 2', 'Figure2_AAPL_Price_Predictions.png'),
    ('Figure 3', 'Figure3_TSLA_Price_Predictions.png'),
    ('Figure 4', 'Figure5_Statistical_Significance_Analysis.png'),
    ('Figure 5', 'Figure4_MSFT_Price_Predictions.png'),
    ('Figure 6', 'Figure6_Permutation_Feature_Importance.png'),
    ('Figure 7', 'Figure7_Directional_Accuracy_Comparison.png'),
]

# Load all images and get aspect ratios
images = []
for label, path in figures:
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        continue
    img = mpimg.imread(path)
    h, w = img.shape[:2]
    images.append((label, img, w / h))
    print(f"[OK] Loaded {label}: {path} ({w}x{h})")

# Fixed width, each row height proportional to image aspect ratio
fig_width = 20
total_height = sum(fig_width / ar for _, _, ar in images)
# Add padding between figures
padding = 0.3
total_height += padding * (len(images) - 1)

fig, axes = plt.subplots(len(images), 1, figsize=(fig_width, total_height),
                         gridspec_kw={'height_ratios': [1/ar for _, _, ar in images]})

for ax, (label, img, _) in zip(axes, images):
    ax.imshow(img)
    ax.axis('off')

plt.subplots_adjust(hspace=0.05, left=0, right=1, top=1, bottom=0)
plt.savefig('All_Figures_Combined.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.2)
plt.close()

print("\n[OK] Saved: All_Figures_Combined.png")
print("\nFigure Order (Manuscript):")
for label, path in figures:
    print(f"  {label}: {path}")
