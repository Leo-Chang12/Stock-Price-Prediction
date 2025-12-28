"""
Create a combined figure showing all 7 figures in the correct order.
Note: Figures 4 and 5 are swapped from their filename numbers.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

# Set up the figure
fig = plt.figure(figsize=(24, 32))
gs = GridSpec(7, 2, figure=fig, hspace=0.3, wspace=0.2)

# Figure order (corrected):
# 1. Model Performance Comparison (full width)
# 2. AAPL Price Predictions (left)
# 3. TSLA Price Predictions (right)
# 4. Statistical Significance (was Fig 5) (full width)
# 5. MSFT Price Predictions (was Fig 4) (full width)
# 6. Permutation Feature Importance (full width)
# 7. Directional Accuracy (full width)

figures_info = [
    ('Figure1_Model_Performance_Comparison.png', 0, slice(None), '1'),
    ('Figure2_AAPL_Price_Predictions.png', 1, 0, '2'),
    ('Figure3_TSLA_Price_Predictions.png', 1, 1, '3'),
    ('Figure5_Statistical_Significance_Analysis.png', 2, slice(None), '4'),  # Was Fig 5, now Fig 4
    ('Figure4_MSFT_Price_Predictions.png', 3, slice(None), '5'),  # Was Fig 4, now Fig 5
    ('Figure6_Permutation_Feature_Importance.png', 4, slice(None), '6'),
    ('Figure7_Directional_Accuracy_Comparison.png', 5, slice(None), '7'),
]

for img_path, row, col, fig_num in figures_info:
    try:
        # Read the image
        img = mpimg.imread(img_path)

        # Create subplot
        if col == slice(None):
            # Full width
            ax = fig.add_subplot(gs[row, :])
        else:
            # Single column
            ax = fig.add_subplot(gs[row, col])

        # Display image
        ax.imshow(img)
        ax.axis('off')

        print(f"[OK] Added Figure {fig_num}: {img_path}")

    except FileNotFoundError:
        print(f"[ERROR] File not found: {img_path}")
        continue

# Adjust layout and save
plt.tight_layout()
plt.savefig('All_Figures_Combined.png', dpi=300, bbox_inches='tight',
           facecolor='white', edgecolor='none')
plt.close()

print("\n[OK] Saved: All_Figures_Combined.png")
print("\nFigure Order (Corrected):")
print("  Figure 1: Model Performance Comparison")
print("  Figure 2: AAPL Stock Price Predictions")
print("  Figure 3: TSLA Stock Price Predictions")
print("  Figure 4: Statistical Significance Analysis")
print("  Figure 5: MSFT Stock Price Predictions")
print("  Figure 6: Permutation-Based Feature Importance")
print("  Figure 7: Directional Accuracy Comparison")
