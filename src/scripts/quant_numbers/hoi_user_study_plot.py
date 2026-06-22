import math 
import numpy as np
import matplotlib.pyplot as plt


# Sample data for 3 methods across 2 categories
data = {"Prefers FUSION": np.array([94, 22]),  # Percentages for Feature Preference and Usability Rating
        "No Visible Difference": np.array([4, 34]),
        "Prefers Other": np.array([30, 72])}

# first compute pvals 
def sign_test_pvalue(n_pos, n_neg, alternative="two-sided"):
    n = n_pos + n_neg
    if n == 0:
        return 1.0

    if alternative == "greater":  # A > B
        return sum(math.comb(n, k) * 0.5**n for k in range(n_pos, n+1))

    if alternative == "less":     # A < B
        return sum(math.comb(n, k) * 0.5**n for k in range(0, n_pos+1))

    # two-sided
    k = min(n_pos, n_neg)
    p = 2 * sum(math.comb(n, i) * 0.5**n for i in range(0, k+1))
    return min(p, 1.0)

# Example
FUSION_vs_GT = [data["Prefers FUSION"][1], data["No Visible Difference"][1], data["Prefers Other"][1]]
print(f"FUSION vs GT: {FUSION_vs_GT}")
p_value = sign_test_pvalue(FUSION_vs_GT[0], FUSION_vs_GT[2], alternative='two-sided')
print(f"p-value FUSION vs GT = {p_value}")

FUSION_vs_HMP = [data["Prefers FUSION"][0], data["No Visible Difference"][0], data["Prefers Other"][0]]
p_value = sign_test_pvalue(FUSION_vs_HMP[0], FUSION_vs_HMP[2], alternative='two-sided')
print(f"p-value FUSION vs HMP = {p_value}")


data_total = sum(data[method][0] for method in data)

categories = ["FUSION\nvs HMP", "FUSION\nvs GT"]
methods = ["Prefers FUSION", "No Visible Difference", "Prefers Other"]

colors = ['#cfe2f3e7', '#d9ead3e7', '#9999e7']
colors = ['#cfe2f3e7', '#d9ead3e7', '#FFB6C1']


bar_height = 0.05
BIG_FONT_SIZE = 32
SMALL_FONT_SIZE = 20

# Set up the figure with two rows - increased height to accommodate legend spacing
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 5.5))
 
# Data for stacked bars
feature_data = [data[method][0] for method in methods]
usability_data = [data[method][1] for method in methods]

# Row 1: Feature Preference (stacked to 100%)
left = 0
for i, value in enumerate(feature_data):
    value = round(value/data_total, 2)
    ax1.barh(0, value, left=left, color=colors[i], height=bar_height)
    # Add percentage label in the middle of each segment
    ax1.text(left + value/2, 0, f"{int(100 * value)}%", va='center', ha='center', fontweight='bold', fontsize=BIG_FONT_SIZE)
    left += value

ax1.set_yticks([0])
ax1.set_yticklabels([''])
ax1.set_xticklabels([''])


for label in ax1.get_xticklabels():
    label.set_fontsize(20)

# Rotated ylabel with increased labelpad to prevent collision
ax1.set_ylabel(categories[0], rotation=0, va='center', labelpad=70, fontsize=BIG_FONT_SIZE)
ax1.set_xlim(0, 1)
ax1.set_ylim(-bar_height/2, bar_height/2) 
ax1.grid(axis='x', linestyle='--', alpha=0.7)


# Row 2: Usability Rating (stacked to 100%)
left = 0
for i, value in enumerate(usability_data):
    value = round(value/data_total, 2)
    ax2.barh(0, value, left=left, color=colors[i], height=bar_height)
    # Add percentage label in the middle of each segment
    ax2.text(left + value/2, 0, f"{int(100 * value)}%", va='center', ha='center', fontweight='bold', fontsize=BIG_FONT_SIZE)
    left += value

ax2.set_yticks([0])
ax2.set_yticklabels([''])


ax2.xaxis.label.set_size(SMALL_FONT_SIZE)
 

# Rotated ylabel with increased labelpad to prevent collision
ax2.set_ylabel(categories[1], rotation=0, va='center', labelpad=70, fontsize=BIG_FONT_SIZE)
ax2.set_xlim(0, 1)
ax2.set_ylim(-bar_height/2, bar_height/2) 
ax2.grid(axis='x', linestyle='--', alpha=0.7)

left_lim = 0.15
right_lim = 0.9

# Add a thick line between the two bars
fig.add_artist(plt.Line2D([left_lim, right_lim], [0.45, 0.45], color='black', linewidth=3, transform=fig.transFigure))


# Create a legend
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(methods))]
fig.legend(handles, methods, loc='upper center', ncol=3, frameon=True, fontsize=SMALL_FONT_SIZE, 
           bbox_to_anchor=(0.5, 1))  # Move legend higher up

  
# Adjust subplot spacing to accommodate legend and prevent ylabel collision
plt.subplots_adjust(left=left_lim, right=right_lim, 
                    top=0.85, bottom=0.0, 
                    wspace=0.0, hspace=-0.1)

plt.savefig('fusion_runs/renders/user_study_results.pdf', dpi=300, bbox_inches='tight')