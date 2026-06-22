import os
import joblib
import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt 
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages

START_ANGLE1 = 90
START_ANGLE2 = 1

PIE_RADIUS = 0.6
INSIDE_LABEL_THRESHOLD_PERCENT = 10.0
INSIDE_LABEL_RADIUS_SCALE = 0.7

BIG_FONT_SIZE = 32
SMALL_FONT_SIZE = 26

FIGSIZE = (12, 10)

legend_properties = {'weight':'bold'}
BOX_TEXTS_COLOR = 'midnightblue'


font_path = '/is/cluster/eduran2/fonts/times.ttf'
font_prop = fm.FontProperties(fname=font_path)


def center_multiline(text: str, width: int) -> str:
    return '\n'.join(line.center(width) for line in text.splitlines())


def add_inside_labels_for_large_slices(ax, wedges, values, names, threshold_percent=INSIDE_LABEL_THRESHOLD_PERCENT):
    """
    Add dataset info inside wedges whose percentage exceeds threshold_percent.
    """
    total_sequences = sum(values)
    if total_sequences == 0:
        return

    for wedge, name, value in zip(wedges, names, values):
        percentage = (value / total_sequences) * 100
        if percentage <= threshold_percent:
            continue

        angle = (wedge.theta1 + wedge.theta2) / 2
        label_x = PIE_RADIUS * INSIDE_LABEL_RADIUS_SCALE * np.cos(np.deg2rad(angle))
        label_y = PIE_RADIUS * INSIDE_LABEL_RADIUS_SCALE * np.sin(np.deg2rad(angle))

        label_text = f'{name}\n{value:,}\n({percentage:.1f}%)'

        ax.text(
            label_x,
            label_y,
            label_text,
            ha='center',
            va='center',
            fontsize=SMALL_FONT_SIZE - 1,
            fontweight='bold',
            color=BOX_TEXTS_COLOR,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=wedge.get_facecolor(), edgecolor='gray', alpha=0.9)
        )


def plot_side_by_side_callouts(
    values_left, names_left, colors_left, xy_list_left, rad_left,
    values_right, names_right, colors_right, xy_list_right, rad_right,
    save_path='side_by_side_callouts.pdf',
    title_left="Body Dataset", title_right="Hand Dataset",
):
    """
    Render two callout pie charts side by side on one PDF page.
    """
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(FIGSIZE[0] * 2, FIGSIZE[1]),
        gridspec_kw={'wspace': -0.2},
    )

    for ax, values, names, colors, xy_list, rad, title in [
        (ax_left, values_left, names_left, colors_left, xy_list_left, rad_left, title_left),
        (ax_right, values_right, names_right, colors_right, xy_list_right, rad_right, title_right),
    ]:
        total_sequences = sum(values)

        wedges, _ = ax.pie(
            values,
            startangle=START_ANGLE1,
            colors=colors,
            radius=PIE_RADIUS,
            wedgeprops=dict(width=PIE_RADIUS, edgecolor='white', linewidth=2),
        )

        add_inside_labels_for_large_slices(ax, wedges, values, names)

        # Calculate angles
        angles = []
        cumulative = START_ANGLE1
        for value in values:
            delta_angle = (value / total_sequences) * 360
            angles.append(cumulative + delta_angle / 2)
            cumulative += delta_angle

        label_radius_list = np.linspace(PIE_RADIUS + 0.25, PIE_RADIUS + 0.25, len(values))

        for i, (wedge, name, value, angle) in enumerate(zip(wedges, names, values, angles)):
            percentage = (value / total_sequences) * 100
            if percentage > INSIDE_LABEL_THRESHOLD_PERCENT:
                continue

            if xy_list is None:
                x = label_radius_list[i] * np.cos(np.deg2rad(angle + START_ANGLE2))
                y = label_radius_list[i] * np.sin(np.deg2rad(angle + START_ANGLE2))
            else:
                x, y = xy_list[i]

            pie_x = PIE_RADIUS * np.cos(np.deg2rad(angle + START_ANGLE2))
            pie_y = PIE_RADIUS * np.sin(np.deg2rad(angle + START_ANGLE2))

            ha = 'left' if x >= 0 else 'right'
            label_text = f'{name}\n{value:,}\n({percentage:.1f}%)'
            width = max(len(name), len(str(value)), len(str(np.round(percentage, 1)))) + 2
            label_text = center_multiline(label_text, width)

            bbox_props = dict(boxstyle="round, pad=0.3", facecolor=wedge.get_facecolor(), edgecolor='gray', alpha=0.8)
            ax.annotate(
                label_text,
                xy=(pie_x, pie_y),
                xytext=(x, y),
                ha=ha, va='center',
                fontsize=SMALL_FONT_SIZE,
                fontweight='bold',
                color=BOX_TEXTS_COLOR,
                bbox=bbox_props,
                arrowprops=dict(arrowstyle='->', connectionstyle=f'arc3,rad={rad}', color='gray', lw=1),
            )

        ax.axis('equal')
        ax.text(0, 0, f'Total\n{total_sequences:,}',
                ha='center', va='center',
                fontsize=SMALL_FONT_SIZE, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        ax.text(0, -PIE_RADIUS - 0.05, title,
                ha='center', va='top',
                fontsize=BIG_FONT_SIZE, fontweight='bold',
                transform=ax.transData)

    fig.subplots_adjust(wspace=-0.4)
    with PdfPages(save_path) as pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Saved side-by-side callouts PDF: {save_path}")


def plot_clean_pie_with_legend(values, names, colors, save_path='clean_pie_chart.png', title="Dataset Distribution"):
    """
    Plots a clean pie chart with a legend and organized layout.
    Args:
        values (list): Values for each slice
        names (list): Names for each slice
        save_path (str): Path to save the chart
        title (str): Title for the chart
    """
    total_sequences = sum(values)
    print(f"Total number of sequences: {total_sequences}")
    
    # Create figure with better proportions - KEY CHANGE: Add figsize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [3, 2]})
     
    # Create pie chart
    wedges, texts = ax1.pie(
        values,
        startangle=START_ANGLE1,
        colors=colors,
        radius=PIE_RADIUS,
        wedgeprops=dict(width=PIE_RADIUS, edgecolor='white', linewidth=2)
    )

    add_inside_labels_for_large_slices(ax1, wedges, values, names)
    
    # Create legend with percentages
    legend_labels = []
    for name, value in zip(names, values):
        percentage = (value / total_sequences) * 100
        legend_labels.append(f'{name}: {value:,} ({percentage:.1f}%)')
    
    # Adjust legend positioning and spacing - BETTER POSITIONING
    legend = ax2.legend(wedges, legend_labels, loc='center', fontsize=SMALL_FONT_SIZE,
                       prop=legend_properties, bbox_to_anchor=(0.1, 0.5))
    legend.set_frame_on(False)
    ax2.axis('off')
    
    # Move total count to bottom
    fig.suptitle(f'Total Sequences: {total_sequences:,}', fontsize=SMALL_FONT_SIZE,
                fontweight='normal', y=0.02)
    
    # Much tighter layout - BETTER SPACING
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
     

def plot_clean_pie_with_callouts(values, names, colors, save_path='clean_pie_chart.png', title="Dataset Distribution", xy_list=None, rad=0.1):
    """
    Plots a pie chart with clean callout labels positioned to avoid overlap.
    
    Args:
        values (list): Values for each slice
        names (list): Names for each slice
        save_path (str): Path to save the chart  
        title (str): Title for the chart
    """
    total_sequences = sum(values)
    print(f"Total number of sequences: {total_sequences}")
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    
    # Create pie chart
    wedges, _ = ax.pie(
        values,
        startangle=START_ANGLE1,
        colors=colors,
        radius=PIE_RADIUS,
        wedgeprops=dict(width=PIE_RADIUS, edgecolor='white', linewidth=2)
    )

    add_inside_labels_for_large_slices(ax, wedges, values, names)
    
    # Calculate angles for each wedge
    angles = []
    cumulative = START_ANGLE1
    

    for value in values:

        delta_angle = (value / total_sequences) * 360
        angle = cumulative + delta_angle/2

        angles.append(angle)

        cumulative += delta_angle
    
    # Position labels to avoid overlap
    label_radius_list = np.linspace(PIE_RADIUS+0.25, PIE_RADIUS+0.25, len(values))
 
    for i, (wedge, name, value, angle) in enumerate(zip(wedges, names, values, angles)):

        percentage = (value / total_sequences) * 100
        if percentage > INSIDE_LABEL_THRESHOLD_PERCENT:
            continue

        # Calculate label position
        if xy_list is None:
            x = label_radius_list[i] * np.cos(np.deg2rad(angle + START_ANGLE2))
            y = label_radius_list[i] * np.sin(np.deg2rad(angle + START_ANGLE2))
        else:
            x, y = xy_list[i]

        # Calculate connection point on pie edge
        pie_x = PIE_RADIUS * np.cos(np.deg2rad(angle + START_ANGLE2))
        pie_y = PIE_RADIUS * np.sin(np.deg2rad(angle + START_ANGLE2))

        # Determine text alignment
        ha = 'left' if x >= 0 else 'right'

        # Create label with percentage
        label_text = f'{name}\n{value:,}\n({percentage:.1f}%)'

        # Calculate max label width
        width = max(len(name), len(str(value)), len(str(np.round(percentage, 1))))
        width += 2

        label_text = center_multiline(label_text, width)

        bbox_props = dict(boxstyle="round, pad=0.2", facecolor=wedge.get_facecolor(), edgecolor='gray', alpha=0.8)

        # Add text with background box
        ax.annotate(label_text,
                   xy=(pie_x, pie_y),
                   xytext=(x, y),
                   ha=ha,
                   va='center',
                   fontsize=SMALL_FONT_SIZE,
                   fontweight='bold',
                   color=BOX_TEXTS_COLOR,
                   bbox=bbox_props,
                   arrowprops=dict(arrowstyle='->',
                                 connectionstyle=f'arc3,rad={rad}',
                                 color='gray', lw=1))
        
        print(x, y)
    
    ax.axis('equal')
    
    # Add total in center
    ax.text(0, 0, f'Total\n{total_sequences:,}', 
           ha='center', va='center', 
           fontsize=SMALL_FONT_SIZE, fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_table_style_pie(values, names, colors, save_path='clean_pie_chart.png', title="Dataset Distribution"):
    """
    Plots a pie chart with a clean table-style layout for labels.
    """
    total_sequences = sum(values)
    print(f"Total number of sequences: {total_sequences}")
    
    fig = plt.figure(figsize=FIGSIZE)

    
    # Create pie chart subplot
    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=1)
     
    wedges, _ = ax1.pie(
        values,
        startangle=START_ANGLE1,
        colors=colors,
        radius=PIE_RADIUS,
        wedgeprops=dict(edgecolor='white', linewidth=2)
    )

    add_inside_labels_for_large_slices(ax1, wedges, values, names)
    
    ax1.set_title(title, fontsize=SMALL_FONT_SIZE, fontweight='bold')
    
    # Create table subplot  
    ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=2)
    ax2.axis('off')
    
    # Prepare table data
    table_data = []
    for i, (name, value) in enumerate(zip(names, values)):
        percentage = (value / total_sequences) * 100
        table_data.append([name, f'{value:,}', f'{percentage:.1f}%'])
    
    # Create table
    table = ax2.table(cellText=table_data,
                     colLabels=['Dataset', 'Sequences', 'Percentage'],
                     cellLoc='left',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color table rows to match pie chart
    for i, color in enumerate(colors):
        table[(i+1, 0)].set_facecolor(color)
        table[(i+1, 1)].set_facecolor(color)
        table[(i+1, 2)].set_facecolor(color)
    
    # Style header
    for j in range(3):
        table[(0, j)].set_facecolor('#40466e')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    # Add total row
    total_row = len(names) + 1
    table.add_cell(total_row, 0, width=1/3, height=1/(len(names)+2), text='TOTAL', loc='left')
    table.add_cell(total_row, 1, width=1/3, height=1/(len(names)+2), text=f'{total_sequences:,}', loc='left')
    table.add_cell(total_row, 2, width=1/3, height=1/(len(names)+2), text='100.0%', loc='left')
    
    for j in range(3):
        table[(total_row, j)].set_facecolor('#d4d4d4')
        table[(total_row, j)].set_text_props(weight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    

# Your existing data loading code
data_cfg = OmegaConf.load('configs/data.yaml')

hand_labels, hand_sizes = [], []
body_labels, body_sizes = [], []

hand_name_dict = {'interhands': 'InterHand2.6M',
                  'reinterhands': 'Re:InterHand',
                  'grab': 'GRAB',
                  'arctic': 'ARCTIC',
                  'moyo': 'MOYO',
                  'samp': 'SAMP',
                  'embody3d': 'Embody3D',
                  'hot3d': 'HOT3D'}

# for hand_path in data_cfg.hand_datapath_list:
#     hand_dname = hand_path.split('/')[-1].split('_')[0]
#     hand_dname = hand_name_dict[hand_dname]     
#     hand_data = joblib.load(hand_path)
#     hand_labels.append(hand_dname)
#     hand_sizes.append(len(hand_data))

# sorted_pairs = sorted(zip(hand_sizes, hand_labels), reverse=False)
# hand_sizes, hand_labels = zip(*sorted_pairs)

hand_sizes = (1641, 4934, 5991, 6256, 7868, 9270)
hand_labels = ('InterHand', 'ARCTIC', 'Re:InterHand', 'MOYO', 'GRAB', 'HOT3D')


for body_path in data_cfg.body_train_datapath_list:
    parts = body_path.split('/')[-1].split('_')
    if len(parts) >= 2 and parts[1] in ('preds'):
        body_dname = f"{parts[0]}_{parts[1]}".upper()
    else:
        body_dname = parts[0].upper()
    body_data = joblib.load(body_path)
    body_labels.append(body_dname)
    body_sizes.append(len(body_data))

sorted_pairs = sorted(zip(body_sizes, body_labels), reverse=False)
body_sizes, body_labels = zip(*sorted_pairs)

# body_sizes = (768, 2243, 3426, 5127, 6932, 16276, 31762, 77610, 90758, 440322)
# body_labels = ("MAMMA", 'SAMP', 'ARCTIC', "MAMMA_PRED", 'GRAB', 'OMOMO', 'INTERX', 'AMASS', 'BEAT2', 'EMBODY3D')
 
os.makedirs('fusion_runs/renders/data_charts', exist_ok=True)

body_xy_list = [
                [0.2, 0.7],
                [-0.13, 0.7],
                [-0.5, 0.5],
                [-0.6, 0.2],
                [-0.9, 0.4],
                [-0.9, 0.1],
                [0.7, -0.2]]
hand_xy_list = [
                [-0.35, 0.65],
                [-0.6, 0.6],
                [-0.9, 0.3],
                [-0.7, -0.5],
                [0.6, -0.5],
                [0.7, 0.3],
                [0.7, 0.1]]


# Define colors for better visibility
colors = plt.cm.Set3(np.linspace(1, 0, len(body_xy_list)))

# Generate different style charts - choose the one you prefer. Make sure largest for hands and body datasets have the same color.

print("Generating hand dataset charts...")
hand_legend_pdf = 'fusion_runs/renders/data_charts/hand_pie_legend.pdf'
hand_table_pdf = 'fusion_runs/renders/data_charts/hand_pie_table.pdf'
hand_callouts_pdf = 'fusion_runs/renders/data_charts/hand_pie_callouts.pdf'

plot_clean_pie_with_legend(hand_sizes, hand_labels, colors[-len(hand_sizes):],
                          hand_legend_pdf,
                          'Hand Dataset Distribution')

plot_table_style_pie(hand_sizes, hand_labels, colors[-len(hand_sizes):],
                     hand_table_pdf,
                     'Hand Dataset Distribution')

plot_clean_pie_with_callouts(hand_sizes, hand_labels, colors[-len(hand_sizes):],
                            hand_callouts_pdf,
                            'Hand Dataset Distribution',
                            xy_list=hand_xy_list,
                            rad=0.1)

print("Generating body dataset charts...")
body_legend_pdf = 'fusion_runs/renders/data_charts/body_pie_legend.pdf'
body_table_pdf = 'fusion_runs/renders/data_charts/body_pie_table.pdf'
body_callouts_pdf = 'fusion_runs/renders/data_charts/body_pie_callouts.pdf'

plot_clean_pie_with_legend(body_sizes, body_labels, colors[:len(body_sizes)],
                          body_legend_pdf,
                          'Body Dataset Distribution')

plot_table_style_pie(body_sizes, body_labels, colors[:len(body_sizes)],
                     body_table_pdf,
                     'Body Dataset Distribution')

plot_clean_pie_with_callouts(body_sizes, body_labels, colors[:len(body_sizes)],
                            body_callouts_pdf,
                            'Body Dataset Distribution',
                            xy_list=body_xy_list,
                            rad=-0.1)


print("Generating side-by-side callouts PDF...")
plot_side_by_side_callouts(
    values_left=body_sizes, names_left=body_labels, colors_left=colors[:len(body_sizes)],
    xy_list_left=body_xy_list, rad_left=-0.1,
    values_right=hand_sizes, names_right=hand_labels, colors_right=colors[-len(hand_sizes):],
    xy_list_right=hand_xy_list, rad_right=0.1,
    save_path='fusion_runs/renders/data_charts/hand_body_callouts.pdf',
    title_left="Body Dataset Distribution", title_right="Hand Dataset Distribution",
)