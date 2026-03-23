import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_final_architecture():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis('off')

    colors = {'blue': '#e3f2fd', 'green': '#e8f5e9', 'yellow': '#fff9c4', 'red': '#ffebee', 'line': '#2c3e50'}
    
    def box(x, y, w, h, color, title, detail="", lw=2):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", 
                                      linewidth=lw, edgecolor=colors['line'], facecolor=color)
        ax.add_patch(rect)
        plt.text(x + w/2, y + h/2 + (0.1 if detail else 0), title, ha='center', va='center', fontsize=13, fontweight='bold')
        if detail:
            plt.text(x + w/2, y + h/2 - 0.3, detail, ha='center', va='center', fontsize=11, style='italic')

    def arrow(x, y, dx, dy):
        ax.arrow(x, y, dx, dy, head_width=0.2, head_length=0.3, fc=colors['line'], ec=colors['line'], length_includes_head=True)

    plt.text(1, 9, "INPUT: Raw Audio Signal", fontsize=14, fontweight='bold', color=colors['line'])
    box(0.5, 7, 3, 1.5, colors['blue'], "1D Feature Extractor", "Conv1d + BN + LIF + Pool")
    arrow(3.7, 7.75, 1, 0)

    box(4.9, 7, 2.5, 1.5, colors['yellow'], "Linear Layer", "15776 -> 250")
    arrow(7.6, 7.75, 1, 0)


    expert_rect = patches.Rectangle((8.8, 3.5), 4.5, 5, linewidth=2, edgecolor=colors['line'], facecolor='#ffffff', linestyle='--')
    ax.add_patch(expert_rect)
    plt.text(11.05, 8.8, "Population Encoding\n(250 LIF Neurons)", ha='center', fontsize=12, fontweight='bold')

    for i in range(4):
        y_pos = 7.5 - (i * 1.2)
        label = f"Class {i} Experts" if i < 3 else "..."
        plt.text(9.1, y_pos, label, fontsize=10, fontweight='bold')
        for j in range(8): 
            color = '#ffa726' if i == 0 else '#cfd8dc'
            circ = patches.Circle((11 + j*0.25, y_pos), 0.08, fc=color, ec=colors['line'], lw=0.5)
            ax.add_patch(circ)

    arrow(13.5, 6, 1, 0)

    box(14.7, 5, 2.8, 2, colors['red'], "Expert Masking", "Zeros out 'wrong'\nexperts during training")

    arrow(16.1, 4.8, 0, -1.3)
    box(14.7, 2, 2.8, 1.3, colors['green'], "Classification", "Spike Count Sum")

    plt.tight_layout()
    plt.savefig("src/demos/siameseSNN/results/imgs/snn.png")
    plt.show()

draw_final_architecture()