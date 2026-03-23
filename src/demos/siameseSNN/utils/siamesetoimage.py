import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_clean_siamese():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.axis('off')

    c = {'back': '#E1F5FE', 'shared': '#FFF9C4', 'head': '#F5F5F5', 'out': '#E8F5E9', 'line': '#37474F'}

    def box(x, y, w, h, color, title, sub="", detail="", lw=2):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", 
                                      linewidth=lw, edgecolor=c['line'], facecolor=color)
        ax.add_patch(rect)
        plt.text(x + w/2, y + h*0.65, title, ha='center', fontsize=12, fontweight='bold')
        if sub: plt.text(x + w/2, y + h*0.4, sub, ha='center', fontsize=10)
        if detail: plt.text(x + w/2, y + h*0.15, detail, ha='center', fontsize=9, style='italic')

    def arrow(x, y, dx, dy, color=c['line']):
        ax.arrow(x, y, dx, dy, head_width=0.2, head_length=0.3, fc=color, ec=color, length_includes_head=True)

    plt.text(10, 9.5, "Siamese SNN: Semantic Audio Embedding", ha='center', fontsize=18, fontweight='bold')

    box(1, 6.5, 3.5, 1.8, c['back'], "Input Audio A", "PopNet Backbone", "(Weights Frozen)")
    box(1, 2.5, 3.5, 1.8, c['back'], "Input Audio B", "PopNet Backbone", "(Weights Frozen)")

    # Conexión a la cabeza compartida (Shared Weights)
    ax.plot([4.7, 5.5, 5.5], [7.4, 7.4, 5.5], color=c['line'], lw=2)
    ax.plot([4.7, 5.5, 5.5], [3.4, 3.4, 4.5], color=c['line'], lw=2)
    arrow(5.5, 5, 1, 0)

    plt.text(8.5, 7.5, "SHARED PROJECTION HEAD", ha='center', fontsize=14, fontweight='bold', color='#1565C0')
    
    box(6.8, 4, 3.5, 2, c['shared'], "Temporal Filter", "Conv1d (k=25)", "Van Rossum Kernel")
    arrow(10.5, 5, 0.8, 0)

    box(11.5, 4, 3.5, 2, c['head'], "Temporal Pooling", "AdaptiveAvgPool1d", "16 Time Bins")
    arrow(15.2, 5, 0.8, 0)

    box(16.2, 3.5, 3.2, 3, c['out'], "MLP Head", "Dense 1024 → 512", "L2 Normalization")

    plt.text(14, 1.5, "LOSS FUNCTION: Contrastive", 
             ha='center', fontsize=12, fontweight='bold', color='red',
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="red"))
    
    ax.annotate("", xy=(17.8, 3.3), xytext=(15.5, 1.8),
                arrowprops=dict(arrowstyle="<-", color='red', connectionstyle="arc3,rad=-0.2", lw=1.5))


    plt.savefig("src/demos/siameseSNN/results/imgs/siamese_snn_architecture.png")

    plt.show()

draw_clean_siamese()