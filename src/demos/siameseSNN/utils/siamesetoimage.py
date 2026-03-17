import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_siamese_architecture():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis('off')

    color_backbone = '#e1f5fe'
    color_conv = '#bbdefb'
    color_pool = '#fff9c4'
    color_mlp = '#f5f5f5'
    color_output = '#e8f5e9'
    color_text = '#333333'
    line_color = '#555555'

    def add_box(x, y, w, h, color, title, subtitle="", subsubtitle=""):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", 
                                      linewidth=1.5, edgecolor=line_color, facecolor=color)
        ax.add_patch(rect)
        plt.text(x + w/2, y + h*0.7, title, ha='center', va='center', fontsize=10, fontweight='bold', color=color_text)
        if subtitle:
            plt.text(x + w/2, y + h*0.4, subtitle, ha='center', va='center', fontsize=9, color=color_text)
        if subsubtitle:
            plt.text(x + w/2, y + h*0.15, subsubtitle, ha='center', va='center', fontsize=8, style='italic', color='#666666')

    def add_arrow(x, y, dx, dy):
        ax.arrow(x, y, dx, dy, head_width=0.15, head_length=0.2, fc=line_color, ec=line_color)

    plt.text(9, 9.5, "Siamese SNN Architecture: From Spikes to Semantic Embedding", 
             ha='center', va='center', fontsize=16, fontweight='bold')

    add_box(0.5, 6.5, 2, 1.5, color_backbone, "Input Audio A", "(Raw 1D Signal)")
    add_box(0.5, 2.5, 2, 1.5, color_backbone, "Input Audio B", "(Raw 1D Signal)")

    add_box(3.5, 6.5, 2.5, 1.5, color_backbone, "PopNetAudio", "Backbone (Frozen)", "250 Expert Neurons")
    add_box(3.5, 2.5, 2.5, 1.5, color_backbone, "PopNetAudio", "Backbone (Frozen)", "250 Expert Neurons")
    
    add_arrow(2.7, 7.25, 0.6, 0)
    add_arrow(2.7, 3.25, 0.6, 0)

    plt.text(6.5, 5, "Shared Weights\nProjection Head", ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
    
    ax.plot([6.2, 6.8, 6.8], [7.25, 7.25, 5.5], color=line_color, linewidth=1.5)
    ax.plot([6.2, 6.8, 6.8], [3.25, 3.25, 4.5], color=line_color, linewidth=1.5)
    add_arrow(6.8, 5, 0.7, 0)

    add_box(7.8, 4, 2.2, 2, color_conv, "Temporal Filter", "nn.Conv1d", "k=25, groups=250")
    
    add_arrow(10.2, 5, 0.5, 0)
    add_box(10.9, 4, 2.2, 2, color_pool, "Temporal Pooling", "AdaptiveAvgPool1d", "16 Time Bins")

    add_arrow(13.3, 5, 0.4, 0)
    add_box(13.9, 4.2, 1.2, 1.6, color_mlp, "Flatten", "4000 features")

    add_arrow(15.3, 5, 0.4, 0)
    mlp_desc = "Linear(4000->1024)\nBatchNorm1d\nLeakyReLU\nDropout(0.2)\nLinear(1024->512)"
    add_box(15.9, 3.5, 2.5, 3, color_mlp, "MLP Head", mlp_desc)

    add_arrow(17.2, 3.3, 0, -0.5)
    add_box(16.1, 1.5, 2.1, 1.2, color_output, "L2 Normalization", "512d Embedding")

    plt.text(9, 1.5, "Contrastive Loss / Triplet Loss", ha='center', va='center', 
             bbox=dict(boxstyle="sawtooth", fc="white", ec="red"), fontweight='bold')
    
    ax.annotate("", xy=(10.5, 1.5), xytext=(16, 2),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='red'))

    notes = (
        "Key Features:\n"
        "• Learned Van Rossum: Conv1d mimics biological spike smoothing.\n"
        "• Temporal Bins: 16 phases capture audio evolution.\n"
        "• High Fidelity: 512d space for lower Factual RMSE.\n"
        "• Shared Backbone: Weights are frozen to preserve Population Coding."
    )
    plt.text(0.5, 0.5, notes, fontsize=10, verticalalignment='bottom', 
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.5))

    plt.tight_layout()
    plt.savefig("siamese_snn_architecture.png", dpi=300, bbox_inches='tight')
    plt.show()

draw_siamese_architecture()