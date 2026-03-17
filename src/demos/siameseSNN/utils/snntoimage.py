import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_backbone_architecture():
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis('off')

    c_conv = '#d1e9ff'     
    c_neuron = '#fff4d1' 
    c_process = '#f0f0f0'   
    c_expert = '#e1f5fe'
    c_mask = '#ffebee'      
    line_c = '#444444'

    def box(x, y, w, h, color, title, sub="", detail="", lw=1.5):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                      linewidth=lw, edgecolor=line_c, facecolor=color)
        ax.add_patch(rect)
        plt.text(x + w/2, y + h*0.7, title, ha='center', fontsize=9, fontweight='bold')
        if sub: plt.text(x + w/2, y + h*0.45, sub, ha='center', fontsize=8)
        if detail: plt.text(x + w/2, y + h*0.2, detail, ha='center', fontsize=7, style='italic')

    def arrow(x, y, dx, dy):
        ax.arrow(x, y, dx, dy, head_width=0.15, head_length=0.2, fc=line_c, ec=line_c)


    plt.text(1, 11, "1D Spiking Convolutional Feature Extraction Block", fontsize=12, fontweight='bold', color='#222222')
    
    box(0.5, 8.5, 2, 1.8, c_conv, "nn.Conv1d", "1 -> 16", "k=80, s=4")
    arrow(2.6, 9.4, 0.5, 0)
    box(3.2, 8.5, 2, 1.8, c_process, "nn.BatchNorm1d", "(16)")
    arrow(5.3, 9.4, 0.5, 0)
    box(5.9, 8.5, 2, 1.8, c_neuron, "snn.Leaky", "LIF Neuron", "beta, threshold")
    arrow(8.0, 9.4, 0.5, 0)
    box(8.6, 8.5, 2, 1.8, c_process, "nn.MaxPool1d", "(4)")
    arrow(10.7, 9.4, 0.5, 0)
    box(11.3, 8.5, 2, 1.8, c_conv, "nn.Conv1d", "16 -> 32", "k=3")
    arrow(13.4, 9.4, 0.5, 0)
    box(14.0, 8.5, 2, 1.8, c_process, "nn.BatchNorm1d", "(32)")
    arrow(16.1, 9.4, 0.5, 0)
    box(16.7, 8.5, 2, 1.8, c_neuron, "snn.Leaky", "LIF Neuron", "output=True")
    arrow(18.8, 9.4, 0.5, 0)
    box(19.4, 8.5, 1.5, 1.8, c_process, "nn.Flatten", "15776 feat")

    ax.plot([20.2, 20.2, 1], [8.5, 7.5, 7.5], color=line_c, linestyle='--')
    arrow(1, 7.5, 0, -0.5)


    plt.text(1, 6.5, "Specialized Spiking Classifier Head (Population Encoding)", fontsize=12, fontweight='bold', color='#222222')

    box(0.5, 4.5, 1.5, 1.8, c_process, "nn.Dropout", "(0.4)")
    arrow(2.1, 5.4, 0.5, 0)
    box(2.7, 4.5, 2, 1.8, c_conv, "nn.Linear", "15776 -> 250")
    arrow(4.8, 5.4, 0.5, 0)

    rect_hidden = patches.Rectangle((5.5, 2.5), 6, 5, linewidth=1.5, edgecolor=line_c, facecolor='#fafafa', linestyle='--')
    ax.add_patch(rect_hidden)
    plt.text(8.5, 7.1, "snn.Leaky Hidden Spiking Neurons\n(250 neurons, self.lif_hid)", ha='center', fontsize=9, fontweight='bold')

    expert_labels = ["Class 0 Experts (0-24)", "Class 1 Experts (25-49)", "...", "Class 9 Experts (225-249)"]
    for i, txt in enumerate(expert_labels):
        y_pos = 6 - (i * 1.1)
        for j in range(8):
            color = '#ffa726' if j < 3 else '#e0e0e0'
            ax.add_patch(patches.Rectangle((6 + j*0.4, y_pos), 0.3, 0.5, facecolor=color, edgecolor=line_c, lw=0.5))
        plt.text(5.4, y_pos + 0.25, txt, ha='right', va='center', fontsize=8)

    arrow(11.6, 5, 0.8, 0)
    box(12.5, 3.5, 3, 3.5, c_mask, "Expert Masking Strategy", "expert_masks [10 x 250]", "Penalizes wrong experts")

    arrow(15.6, 5.25, 1, 0)
    box(16.7, 4.5, 2.8, 1.5, '#e8f5e9', "Classification Output", "Sum of Spikes per Group")

    arrow(14, 3.4, 0, -1)
    plt.text(14, 2, "Loss Function Calculation\n(CrossEntropy / MSE on Spikes)", ha='center', fontsize=9, fontweight='bold', bbox=dict(facecolor='white', edgecolor='red'))

    plt.text(0.5, 0.5, "Key Concept: Population Coding\nEach class is assigned to a fixed sub-population of 25 neurons.\nThe mask ensures that only the correct 'experts' learn the features of their class.", 
             fontsize=10, style='italic', bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.5))

    plt.tight_layout()
    plt.savefig("backbone_popnetaudio_architecture.png", dpi=300, bbox_inches='tight')
    plt.show()

draw_backbone_architecture()