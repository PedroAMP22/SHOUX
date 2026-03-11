import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from classes import AudioMNISTEvalDataset, PopNetAudio

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "AudioMNIST_split"
MODEL_PATH = "./src/demos/siameseSNN/models/snn_pop_audio_explainable.pth" 
OUTPUT_FILE = "./src/demos/siameseSNN/results/biological_pathway_results.txt"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def run_biological_test():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: no model founf {MODEL_PATH}")
        return

    # Load the model
    print("Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Restore it from checkpoint
    model = PopNetAudio(num_neurons_hid=250, num_classes=10).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # Experts mapping
    neurons_per_class = checkpoint['neurons_per_class']
    class_experts = {i: list(range(i * neurons_per_class, (i + 1) * neurons_per_class)) for i in range(10)}

    # Preapring test ds
    test_ds = AudioMNISTEvalDataset(f"{DATA_ROOT}/test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Inference and report
    matches = 0
    
    with open(OUTPUT_FILE, "w") as f:
        f.write(f"{'ID':<25} | {'REAL':<4} | {'PRED':<4} | {'Experts Sequence (E0-E9)'}\n")
        f.write("-" * 115 + "\n")

        with torch.no_grad():
            for data, label, filename in tqdm(test_loader):
                data = data.to(device)
                # spk_out: [steps, batch, num_classes]
                # spk_hid: [steps, batch, num_neurons_hid]
                spk_out, spk_hid = model(data) 

                # Inference, total add per class
                pred = spk_out.sum(dim=0).argmax(dim=1).item()
                real = label.item()
                if pred == real: matches += 1
                
                # Total activity in hidden layer
                hid_activity = spk_hid.sum(dim=0).squeeze() 
                
                # Spikes per expert
                expert_activations = []
                for c in range(10):
                    indices = class_experts[c]
                    total_act = hid_activity[indices].sum().item()
                    expert_activations.append(int(total_act))

                counts_str = " ".join([f"E{i}:{c:<4}" for i, c in enumerate(expert_activations)])
                status = " Correct" if pred == real else " Wrong"
                
                f.write(f"{filename[0]:<25} | {real:<4} | {pred:<4} {status} | {counts_str}\n")

        acc = 100 * matches / len(test_ds)
        f.write("-" * 115 + "\n")
        f.write(f"Final Acc: {acc:.2f}%\n")

if __name__ == "__main__":
    run_biological_test()