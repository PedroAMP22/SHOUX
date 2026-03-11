import torch
import torch.nn.functional as F
import numpy as np
import os 
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, accuracy_score

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if directorio_raiz not in sys.path:
    sys.path.append(directorio_raiz)

from classes import SiameseAudioMNIST, SiameseSNN

# Config
DATA_ROOT = "AudioMNIST_split"
BACKBONE_PATH = "src/demos/siameseSNN/models/snn_pop_audio_explainable.pth" 
SIAMESE_SAVE_PATH = "./src/demos/siameseSNN/models/snn_siamese_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_white_noise(waveform, snr_db=0):
    std = waveform.std()
    if std == 0: return waveform
    noise_std = std / (10**(snr_db/20))
    noise = torch.randn_like(waveform) * noise_std
    noisy_waveform = waveform + noise
    return noisy_waveform / (noisy_waveform.abs().max() + 1e-8)

# Train / Val
def contrastive_loss(emb1, emb2, label, margin=1.0):
    dist = F.pairwise_distance(emb1, emb2)
    # 1 same, 0 diff
    return torch.mean(label * torch.pow(dist, 2) + (1-label) * torch.pow(torch.clamp(margin-dist, min=0.0), 2))

if __name__ == "__main__":
    # Load the model (backbone)
    model = SiameseSNN(BACKBONE_PATH).to(device)
    
    # Check if exists
    if os.path.exists(SIAMESE_SAVE_PATH):
        print(f"Model found {SIAMESE_SAVE_PATH}")
        model.load_state_dict(torch.load(SIAMESE_SAVE_PATH, map_location=device))
    else:
        print("Creating new one")
        # Train only last layer
        optimizer = torch.optim.Adam(model.fc_siamese.parameters(), lr=1e-3)
        train_loader = DataLoader(SiameseAudioMNIST(f"{DATA_ROOT}/train"), batch_size=32, shuffle=True)

        for epoch in range(5):
            model.train()
            total_loss = 0
            for x1, x2, label in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                optimizer.zero_grad()
                emb1, emb2, _, _ = model(x1, x2)
                loss = contrastive_loss(emb1, emb2, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Loss: {total_loss/len(train_loader):.4f}")
        
        torch.save(model.state_dict(), SIAMESE_SAVE_PATH)
        print(f"Save {SIAMESE_SAVE_PATH}")

    # --- B. EVALUACIÓN ---
    print("\nEval...")
    model.eval()
    
    
    all_distances = []
    all_labels = []
    total_spikes_acum = 0
    total_posible_spikes_acum = 0

    test_loader = DataLoader(SiameseAudioMNIST(f"{DATA_ROOT}/test"), batch_size=32, shuffle=False)

    with torch.no_grad():
        for x1, x2, label in tqdm(test_loader, desc=f"Evaluating"):
            
            x1, x2 = x1.to(device), x2.to(device)
            
            emb1, emb2, spk1, spk2 = model.forward(x1, x2) 
            
            dist = F.pairwise_distance(emb1, emb2)
            
            all_distances.extend(dist.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            

        # Metics
        all_distances = np.array(all_distances)
        all_labels = np.array(all_labels)
        scores = -all_distances 
        fpr, tpr, thresholds = roc_curve(all_labels, scores)
        roc_auc = auc(fpr, tpr)
        
        optimal_idx = np.argmax(tpr - fpr)
        opt_threshold = -thresholds[optimal_idx]
        predictions = (all_distances <= opt_threshold).astype(int)
        acc = accuracy_score(all_labels, predictions)
        

        print(f"\Results:")
        print(f"Accuracy: {acc*100:.2f}% | AUC: {roc_auc:.4f}")