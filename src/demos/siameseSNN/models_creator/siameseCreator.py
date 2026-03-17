import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os 
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if directorio_raiz not in sys.path:
    sys.path.append(directorio_raiz)

from classes import SiameseAudioMNIST, SiameseSNN

# Config
DATA_ROOT = "AudioMNIST_split"
BACKBONE_PATH = "src/demos/siameseSNN/models/snn_pop_audio_explainable.pth" 
SIAMESE_SAVE_PATH = "./src/demos/siameseSNN/models/snn_siamese_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def improved_contrastive_loss(emb1, emb2, label, margin=1.5):
    dist = F.pairwise_distance(emb1, emb2)
    # Para label=1 (mismo), forzamos distancia 0 con mucha fuerza (pow 2)
    loss_same = label * torch.pow(dist, 2)
    # Para label=0 (distinto), forzamos a que se alejen más allá del margin
    loss_diff = (1 - label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    return torch.mean(loss_same + loss_diff)

if __name__ == "__main__":
    model = SiameseSNN(BACKBONE_PATH).to(device)

    trainable_params = list(model.fc_siamese.parameters()) + list(model.temporal_filter.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=5e-4) 
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    train_loader = DataLoader(SiameseAudioMNIST(f"{DATA_ROOT}/train"), batch_size=64, shuffle=True) 

    epochs = 25 
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for x1, x2, label in pbar:
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)
            
            optimizer.zero_grad()
            emb1, emb2, _, _ = model.forward(x1, x2)
            
            loss = improved_contrastive_loss(emb1, emb2, label, margin=1.5)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)
        print(f"Mean Loss: {avg_loss:.4f}")
        
        torch.save(model.state_dict(), SIAMESE_SAVE_PATH)

    model.eval()
    all_distances = []
    all_labels = []
    test_loader = DataLoader(SiameseAudioMNIST(f"{DATA_ROOT}/test"), batch_size=32, shuffle=False)

    with torch.no_grad():
        for x1, x2, label in tqdm(test_loader, desc="Testing"):
            x1, x2 = x1.to(device), x2.to(device)
            emb1, emb2, _, _ = model.forward(x1, x2) 
            dist = F.pairwise_distance(emb1, emb2)
            all_distances.extend(dist.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    all_distances = np.array(all_distances)
    all_labels = np.array(all_labels)
    scores = -all_distances 
    fpr, tpr, _ = roc_curve(all_labels, scores)
    roc_auc = auc(fpr, tpr)
    print(f"\nAUC = {roc_auc:.4f}")
