import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os 
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

# Asegurar rutas
directorio_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if directorio_raiz not in sys.path:
    sys.path.append(directorio_raiz)

from classes import SiameseAudioMNIST, SiameseSNN

# Config
DATA_ROOT = "AudioMNIST_split"
BACKBONE_PATH = "src/demos/siameseSNN/models/snn_pop_audio_explainable.pth" 
SIAMESE_SAVE_PATH = "./src/demos/siameseSNN/models/snn_siamese_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def improved_contrastive_loss(emb1, emb2, label, margin=1.5):
    dist = F.pairwise_distance(emb1, emb2)
    loss_same = label * torch.pow(dist, 2)
    loss_diff = (1 - label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)
    return torch.mean(loss_same + loss_diff)

def validate(model, loader, device, margin=1.5):
    model.eval()
    val_loss = 0
    all_dists = []
    all_labels = []
    
    with torch.no_grad():
        for x1, x2, label in loader:
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)
            emb1, emb2, _, _ = model.forward(x1, x2)
            
            loss = improved_contrastive_loss(emb1, emb2, label, margin)
            val_loss += loss.item()
            
            dist = F.pairwise_distance(emb1, emb2)
            all_dists.extend(dist.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            
    all_dists = np.array(all_dists)
    all_labels = np.array(all_labels)
    fpr, tpr, _ = roc_curve(all_labels, -all_dists)
    val_auc = auc(fpr, tpr)
    
    return val_loss / len(loader), val_auc

if __name__ == "__main__":
    model = SiameseSNN(BACKBONE_PATH).to(device)

    trainable_params = list(model.fc_siamese.parameters()) + list(model.temporal_filter.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=5e-4, weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    train_loader = DataLoader(SiameseAudioMNIST(f"{DATA_ROOT}/train"), batch_size=64, shuffle=True) 
    val_loader = DataLoader(SiameseAudioMNIST(f"{DATA_ROOT}/val"), batch_size=64, shuffle=False)
    test_loader = DataLoader(SiameseAudioMNIST(f"{DATA_ROOT}/test"), batch_size=32, shuffle=False)

    epochs = 25 
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for x1, x2, label in pbar:
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)
            
            optimizer.zero_grad()
            emb1, emb2, _, _ = model.forward(x1, x2)
            loss = improved_contrastive_loss(emb1, emb2, label, margin=1.5)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)

        avg_val_loss, val_auc = validate(model, val_loader, device)
        
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SIAMESE_SAVE_PATH)
    
    model.load_state_dict(torch.load(SIAMESE_SAVE_PATH))
    model.eval()
    
    test_dists = []
    test_labels = []

    with torch.no_grad():
        for x1, x2, label in tqdm(test_loader, desc="Final Test"):
            x1, x2 = x1.to(device), x2.to(device)
            emb1, emb2, _, _ = model.forward(x1, x2) 
            dist = F.pairwise_distance(emb1, emb2)
            test_dists.extend(dist.cpu().numpy())
            test_labels.extend(label.cpu().numpy())

    test_dists = np.array(test_dists)
    test_labels = np.array(test_labels)
    fpr, tpr, _ = roc_curve(test_labels, -test_dists)
    test_auc = auc(fpr, tpr)
    
    print(f"\nTEST AUC: {test_auc:.4f}")