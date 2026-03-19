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

# Configuración
DATA_ROOT = "AudioMNIST_split"
BACKBONE_PATH = "src/demos/siameseSNN/models/snn_pop_audio_explainable.pth" 
SIAMESE_SAVE_PATH = "./src/demos/siameseSNN/models/snn_siamese_model.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# LA MAGIA: HIERARCHICAL CONTRASTIVE LOSS
# ==========================================
def hierarchical_contrastive_loss(emb1, emb2, same_digit, same_speaker, margin_diff=1.5, margin_sim=0.5):
    dist = F.pairwise_distance(emb1, emb2)
    
    # Nivel 1: Mismo dígito, mismo locutor -> Distancia ideal = 0.0
    loss_exact = (same_digit * same_speaker) * torch.pow(dist, 2)
    
    # Nivel 2: Mismo dígito, DISTINTO locutor -> Distancia ideal = margin_sim (ej. 0.5)
    # Queremos que estén en el mismo macro-cluster, pero respetando su diferencia acústica
    loss_similar = (same_digit * (1 - same_speaker)) * torch.pow(dist - margin_sim, 2)
    
    # Nivel 3: Distinto dígito -> Distancia ideal > margin_diff (ej. 1.5)
    loss_diff = (1 - same_digit) * torch.pow(torch.clamp(margin_diff - dist, min=0.0), 2)
    
    return torch.mean(loss_exact + loss_similar + loss_diff)

def validate(model, loader, device):
    model.eval()
    val_loss = 0
    all_dists = []
    all_labels =[] # Guardaremos same_digit para el AUC
    
    with torch.no_grad():
        for x1, x2, same_digit, same_speaker in loader:
            x1, x2 = x1.to(device), x2.to(device)
            same_digit, same_speaker = same_digit.to(device), same_speaker.to(device)
            
            emb1, emb2, _, _ = model.forward(x1, x2)
            loss = hierarchical_contrastive_loss(emb1, emb2, same_digit, same_speaker)
            val_loss += loss.item()
            
            dist = F.pairwise_distance(emb1, emb2)
            all_dists.extend(dist.cpu().numpy())
            all_labels.extend(same_digit.cpu().numpy())
            
    fpr, tpr, _ = roc_curve(all_labels, -np.array(all_dists))
    val_auc = auc(fpr, tpr)
    return val_loss / len(loader), val_auc

if __name__ == "__main__":
    model = SiameseSNN(BACKBONE_PATH).to(device)

    trainable_params = list(model.fc_siamese.parameters()) + list(model.temporal_filter.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    train_loader = DataLoader(SiameseAudioMNIST(f"{DATA_ROOT}/train"), batch_size=64, shuffle=True) 
    val_loader = DataLoader(SiameseAudioMNIST(f"{DATA_ROOT}/val"), batch_size=64, shuffle=False)
    test_loader = DataLoader(SiameseAudioMNIST(f"{DATA_ROOT}/test"), batch_size=32, shuffle=False)

    epochs = 25
    best_val_loss = float('inf')

    print(f"{device}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for x1, x2, same_digit, same_speaker in pbar:
            x1, x2 = x1.to(device), x2.to(device)
            same_digit, same_speaker = same_digit.to(device), same_speaker.to(device)
            
            optimizer.zero_grad()
            emb1, emb2, _, _ = model.forward(x1, x2)
            
            loss = hierarchical_contrastive_loss(emb1, emb2, same_digit, same_speaker)
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
            print(f"Model save (Val Loss: {best_val_loss:.4f})")
    
    # --- TEST FINAL ---
    print("\n" + "="*30)
    print("Eval")
    print("="*30)
    
    model.load_state_dict(torch.load(SIAMESE_SAVE_PATH))
    model.eval()
    
    test_dists =[]
    test_labels = []
    with torch.no_grad():
        for x1, x2, same_digit, same_speaker in tqdm(test_loader, desc="Evaluating Test Set"):
            x1, x2 = x1.to(device), x2.to(device)
            same_digit = same_digit.to(device)
            
            emb1, emb2, _, _ = model.forward(x1, x2)
            dist = F.pairwise_distance(emb1, emb2)
            
            test_dists.extend(dist.cpu().numpy())
            test_labels.extend(same_digit.cpu().numpy())
    
    fpr, tpr, _ = roc_curve(test_labels, -np.array(test_dists))
    test_auc = auc(fpr, tpr)
    print(f"Test AUC: {test_auc:.4f}")
