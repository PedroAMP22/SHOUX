import torch
import torch.nn.functional as F
import numpy as np
import os 

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, accuracy_score

from src.demos.siameseSNN.classes import SiameseAudioMNIST, SiameseSNN

# --- CONFIGURACIÓN ---
DATA_ROOT = "AudioMNIST_split"
BACKBONE_PATH = "src/demos/siameseSNN/models/snn_pop_audio_explainable.pth" 
SIAMESE_SAVE_PATH = "./src/demos/siameseSNN/models/snn_siamese_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_white_noise(waveform, snr_db=0):
    """Añade ruido y re-normaliza para mantener la dinámica de la SNN"""
    std = waveform.std()
    if std == 0: return waveform
    noise_std = std / (10**(snr_db/20))
    noise = torch.randn_like(waveform) * noise_std
    noisy_waveform = waveform + noise
    # Re-normalización crítica para que el backbone no se sature ante el ruido
    return noisy_waveform / (noisy_waveform.abs().max() + 1e-8)

# --- ENTRENAMIENTO Y EVALUACIÓN ---
def contrastive_loss(emb1, emb2, label, margin=1.0):
    dist = F.pairwise_distance(emb1, emb2)
    # label=1 (mismo), label=0 (diferente)
    return torch.mean(label * torch.pow(dist, 2) + (1-label) * torch.pow(torch.clamp(margin-dist, min=0.0), 2))

if __name__ == "__main__":
    # Inicializar modelo (SiameseSNN ahora usa el backbone de 250 expertos)
    model = SiameseSNN(BACKBONE_PATH).to(device)
    
    # --- A. VERIFICAR SI EL MODELO YA EXISTE ---
    if os.path.exists(SIAMESE_SAVE_PATH):
        print(f"Modelo encontrado en {SIAMESE_SAVE_PATH}. Cargando pesos...")
        model.load_state_dict(torch.load(SIAMESE_SAVE_PATH, map_location=device))
    else:
        print("Modelo no encontrado. Iniciando entrenamiento de la Red Siamesa...")
        # Entrenamos solo la capa de proyección siamesa
        optimizer = torch.optim.Adam(model.fc_siamese.parameters(), lr=1e-3)
        train_loader = DataLoader(SiameseAudioMNIST(f"{DATA_ROOT}/train"), batch_size=32, shuffle=True)

        for epoch in range(5):
            model.train()
            total_loss = 0
            for x1, x2, label in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                optimizer.zero_grad()
                emb1, emb2 = model(x1, x2)
                loss = contrastive_loss(emb1, emb2, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Loss: {total_loss/len(train_loader):.4f}")
        
        torch.save(model.state_dict(), SIAMESE_SAVE_PATH)
        print(f"Guardado en {SIAMESE_SAVE_PATH}")

    # --- B. EVALUACIÓN ---
    print("\nIniciando Benchmark...")
    model.eval()
    
    
    all_distances = []
    all_labels = []
    total_spikes_acum = 0
    total_posible_spikes_acum = 0

    test_loader = DataLoader(SiameseAudioMNIST(f"{DATA_ROOT}/test"), batch_size=32, shuffle=False)

    with torch.no_grad():
        for x1, x2, label in tqdm(test_loader, desc=f"Evaluando"):
            
            x1, x2 = x1.to(device), x2.to(device)
            
            # forward_full devuelve los tensores de spikes para medir energía
            emb1, emb2, spk1, spk2 = model.forward_full(x1, x2) 
            
            dist = F.pairwise_distance(emb1, emb2)
            
            all_distances.extend(dist.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            

        # Métricas
        all_distances = np.array(all_distances)
        all_labels = np.array(all_labels)
        scores = -all_distances 
        fpr, tpr, thresholds = roc_curve(all_labels, scores)
        roc_auc = auc(fpr, tpr)
        
        optimal_idx = np.argmax(tpr - fpr)
        opt_threshold = -thresholds[optimal_idx]
        predictions = (all_distances <= opt_threshold).astype(int)
        acc = accuracy_score(all_labels, predictions)
        

        print(f"\nRESULTADOS:")
        print(f"Accuracy: {acc*100:.2f}% | AUC: {roc_auc:.4f}")