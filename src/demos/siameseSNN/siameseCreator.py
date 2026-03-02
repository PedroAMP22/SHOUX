"""
In this code we propose a siamese SNNs architecture to compare similarity
between audios. We will use some soa metris to evaluate our model.


"""
import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, accuracy_score

from classes import SiameseAudioMNIST, SiameseSNN

# --- CONFIGURACIÓN ---
DATA_ROOT = "AudioMNIST_split"
BACKBONE_PATH = "./src/demos/siameseSNN/models/snn_pop_audio.pth"
SIAMESE_SAVE_PATH = "./src/demos/siameseSNN/models/snn_siamese_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_white_noise(waveform, snr_db=0):
    """Añade ruido blanco Gaussiano para el test de robustez"""
    std = waveform.std()
    if std == 0: return waveform
    noise_std = std / (10**(snr_db/20))
    noise = torch.randn_like(waveform) * noise_std
    return waveform + noise

# --- 5. ENTRENAMIENTO Y EVALUACIÓN COMPARATIVA ---
def contrastive_loss(emb1, emb2, label, margin=1.0):
    dist = F.pairwise_distance(emb1, emb2)
    return torch.mean(label * torch.pow(dist, 2) + (1-label) * torch.pow(torch.clamp(margin-dist, min=0.0), 2))

if __name__ == "__main__":
    # A. Entrenar Siamesa
    model = SiameseSNN(BACKBONE_PATH).to(device)
    optimizer = torch.optim.Adam(model.fc_siamese.parameters(), lr=1e-3)
    train_loader = DataLoader(SiameseAudioMNIST(f"{DATA_ROOT}/train"), batch_size=32, shuffle=True)

    print("Entrenando Red Siamesa...")
    for epoch in range(5):
        model.train()
        total_loss = 0
        for x1, x2, label in tqdm(train_loader):
            x1, x2, label = x1.to(device), x2.to(device), label.to(device)
            optimizer.zero_grad()
            emb1, emb2 = model(x1, x2)
            loss = contrastive_loss(emb1, emb2, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")
    
    torch.save(model.state_dict(), SIAMESE_SAVE_PATH)

    # B. Evaluación en Test (Limpio y con Ruido)
    print("\nEvaluando modelo...")
    model.eval()
    
    # Vamos a evaluar dos veces: una en limpio y otra a 0dB (mucho ruido)
    for mode in ["Limpio", "Ruido_0dB"]:
        all_distances = []
        all_labels = []
        total_spikes_layer = 0
        total_neurons_steps = 0

        test_loader = DataLoader(SiameseAudioMNIST(f"{DATA_ROOT}/test"), batch_size=32, shuffle=False)

        with torch.no_grad():
            for x1, x2, label in tqdm(test_loader, desc=f"Evaluando {mode}"):
                if mode == "Ruido_0dB":
                    x1 = add_white_noise(x1, snr_db=0)
                    x2 = add_white_noise(x2, snr_db=0)
                
                x1, x2 = x1.to(device), x2.to(device)
                
                # IMPORTANTE: Tu modelo debe devolver los spikes para medir energía
                # Si tu forward de SiameseSNN devuelve (emb1, emb2, spk_hid1, spk_hid2)
                emb1, emb2, spk1, spk2 = model.forward_full(x1, x2) 
                
                dist = F.pairwise_distance(emb1, emb2)
                
                all_distances.extend(dist.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                
                # Contar spikes para métrica de energía
                total_spikes_layer += (spk1.sum().item() + spk2.sum().item())
                total_neurons_steps += (spk1.numel() + spk2.numel())

        # --- CÁLCULOS ROC / YOUDEN (Igual que tu código) ---
        all_distances = np.array(all_distances)
        all_labels = np.array(all_labels)
        scores = -all_distances
        fpr, tpr, thresholds = roc_curve(all_labels, scores)
        roc_auc = auc(fpr, tpr)
        optimal_idx = np.argmax(tpr - fpr)
        opt_threshold = -thresholds[optimal_idx]
        predictions = (all_distances <= opt_threshold).astype(int)
        acc = accuracy_score(all_labels, predictions)
        
        # --- MÉTRICA DE ENERGÍA ---
        firing_rate = (total_spikes_layer / total_neurons_steps) * 100

        print(f"\n--- RESULTADOS {mode} ---")
        print(f"Accuracy: {acc*100:.2f}% | AUC: {roc_auc:.4f}")
        print(f"Firing Rate (Actividad): {firing_rate:.2f}%")
        # El ahorro energético es: (1 - (Firing_Rate_SNN / Actividad_DNN_aprox_50%))
        print(f"Umbral óptimo: {opt_threshold:.4f}")