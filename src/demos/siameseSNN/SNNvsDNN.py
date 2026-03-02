# In this code we will compare the proposed SNN vs the DNNs of the soa
# We will use Siamese AST as the DNN model

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, accuracy_score
import torch
from transformers import ASTModel, ASTFeatureExtractor
import torch.nn.functional as F
import numpy as np

# --- CONFIGURACIÓN DEL COMPETIDOR SOTA (DNN) ---
class SiameseAST_SOTA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Cargamos el Transformer de Audio de Google (SOTA)
        self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.feature_extractor = ASTFeatureExtractor()
        
    def forward_one(self, x):
        # El AST espera espectrogramas de tamaño fijo
        # x shape: [batch, 1, time] -> transformar a lo que pida el AST
        outputs = self.ast(x)
        return outputs.last_hidden_state.mean(dim=1) # Global Average Pooling

    def forward(self, x1, x2):
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        return emb1, emb2

# --- MÉTRICAS DE COMPARACIÓN REALISTA ---

def calculate_complexity(model_type="SNN", firing_rate=0.0314):
    """
    Cálculo teórico de energía para el paper (basado en literatura de 2024-2026)
    Coste MAC (DNN) = 4.6 pJ
    Coste SOP (SNN) = 0.9 pJ
    """
    if model_type == "SOTA_AST":
        # Un Transformer tiene aprox 80M de parámetros y billones de MACs
        ops = 1.2e9 # 1.2 Giga Operations (Estimado AST)
        energy = ops * 4.6e-12 # en Joules
    else:
        # Tu SNN: Operaciones proporcionales al Firing Rate
        # Suponiendo una arquitectura similar en tamaño
        ops = 1.2e9 * firing_rate 
        energy = ops * 0.9e-12 # en Joules (Suma de picos es más barata)
        
    return energy * 1000 # Convertir a mJ

# Importa tus clases (Asegúrate de que estén en tu archivo classes.py)
from classes import SiameseAudioMNIST, SiameseSNN, SiameseDNN_Baseline 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FUNCIONES DE UTILIDAD ---

def add_noise(waveform, snr_db=0):
    """Añade ruido blanco para test de robustez (0dB es mucho ruido)"""
    # waveform shape: [batch, 1, time] o similar
    std = waveform.std()
    noise_std = std / (10**(snr_db/20))
    noise = torch.randn_like(waveform) * noise_std
    return waveform + noise

def get_performance_metrics(model, dataloader, device, is_snn=False, snr_db=None):
    model.eval()
    distances = []
    labels = []
    total_spikes = 0
    total_ops = 0 # Para DNN (MACs) o SNN (SOPs)

    with torch.no_grad():
        for x1, x2, label in dataloader:
            if snr_db is not None:
                x1 = add_noise(x1, snr_db)
                x2 = add_noise(x2, snr_db)
            
            x1, x2 = x1.to(device), x2.to(device)
            
            if is_snn:
                # En SNN medimos la actividad (spikes)
                emb1, emb2, spk_count = model.forward_with_spikes(x1, x2) 
                total_spikes += spk_count
            else:
                emb1, emb2 = model(x1, x2)
                # En DNN calculamos MACs teóricos (simplificado para el paper)
                # total_ops += batch_size * num_parameters_of_backbone
            
            dist = F.pairwise_distance(emb1, emb2)
            distances.extend(dist.cpu().numpy())
            labels.extend(label.cpu().numpy())

    distances = np.array(distances)
    labels = np.array(labels)
    
    # ROC y Youden
    fpr, tpr, thresholds = roc_curve(labels, -distances)
    roc_auc = auc(fpr, tpr)
    optimal_idx = np.argmax(tpr - fpr)
    opt_threshold = -thresholds[optimal_idx]
    
    preds = (distances <= opt_threshold).astype(int)
    acc = accuracy_score(labels, preds)
    
    return acc, roc_auc, opt_threshold, total_spikes

# --- FLUJO PRINCIPAL ---

if __name__ == "__main__":
    # 1. Cargar Datos
    test_loader = DataLoader(SiameseAudioMNIST("AudioMNIST_split/test"), batch_size=32, shuffle=False)

    # 2. Cargar Modelos
    # SNN (Tu modelo entrenado)
    snn_siamese = SiameseSNN(backbone_path=None).to(device)
    snn_siamese.load_state_dict(torch.load("./src/demos/siameseSNN/models/snn_siamese_model.pt"))
    
    # DNN (Baseline: Una CNN Siamesa estándar para comparar)
    dnn_siamese = SiameseAST_SOTA().to(device)
    # (Asumimos que has entrenado dnn_siamese previamente para que sea una pelea justa)
    # dnn_siamese.load_state_dict(torch.load("dnn_siamese_model.pt"))

    print("\n" + "="*60)
    print(f"{'MÉTRICA':<20} | {'DNN (Baseline)':<15} | {'SNN (Tuya)':<15}")
    print("-"*60)

    # --- EVALUACIÓN EN LIMPIO ---
    acc_dnn, auc_dnn, _, _ = get_performance_metrics(dnn_siamese, test_loader, device, is_snn=False)
    acc_snn, auc_snn, _, spks = get_performance_metrics(snn_siamese, test_loader, device, is_snn=True)

    print(f"{'Accuracy (Limpio)':<20} | {acc_dnn*100:>13.2f}% | {acc_snn*100:>13.2f}%")
    print(f"{'AUC (ROC)':<20} | {auc_dnn:>14.4f} | {auc_snn:>14.4f}")

    # --- EVALUACIÓN CON RUIDO (Robustez) ---
    acc_dnn_n, _, _, _ = get_performance_metrics(dnn_siamese, test_loader, device, is_snn=False, snr_db=0)
    acc_snn_n, _, _, _ = get_performance_metrics(snn_siamese, test_loader, device, is_snn=True, snr_db=0)

    print(f"{'Accuracy (Ruido 0dB)':<20} | {acc_dnn_n*100:>13.2f}% | {acc_snn_n*100:>13.2f}%")

    # --- MÉTRICAS DE ENERGÍA (Estimadas) ---
    # Una DNN hace aprox 100% de operaciones MAC
    # Tu SNN reportó un Firing Rate de 3.14%
    energy_dnn = 100.0 
    energy_snn = 3.14 * (0.9 / 4.6) * 100 # Ajuste pJ (Suma vs MAC)
    
    print(f"{'Costo Energético %':<20} | {energy_dnn:>14.1f}% | {energy_snn:>14.1f}%")
    print("="*60)
    print("\nInterpretación para el Paper:")
    print(f"La SNN es {(energy_dnn/energy_snn):.1f} veces más eficiente que la DNN.")
    if acc_snn_n > acc_dnn_n:
        print(f"La SNN es {(acc_snn_n - acc_dnn_n)*100:.1f}% más robusta al ruido extremo.")