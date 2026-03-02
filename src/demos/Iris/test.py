import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import numpy as np
from scipy.io import wavfile
import os

# --- 1. MODEL DEFINITION & LOADING ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("./src/demos/Iris/snn_pop_model.pth", weights_only=False)

def population_encode(batch, bins=15):
    batch_size, num_features = batch.shape
    encoded = torch.zeros(batch_size, num_features * bins)
    batch = torch.clamp(batch, 0, 1) 
    for i in range(num_features):
        vals = (batch[:, i] * (bins - 1)).long()
        for j, v in enumerate(vals):
            v = min(max(v.item(), 0), bins - 1)
            encoded[j, i * bins + v] = 1.0
    return encoded

class PopNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(60, 64)
        self.lif1 = snn.Leaky(beta=0.8, spike_grad=surrogate.atan())
        self.fc2 = nn.Linear(64, 3)
        self.lif2 = snn.Leaky(beta=0.8, spike_grad=surrogate.atan())

    def forward(self, x):
        x_pop = population_encode(x).to(device)
        mem1, mem2 = self.lif1.init_leaky(), self.lif2.init_leaky()
        spk2_rec, spk1_rec = [], []
        for step in range(25):
            cur1 = self.fc1(x_pop)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk1_rec.append(spk1); spk2_rec.append(spk2)
        return torch.stack(spk2_rec), torch.stack(spk1_rec), x_pop

net = PopNet().to(device)
net.load_state_dict(checkpoint['model_state'])
feature_experts = checkpoint['feature_experts']
X_test, y_test = checkpoint['test_data']
target_names = checkpoint['target_names']

# --- 2. THE IMPROVED SONIFICATION FUNCTION ---
def generate_input_sonification(x_pop_vector, filename):
    """ 
    8s audio sequential (2s per feature):
    - Notes: Do (SL), Re (SW), Mi (PL), Fa (PW)
    - Volume: Proportional to the BIN value (Small feature = Quiet, Large = Loud)
    """
    fs = 44100
    sec_per_feat = 2.0
    bins_per_feat = 15
    total_samples = int(fs * sec_per_feat * 4)
    full_audio = np.zeros(total_samples)
    
    # Notas Musicales Claras (Do4, Re4, Mi4, Fa4)
    freqs = [261.63, 293.66, 329.63, 349.23] 
    samples_per_feat = int(fs * sec_per_feat)

    for f_idx in range(4):
        # Identificar qué Bin está activo (0 a 14)
        feature_slice = x_pop_vector[f_idx * bins_per_feat : (f_idx + 1) * bins_per_feat]
        active_bin = torch.argmax(feature_slice).item()
        
        # --- LÓGICA DE VOLUMEN (PROPORCIONAL AL BIN) ---
        # Bin 0 (mínimo) -> Volumen 0.05 (Casi nada)
        # Bin 14 (máximo) -> Volumen 1.0 (Máximo)
        # Usamos potencia 1.5 para que la diferencia sea visualmente evidente
        intensity = np.power((active_bin + 1) / bins_per_feat, 1.5)
        
        t = np.linspace(0, sec_per_feat, samples_per_feat, endpoint=False)
        
        # Generamos la nota Do, Re, Mi o Fa
        freq = freqs[f_idx]
        
        # Onda senoidal con un toque de armónicos para que suene "limpio" pero con cuerpo
        sine = np.sin(2 * np.pi * freq * t)
        note = sine * np.exp(-3 * t / sec_per_feat) # Decaimiento exponencial (tipo campana)
        
        start = int(f_idx * samples_per_feat)
        # Aplicamos la intensidad basada en el BIN
        full_audio[start : start + samples_per_feat] = note * intensity

    # --- NO NORMALIZAR POR ARCHIVO ---
    # Usamos un factor de escala global para todos los audios.
    # De esta forma, una flor pequeña suena FLOJO y una grande suena FUERTE.
    global_gain = 0.8
    wavfile.write(filename, fs, (full_audio * global_gain * 32767).astype(np.int16))
    
# --- 3. TEST LOOP ---
log_file = "./src/demos/Iris/results/test_results.txt"
print(f"{'ID':<3} | {'REAL':<12} | {'PRED':<12} | {'STATUS'}")
print("-" * 50)

with open(log_file, "w") as f:
    f.write(f"{'ID':<3} | {'REAL':<12} | {'PRED':<12} | {'STATUS':<10} | {'HIDDEN SPIKE COUNTS'}\n")
    f.write("-" * 90 + "\n")

    for i in range(len(X_test)):
        sample = X_test[i:i+1].to(device)
        real_idx = y_test[i].item()
        
        with torch.no_grad():
            spk_out, spk_hid, x_pop = net(sample)
            pred_idx = torch.argmax(spk_out.sum(dim=0)).item()
            
            hidden_spikes = spk_hid.squeeze(1).cpu()
            x_pop_vector = x_pop.squeeze(0).cpu()
            
        status = "MATCH ✅" if (real_idx == pred_idx) else "MISMATCH ❌"
        expert_counts = [int(hidden_spikes[:, feature_experts[j]].sum()) for j in range(4)]
        
        print(f"{i:02} | {target_names[real_idx].upper():12} | {target_names[pred_idx].upper():12} | {status}")
        f.write(f"{i:02} | {target_names[real_idx].upper():12} | {target_names[pred_idx].upper():12} | {status:10} | {expert_counts}\n")
        
        audio_name = f"./src/demos/Iris/results/{i:02}_{target_names[real_idx].upper()}_as_{target_names[pred_idx].upper()}.wav"
        generate_input_sonification(x_pop_vector, audio_name)

print("\nSuccess! Open the files in Audacity. The 'triangles' should now have different heights.")