import torch
import torch.nn.functional as F
import torchaudio
import os
import random
from tqdm import tqdm
from classes import SiameseSNN

# --- CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "AudioMNIST_split/test"
BACKBONE_CHECKPOINT = "./src/demos/siameseSNN/models/snn_pop_audio.pth"
SIAMESE_MODEL_PATH = "./src/demos/siameseSNN/models/snn_siamese_model.pt"
OUTPUT_FILE = "./src/demos/siameseSNN/results/siamese_explainability_report.txt"

# --- 2. UTILS ---

def load_audio(path, sample_rate=8000, num_samples=8000):
    waveform, sr = torchaudio.load(path, backend="soundfile")
    waveform = waveform - waveform.mean()
    waveform = waveform / (waveform.abs().max() + 1e-8)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    if waveform.size(1) > num_samples:
        waveform = waveform[:, :num_samples]
    else:
        waveform = torch.nn.functional.pad(waveform, (0, num_samples - waveform.size(1)))
    return waveform.to(device)

# --- 3. TEST SCRIPT ---

def run_test():
    # A. Cargar el mapa de expertos
    print("Analizando pesos para encontrar Secuencias de Identidad Únicas...")
    checkpoint_bg = torch.load(BACKBONE_CHECKPOINT, map_location=device)
    weights = checkpoint_bg['model_state']['fc2.weight'].cpu()
    
    class_experts = {i: [] for i in range(10)}
    neurons_assigned = 0

    for n in range(256):
        w_neuron = weights[:, n]
        best_class = torch.argmax(w_neuron).item()
        
        sorted_weights, _ = torch.sort(w_neuron, descending=True)
        margin = sorted_weights[0] - sorted_weights[1]
        
        if w_neuron[best_class] > 0.15 and margin > 0.1:
            class_experts[best_class].append(n)
            neurons_assigned += 1

    print(f"Total neuronas puras asignadas: {neurons_assigned} de 256")
    for c in range(10):
        print(f"Clase {c}: {len(class_experts[c])} expertos puros")

    # B. Cargar Modelo Siamese
    model = SiameseSNN(BACKBONE_CHECKPOINT).to(device)
    model.load_state_dict(torch.load(SIAMESE_MODEL_PATH, map_location=device))
    model.eval()

    # C. Preparar archivos
    all_files = []
    for root, dirs, files in os.walk(DATA_ROOT):
        for f in files:
            if f.endswith(".wav"):
                label = int(os.path.basename(root))
                all_files.append((os.path.join(root, f), label, f))

    print(f"Generando reporte de RUTAS NEURONALES en {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, "w") as f:
        f.write("REPORTE DE EXPLICABILIDAD POR AISLAMIENTO DE RUTAS (SNN)\n")
        f.write("-" * 160 + "\n")
        # Cambiamos las cabeceras para ver la clase real y el experto activado de cada uno
        f.write(f"{'FILE A (REAL)':<20} | {'FILE B (REAL)':<20} | {'DIST':<6} | {'RUTA ACTIVADA EN A':<40} | {'RUTA ACTIVADA EN B':<40}\n")
        f.write("-" * 160 + "\n")

        for i in tqdm(range(100)):
            item_a = random.choice(all_files)
            item_b = random.choice(all_files)
            
            audio_a = load_audio(item_a[0])
            audio_b = load_audio(item_b[0])
            
            with torch.no_grad():
                emb_a, act_a = model.get_embedding(audio_a.unsqueeze(0))
                emb_b, act_b = model.get_embedding(audio_b.unsqueeze(0))
                
                dist = F.pairwise_distance(emb_a, emb_b).item()
                
                act_a = act_a.squeeze()
                act_b = act_b.squeeze()
                
                # Función interna para obtener qué expertos se activaron más
                def get_pathway_desc(activity, class_experts):
                    path_desc = []
                    for c in range(10):
                        idx = class_experts[c]
                        if not idx: continue
                        
                        # Promedio de spikes POR NEURONA del grupo
                        avg_spikes = activity[idx].sum().item() / len(idx)
                        
                        if avg_spikes > 1.5: # Umbral de activación promedio
                            path_desc.append((c, avg_spikes))
                    
                    # Ordenar por los que más dispararon en promedio
                    path_desc.sort(key=lambda x: x[1], reverse=True)
                    
                    res = [f"E{c}({avg:.1f} spk/n)" for c, avg in path_desc]
                    return " ".join(res) if res else "INACTIVO"
                
                ruta_a = get_pathway_desc(act_a, class_experts)
                ruta_b = get_pathway_desc(act_b, class_experts)

                name_a = f"{item_a[2]} ({item_a[1]})"
                name_b = f"{item_b[2]} ({item_b[1]})"

                f.write(f"{name_a:<20} | {name_b:<20} | {dist:.2f} | {ruta_a:<40} | {ruta_b:<40}\n")

    print(f"¡Reporte finalizado! Revisa {OUTPUT_FILE}")

if __name__ == "__main__":
    run_test()