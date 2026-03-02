import torch
import torch.nn as nn
import snntorch as snn
from snntorch import utils, surrogate
import torchaudio
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
from classes import AudioMNISTEvalDataset, PopNetAudio
# --- CONFIGURACIÓN (Debe coincidir con el entrenamiento) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "AudioMNIST_split"
MODEL_PATH = "./src/demos/siameseSNN/models/snn_pop_audio.pth"
OUTPUT_FILE = "./src/demos/siameseSNN/results/biological_pathway_results.txt"

num_steps = 50
beta = 0.9
v_threshold = 1.2
spike_grad = surrogate.atan()


def run_biological_test():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encontró el modelo en {MODEL_PATH}")
        return

    # A. Cargar Modelo y Mapa de Expertos
    print("Cargando modelo y mapeando expertos por pesos...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = PopNetAudio().to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # Si el checkpoint no tiene los expertos, los calculamos ahora (IDEM al método Iris)
    if 'class_experts' in checkpoint:
        class_experts = checkpoint['class_experts']
    else:
        # Analizamos fc2: qué neuronas tienen pesos más fuertes
        weights = model.fc2.weight.data.cpu() 
        class_experts = {i: [] for i in range(10)}
        for n_idx in range(256):
            w_neuron = weights[:, n_idx]
            # Buscamos la mejor y la segunda mejor
            sorted_w, _ = torch.sort(w_neuron, descending=True)
            best_class = torch.argmax(w_neuron).item()
            
            # FILTRO MÁS SEGURO:
            # 1. El peso debe ser positivo.
            # 2. Debe ser al menos un poco mejor que el segundo (margen > 0.05).
            margin = sorted_w[0] - sorted_w[1]
            if sorted_w[0] > 0 and margin > 0.05:
                class_experts[best_class].append(n_idx)

        for c in range(10):
            print(f"Expertos Clase {c}: {len(class_experts[c])}")

    # B. Preparar Datos de Test
    test_ds = AudioMNISTEvalDataset(f"{DATA_ROOT}/test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # C. Inferencia y Reporte
    print(f"Procesando {len(test_ds)} archivos...")
    matches = 0
    
    with open(OUTPUT_FILE, "w") as f:
        f.write(f"{'ID':<25} | {'REAL':<4} | {'PRED':<4} | {'ACTIVACIÓN DE SECUENCIAS EXPERTAS (E0-E9)'}\n")
        f.write("-" * 115 + "\n")

        with torch.no_grad():
            for data, label, filename in tqdm(test_loader):
                data = data.to(device)
                spk_out, spk_hid = model(data) # spk_hid: [steps, 1, 256]

                total_hid_spikes = spk_hid.sum().item()
                if total_hid_spikes == 0:
                     print(f"ALERTA: El archivo {filename[0]} no generó NI UN SOLO SPIKE en la capa oculta.")
                
                # Predicción: neurona de salida con más spikes
                pred = spk_out.sum(dim=0).argmax(dim=1).item()
                real = label.item()
                if pred == real: matches += 1
                
                # Actividad total de la población oculta (suma en el tiempo)
                hid_activity = spk_hid.sum(dim=0).squeeze() # [256]
                
                # Contar cuánto disparó cada grupo de expertos
                expert_activations = []
                for c in range(10):
                    indices = class_experts[c]
                    if len(indices) > 0:
                        total_act = hid_activity[indices].sum().item()
                    else:
                        total_act = 0
                    expert_activations.append(int(total_act))

                counts_str = " ".join([f"E{i}:{c:<4}" for i, c in enumerate(expert_activations)])
                status = "✓" if pred == real else "✗"
                
                f.write(f"{filename[0]:<25} | {real:<4} | {pred:<4} {status} | {counts_str}\n")

        acc = 100 * matches / len(test_ds)
        f.write("-" * 115 + "\n")
        f.write(f"ACCURACY FINAL: {acc:.2f}%\n")

    print(f"¡Hecho! Reporte generado en: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_biological_test()