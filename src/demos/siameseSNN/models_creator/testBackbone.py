import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from classes import AudioMNISTEvalDataset, PopNetAudio

# --- CONFIGURACIÓN ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "AudioMNIST_split"
# Asegúrate de usar la ruta del modelo guardado con la nueva arquitectura
MODEL_PATH = "./src/demos/siameseSNN/models/snn_pop_audio_explainable.pth" 
OUTPUT_FILE = "./src/demos/siameseSNN/results/biological_pathway_results.txt"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def run_biological_test():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encontró el modelo en {MODEL_PATH}")
        return

    # A. Cargar Modelo
    print("Cargando modelo de votación directa...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Instanciamos el modelo con los mismos parámetros que el entrenamiento
    model = PopNetAudio(num_neurons_hid=250, num_classes=10).to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    # B. Mapeo de Expertos (Ahora es fijo por diseño)
    # Ya no necesitamos calcularlo, es una propiedad estructural de la red
    neurons_per_class = checkpoint['neurons_per_class']
    class_experts = {i: list(range(i * neurons_per_class, (i + 1) * neurons_per_class)) for i in range(10)}

    # C. Preparar Datos de Test
    test_ds = AudioMNISTEvalDataset(f"{DATA_ROOT}/test")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # D. Inferencia y Reporte
    print(f"Procesando {len(test_ds)} archivos...")
    matches = 0
    
    with open(OUTPUT_FILE, "w") as f:
        f.write(f"{'ID':<25} | {'REAL':<4} | {'PRED':<4} | {'ACTIVACIÓN DE SECUENCIAS EXPERTAS (E0-E9)'}\n")
        f.write("-" * 115 + "\n")

        with torch.no_grad():
            for data, label, filename in tqdm(test_loader):
                data = data.to(device)
                # spk_out: [steps, batch, num_classes]
                # spk_hid: [steps, batch, num_neurons_hid]
                spk_out, spk_hid = model(data) 

                # Inferencia: Suma de spikes de cada grupo experto en el tiempo
                # spk_out ya es la suma de los grupos expertos por diseño
                pred = spk_out.sum(dim=0).argmax(dim=1).item()
                real = label.item()
                if pred == real: matches += 1
                
                # Actividad total de la población oculta (suma en el tiempo)
                hid_activity = spk_hid.sum(dim=0).squeeze() # [250]
                
                # Contar cuánto disparó cada grupo de expertos
                expert_activations = []
                for c in range(10):
                    indices = class_experts[c]
                    # Sumamos la actividad de las 25 neuronas de este experto
                    total_act = hid_activity[indices].sum().item()
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