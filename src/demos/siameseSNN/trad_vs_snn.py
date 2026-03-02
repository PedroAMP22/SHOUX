import torch
import torch.nn.functional as F
import numpy as np
import librosa
import os
import random
from tqdm import tqdm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from classes import SiameseSNN, SiameseAudioMNIST # Asegúrate de que coincida con tu archivo

# --- CONFIGURACIÓN ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BACKBONE_CHECKPOINT = "./src/demos/siameseSNN/models/snn_pop_audio.pth"
SIAMESE_MODEL_PATH = "./src/demos/siameseSNN/models/snn_siamese_model.pt"
DATA_ROOT = "AudioMNIST_split"
OUTPUT_FILE = "./src/demos/siameseSNN/results/benchmark_detallado.txt"
NUM_TESTS = 50 

class AudioComparator:
    def __init__(self, sample_rate=8000):
        self.sr = sample_rate

    def get_traditional_metrics(self, y1, y2):
        y1, y2 = y1.flatten(), y2.flatten()
        metrics = {}
        # MFCC
        mfcc1 = librosa.feature.mfcc(y=y1, sr=self.sr, n_mfcc=13)
        mfcc2 = librosa.feature.mfcc(y=y2, sr=self.sr, n_mfcc=13)
        v1, v2 = np.mean(mfcc1, axis=1), np.mean(mfcc2, axis=1)
        metrics['mfcc_cos'] = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        # DTW
        s1, s2 = y1[::10].reshape(-1, 1), y2[::10].reshape(-1, 1)
        d_dist, _ = fastdtw(s1, s2, dist=euclidean)
        metrics['dtw_dist'] = d_dist
        # Pearson
        metrics['corr'] = np.corrcoef(y1, y2)[0, 1]
        return metrics

# MODIFICACIÓN DEL DATASET PARA EL REPORTE
class SiameseEvalDataset(SiameseAudioMNIST):
    def __getitem__(self, idx):
        x1, label1 = self._get_audio(idx)
        is_same = random.randint(0, 1)
        if is_same:
            idx2 = random.choice(self.label_to_indices[label1])
            x2, label2 = self._get_audio(idx2)
        else:
            label2 = random.choice([l for l in range(10) if l != label1])
            idx2 = random.choice(self.label_to_indices[label2])
            x2, label2 = self._get_audio(idx2)
        # Devolvemos también label1 y label2
        return x1, x2, label1, label2, torch.tensor(is_same, dtype=torch.float32)

def run_benchmark():
    model = SiameseSNN(BACKBONE_CHECKPOINT).to(device)
    model.load_state_dict(torch.load(SIAMESE_MODEL_PATH, map_location=device))
    model.eval()

    comparator = AudioComparator(sample_rate=8000)
    # Usamos el dataset modificado para evaluación
    test_ds = SiameseEvalDataset(f"{DATA_ROOT}/test")

    print(f"Generando benchmark detallado en {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, "w") as f:
        f.write("BENCHMARK DETALLADO: SNN SIAMESA VS MÉTODOS TRADICIONALES\n")
        f.write("="*145 + "\n")
        header = f"{'PAR':<4} | {'CLASES':<10} | {'TIPO':<5} | {'SNN DIST':<10} | {'MFCC COS':<10} | {'DTW DIST':<10} | {'CORR':<8}\n"
        f.write(header)
        f.write("-" * 145 + "\n")

        for i in tqdm(range(NUM_TESTS)):
            idx = random.randint(0, len(test_ds)-1)
            x1, x2, c1, c2, label = test_ds[idx]
            
            with torch.no_grad():
                # Desempaquetamos (emb, activity)
                emb1, _ = model.get_embedding(x1.unsqueeze(0).to(device))
                emb2, _ = model.get_embedding(x2.unsqueeze(0).to(device))
                snn_dist = F.pairwise_distance(emb1, emb2).item()

            y1, y2 = x1.squeeze().numpy(), x2.squeeze().numpy()
            trad = comparator.get_traditional_metrics(y1, y2)

            tipo = "SAME" if label == 1 else "DIFF"
            clases_str = f"{c1} vs {c2}"
            
            line = (f"{i:03d}  | {clases_str:<10} | {tipo:<5} | {snn_dist:<10.4f} | {trad['mfcc_cos']:<10.4f} | "
                    f"{trad['dtw_dist']:<10.1f} | {trad['corr']:.4f}\n")
            f.write(line)

    print(f"Benchmark completado.")

if __name__ == "__main__":
    run_benchmark()