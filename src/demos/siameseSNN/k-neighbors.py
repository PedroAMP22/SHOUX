import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio
import pywt  

from classes import SiameseSNN, AudioMNISTEvalDataset

# ==========================================
# 1. GLOBAL SNN PARAMETERS
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "AudioMNIST_split"
BACKBONE_PATH = "src/demos/siameseSNN/models/snn_pop_audio_explainable.pth" 
SIAMESE_SAVE_PATH = "./src/demos/siameseSNN/models/snn_siamese_model.pt"

# ==========================================
# 2. VAD Y VAN ROSSUM
# ==========================================
def apply_vad_and_align(waveform, threshold_ratio=0.05):
    abs_wave = waveform.abs().squeeze(0)
    max_amp = abs_wave.max()
    threshold = max_amp * threshold_ratio
    active_indices = torch.where(abs_wave > threshold)[0]
    if len(active_indices) == 0: return waveform
    start_idx = active_indices[0].item()
    cropped = waveform[:, start_idx:]
    return torch.nn.functional.pad(cropped, (0, waveform.size(1) - cropped.size(1)))

def van_rossum_distance(spk1, spk2, tau=0.1):
    """
    spk1, spk2: [num_steps, num_neurons]
    """
    T = spk1.size(0)
    t = torch.arange(T, device=spk1.device).float()
    kernel = torch.exp(-t / (tau * T)).view(1, 1, -1)
    
    # Convolución por neurona (spk1.t() es [neuronas, steps])
    f1 = F.conv1d(spk1.t().unsqueeze(1), kernel, padding=T-1)[:, :, :T]
    f2 = F.conv1d(spk2.t().unsqueeze(1), kernel, padding=T-1)[:, :, :T]
    return torch.sqrt(torch.sum((f1 - f2)**2))

# ==========================================
# 3. BASELINE FEATURE EXTRACTORS
# ==========================================
def extract_mfcc(waveform, sample_rate=8000):
    aligned_wave = apply_vad_and_align(waveform)
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate, n_mfcc=13,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
    )
    mfcc = mfcc_transform(aligned_wave)
    
    return mfcc.contiguous().reshape(-1)

def extract_dwt(waveform, wavelet='db4', level=4):
    aligned_wave = apply_vad_and_align(waveform)
    wave_np = aligned_wave.squeeze(0).numpy()
    coeffs = pywt.wavedec(wave_np, wavelet, level=level)
    features = []
    for c in coeffs:
        features.extend([np.mean(c), np.std(c), np.sum(c**2)])
    features = np.array(features)
    norm = np.linalg.norm(features)
    return torch.tensor(features / (norm + 1e-9), dtype=torch.float32)

def extract_raw(waveform):
    return apply_vad_and_align(waveform).squeeze(0)

# ==========================================
# 4. BUILDER
# ==========================================
def build_all_databases(model, dataloader, device):
    db = {'snn':[], 'spikes':[], 'mfcc':[], 'dwt':[], 'pearson': [], 'labels': [], 'filenames':[]}
    print("Building database...")
    model.eval()
    with torch.no_grad():
        for waveforms, labels, filenames in dataloader:
            # waveforms viene como [32, 1, 8000] (si el dataset ya tiene el canal)
            # o [32, 8000] (si no lo tiene).
            
            # Si tiene 4 dimensiones [32, 1, 1, 8000], lo aplanamos a 3D
            if waveforms.dim() == 4:
                waveforms = waveforms.squeeze(2) # Elimina la dimensión extra
            
            # Si solo tiene 2 dimensiones [32, 8000], añadimos el canal
            elif waveforms.dim() == 2:
                waveforms = waveforms.unsqueeze(1)
            
            # Ahora waveforms es [32, 1, 8000]
            waveforms_gpu = waveforms.to(device)
            
            snn_embs, spk_hid = model.get_embedding(waveforms_gpu)
            db['snn'].append(snn_embs.cpu())
            db['spikes'].append(spk_hid.cpu())
            
            for i in range(waveforms.size(0)):
                wave = waveforms[i]
                db['mfcc'].append(extract_mfcc(wave).unsqueeze(0))
                db['dwt'].append(extract_dwt(wave).unsqueeze(0))
                db['pearson'].append(extract_raw(wave).unsqueeze(0))
                db['labels'].append(labels[i].item())
                db['filenames'].append(filenames[i])

    db['snn'] = torch.cat(db['snn'], dim=0)
    db['spikes'] = torch.cat(db['spikes'], dim=1).transpose(0, 1) # [N, steps, neurons]
    db['mfcc'] = torch.cat(db['mfcc'], dim=0)
    db['dwt'] = torch.cat(db['dwt'], dim=0)
    db['pearson'] = torch.cat(db['pearson'], dim=0)
    return db

# ==========================================
# 5. EVALUATION
# ==========================================
def evaluate_query(q_wave, q_label, q_fname, model, db, k=5, device='cuda'):
    model.eval()
    if q_wave.dim() == 2:
        q_wave = q_wave.unsqueeze(0) # Si era [1, 8000], ahora es [1, 1, 8000]
    elif q_wave.dim() == 1:
        q_wave = q_wave.unsqueeze(0).unsqueeze(0) # Si era [8000], ahora es [1, 1, 8000]
    with torch.no_grad():
        q_snn, q_spk = model.get_embedding(q_wave.to(device))
        q_snn, q_spk = q_snn.cpu(), q_spk.cpu().squeeze(1) # [steps, neurons]

    q_mfcc = extract_mfcc(q_wave.squeeze(0)).unsqueeze(0)
    q_dwt = extract_dwt(q_wave.squeeze(0)).unsqueeze(0)
    q_pearson = extract_raw(q_wave.squeeze(0)).unsqueeze(0)

    # 1. Métodos Estándar
    results = {}
    feats = {"SNN Siamese": (q_snn, db['snn'], "euclidean"), "MFCC": (q_mfcc, db['mfcc'], "cosine"),
             "DWT": (q_dwt, db['dwt'], "euclidean"), "Pearson": (q_pearson, db['pearson'], "pearson")}

    for name, (q_f, d_f, mode) in feats.items():
        if mode == "euclidean": dists = torch.cdist(q_f, d_f, p=2).squeeze(0)
        elif mode == "cosine": dists = 1 - F.cosine_similarity(q_f, d_f)
        else: # Pearson
            q_c = q_f - q_f.mean(); d_c = d_f - d_f.mean(dim=1, keepdim=True)
            dists = 1 - F.cosine_similarity(q_c, d_c)
        
        # Filtrado para Precision
        _, idxs = torch.topk(dists, k + 1, largest=False)
        labels = [db['labels'][idx] for idx in idxs if db['filenames'][idx] != q_fname][:k]
        results[name] = sum([1 for l in labels if l == q_label]) / k

    # 2. Van Rossum (Re-ranking sobre top 20 para eficiencia)
    dists_snn = torch.cdist(q_snn, db['snn'], p=2).squeeze(0)
    top20_idx = torch.topk(dists_snn, 20, largest=False)[1]
    vr_dists = torch.tensor([van_rossum_distance(q_spk, db['spikes'][i]) for i in top20_idx])
    final_vr_idx = top20_idx[torch.argsort(vr_dists)][:k]
    vr_labels = [db['labels'][i] for i in final_vr_idx]
    results["Van Rossum"] = sum([1 for l in vr_labels if l == q_label]) / k

    return results

if __name__ == "__main__":
    model = SiameseSNN(BACKBONE_PATH).to(device)

    checkpoint = torch.load(SIAMESE_SAVE_PATH, map_location=device)
    state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True) 

    
    print("Modelo cargado correctamente ignorando buffers estadísticos.")

    test_ds = AudioMNISTEvalDataset(f"{DATA_ROOT}/test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False) # Agregamos batch_size   
    db = build_all_databases(model, test_loader, device)

    avg_prec = {m: 0.0 for m in ["SNN Siamese", "MFCC", "DWT", "Pearson", "Van Rossum"]}
    for i in random.sample(range(len(test_ds)), 100):
        res = evaluate_query(*test_ds[i], model, db)
        for m in avg_prec: avg_prec[m] += res[m]

    print(f"\n{'='*50}\nFINAL PRECISION@5 (Más alto es mejor)\n{'='*50}")
    for m, p in sorted(avg_prec.items(), key=lambda x: x[1], reverse=True):
        print(f"{m}: {p:.2f}%")