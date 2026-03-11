import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio
import pywt
from tqdm import tqdm

from classes import SiameseSNN, AudioMNISTEvalDataset

# ==========================================
# 1. GLOBAL PARAMETERS
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "AudioMNIST_split"
BACKBONE_PATH = "src/demos/siameseSNN/models/snn_pop_audio_explainable.pth" 
SIAMESE_SAVE_PATH = "./src/demos/siameseSNN/models/snn_siamese_model.pt"
TARGET_LEN = 8000 

# ==========================================
# 2. UTILIDADES (VAD, RUIDO, DISTANCIAS)
# ==========================================
def apply_vad_and_align(waveform, threshold_ratio=0.05, target_len=TARGET_LEN):
    abs_wave = waveform.abs().squeeze()
    max_amp = abs_wave.max()
    if max_amp == 0: return torch.zeros((1, target_len))
    threshold = max_amp * threshold_ratio
    active_indices = torch.where(abs_wave > threshold)[0]
    if len(active_indices) == 0: return torch.zeros((1, target_len))
    start_idx, end_idx = active_indices[0].item(), active_indices[-1].item()
    trimmed = waveform[:, start_idx:end_idx+1]
    if trimmed.size(1) > target_len: return trimmed[:, :target_len]
    else: return F.pad(trimmed, (0, target_len - trimmed.size(1)))

def add_white_noise(waveform, snr_db):
    if snr_db is None: return waveform
    sig_avg_watts = torch.mean(waveform ** 2)
    sig_avg_db = 10 * torch.log10(sig_avg_watts)
    noise_avg_db = sig_avg_db - snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    noise = torch.randn_like(waveform) * torch.sqrt(noise_avg_watts)
    return waveform + noise

def calculate_rmse(vec1, vec2_matrix):
    """ RMSE para xAI: mide el error físico entre embeddings """
    mse = torch.mean((vec1 - vec2_matrix)**2, dim=1)
    return torch.sqrt(mse)

def van_rossum_distance(spk1, spk2, tau=0.1):
    T, N = spk1.shape
    s1, s2 = spk1.t().unsqueeze(1), spk2.t().unsqueeze(1)
    t = torch.arange(T, device=spk1.device).float()
    kernel = torch.exp(-t / (tau * T)).view(1, 1, -1)
    f1 = F.conv1d(s1, kernel, padding=T-1)[:, :, :T]
    f2 = F.conv1d(s2, kernel, padding=T-1)[:, :, :T]
    return torch.sqrt(torch.sum((f1 - f2)**2) / N).item()

# ==========================================
# 3. FEATURE EXTRACTORS
# ==========================================
def extract_mfcc(waveform, sample_rate=8000):
    aligned = apply_vad_and_align(waveform)
    mfcc_tr = torchaudio.transforms.MFCC(
        sample_rate=sample_rate, n_mfcc=13,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
    ).to(waveform.device)
    return mfcc_tr(aligned).contiguous().view(-1)

def extract_dwt(waveform, wavelet='db4', level=4):
    aligned = apply_vad_and_align(waveform)
    wave_np = aligned.squeeze(0).cpu().numpy()
    coeffs = pywt.wavedec(wave_np, wavelet, level=level)
    features = []
    for c in coeffs: features.extend([np.mean(c), np.std(c), np.sum(c**2)])
    features = np.array(features)
    return torch.tensor(features / (np.linalg.norm(features) + 1e-9), dtype=torch.float32)

# ==========================================
# 4. DATABASE BUILDER
# ==========================================
def build_all_databases(model, dataloader, device):
    db = {'snn':[], 'spikes':[], 'mfcc':[], 'dwt':[], 'raw_c': [], 'labels':[], 'filenames':[]}
    print("Building database...")
    model.eval()
    with torch.no_grad():
        for waveforms, labels, filenames in tqdm(dataloader):
            if waveforms.dim() == 4: waveforms = waveforms.squeeze(2)
            if waveforms.dim() == 2: waveforms = waveforms.unsqueeze(1)
            waveforms_gpu = waveforms.to(device)
            
            snn_embs, _ = model.get_embedding(waveforms_gpu)
            _, spk_hid = model.backbone(waveforms_gpu)
            
            db['snn'].append(snn_embs.cpu())
            db['spikes'].append(spk_hid.transpose(0, 1).cpu())
            
            for i in range(waveforms.size(0)):
                w = waveforms[i]
                db['mfcc'].append(extract_mfcc(w).unsqueeze(0).cpu())
                db['dwt'].append(extract_dwt(w).unsqueeze(0).cpu())
                raw = apply_vad_and_align(w).squeeze(0).cpu()
                db['raw_c'].append((raw - raw.mean()).unsqueeze(0))
                db['labels'].append(labels[i].item())
                db['filenames'].append(filenames[i])

    for k in ['snn', 'spikes', 'mfcc', 'dwt', 'raw_c']: db[k] = torch.cat(db[k], dim=0)
    db['labels'] = torch.tensor(db['labels'])
    db['filenames'] = np.array(db['filenames'])
    return db

# ==========================================
# 5. EVALUACIÓN DE ROBUSTEZ (5 MÉTODOS)
# ==========================================
def evaluate_robustness(q_wave_clean, q_label, q_fname, model, db, snr_db, k=5, device='cuda'):
    model.eval()
    q_wave = add_white_noise(q_wave_clean, snr_db)
    if q_wave.dim() == 1: q_wave = q_wave.unsqueeze(0).unsqueeze(0)
    if q_wave.dim() == 2: q_wave = q_wave.unsqueeze(0)
        
    with torch.no_grad():
        q_snn, _ = model.get_embedding(q_wave.to(device))
        _, q_spk_rec = model.backbone(q_wave.to(device))
        q_snn, q_spk = q_snn.cpu(), q_spk_rec.cpu().squeeze(1) 

    q_mfcc = extract_mfcc(q_wave.squeeze(0)).unsqueeze(0).cpu()
    q_dwt = extract_dwt(q_wave.squeeze(0)).unsqueeze(0).cpu()
    q_raw = apply_vad_and_align(q_wave.squeeze(0)).squeeze(0).cpu()
    q_raw_c = (q_raw - q_raw.mean()).unsqueeze(0)

    mask = (db['filenames'] != q_fname)
    db_labels = db['labels'][mask]
    res = {}

    # 1. SNN Siamese (Euclidean)
    d_snn = torch.cdist(q_snn, db['snn'][mask], p=2).squeeze(0)
    res["SNN Siamese"] = (db_labels[torch.topk(d_snn, k, largest=False)[1]] == q_label).float().mean().item()

    # 2. MFCC (Cosine)
    d_mfcc = 1 - F.cosine_similarity(q_mfcc, db['mfcc'][mask])
    res["MFCC"] = (db_labels[torch.topk(d_mfcc, k, largest=False)[1]] == q_label).float().mean().item()

    # 3. DWT (Euclidean)
    d_dwt = torch.cdist(q_dwt, db['dwt'][mask], p=2).squeeze(0)
    res["DWT"] = (db_labels[torch.topk(d_dwt, k, largest=False)[1]] == q_label).float().mean().item()

    # 4. Pearson (Cosine on centered)
    d_pearson = 1 - F.cosine_similarity(q_raw_c, db['raw_c'][mask])
    res["Pearson"] = (db_labels[torch.topk(d_pearson, k, largest=False)[1]] == q_label).float().mean().item()

    # 5. Van Rossum (Re-ranking)
    _, top20 = torch.topk(d_snn, 20, largest=False)
    vr_dists = torch.tensor([van_rossum_distance(q_spk, db['spikes'][mask][idx]) for idx in top20])
    vr_top_k = db_labels[top20[torch.topk(vr_dists, k, largest=False)[1]]]
    res["Van Rossum"] = (vr_top_k == q_label).float().mean().item()

    return res

# ==========================================
# 6. RECALL POR CLASE (RMSE)
# ==========================================
def evaluate_recall_rmse(db, k=5):
    labels = db['labels'].unique().tolist()
    stats = {int(l): {'hits': 0, 'total': 0} for l in labels}
    
    print(f"\nCalculando Recall@K (K={k}) por clase usando RMSE...")
    for i in tqdm(range(len(db['filenames']))):
        q_emb, q_label, q_fname = db['snn'][i].unsqueeze(0), int(db['labels'][i]), db['filenames'][i]
        
        dists = calculate_rmse(q_emb, db['snn'])
        mask = (db['filenames'] != q_fname)
        
        _, top_idxs = torch.topk(dists[mask], k, largest=False)
        hits = (db['labels'][mask][top_idxs] == q_label).sum().item()
        
        stats[q_label]['hits'] += (hits / k)
        stats[q_label]['total'] += 1

    return {l: (stats[l]['hits']/stats[l]['total'])*100 for l in stats}

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    model = SiameseSNN(BACKBONE_PATH).to(device)
    ckpt = torch.load(SIAMESE_SAVE_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state'] if 'model_state' in ckpt else ckpt)

    test_ds = AudioMNISTEvalDataset(f"{DATA_ROOT}/test")
    db = build_all_databases(model, DataLoader(test_ds, batch_size=32), device)

    # --- TEST 1: ROBUSTEZ ---
    levels = [None, 10, 0]
    methods = ["SNN Siamese", "Van Rossum", "MFCC", "DWT", "Pearson"]
    N_SAMPLES = 100
    indices = random.sample(range(len(test_ds)), N_SAMPLES)
    
    robust_report = {lvl: {m: 0.0 for m in methods} for lvl in levels}
    for lvl in levels:
        print(f"\nEvaluando Robustez SNR: {lvl if lvl is not None else 'Clean'} dB")
        for i in tqdm(indices):
            res = evaluate_robustness(*test_ds[i], model, db, snr_db=lvl, device=device)
            for m in methods: robust_report[lvl][m] += res[m]

    print(f"\n{'='*85}")
    print(f"{'Métrica':<15} | {'Limpio':<12} | {'SNR 10dB':<12} | {'SNR 0dB':<12}")
    print(f"{'='*85}")
    for m in methods:
        vals = [f"{(robust_report[l][m]/N_SAMPLES)*100:>10.1f}%" for l in levels]
        print(f"{m:<15} | {' | '.join(vals)}")

    # --- TEST 2: RECALL POR CLASE (RMSE) ---
    recall_class = evaluate_recall_rmse(db, k=5)
    print(f"\n{'='*40}")
    print(f"{'Dígito':<10} | {'Recall@5 (RMSE)':<20}")
    print(f"{'='*40}")
    for l in sorted(recall_class.keys()):
        print(f"Clase {l:<5} | {recall_class[l]:>18.2f}%")
    print(f"{'='*40}")
    print(f"MEDIA TOTAL | {sum(recall_class.values())/len(recall_class):>18.2f}%")