import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio
import pywt
from tqdm import tqdm

from classes import SiameseSNN, AudioMNISTEvalDataset

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "AudioMNIST_split"
BACKBONE_PATH = "src/demos/siameseSNN/models/snn_pop_audio_explainable.pth" 
SIAMESE_SAVE_PATH = "./src/demos/siameseSNN/models/snn_siamese_model.pt"
DB_CACHE_PATH = "db_cache.pt" 
TARGET_LEN = 8000 

bundle = torchaudio.pipelines.WAV2VEC2_BASE
gold_model = bundle.get_model().to(device)
gold_model.eval()

# Metrics functions
def extract_gold_standard(waveform, sample_rate=8000):
    with torch.no_grad():
        resampler = torchaudio.transforms.Resample(sample_rate, 16000).to(device)
        wave_16k = resampler(waveform.to(device))
        if wave_16k.dim() == 3: wave_16k = wave_16k.squeeze(1)
        features, _ = gold_model(wave_16k)
        return features.mean(dim=1).cpu()

def apply_vad_and_align(waveform, threshold_ratio=0.05, target_len=TARGET_LEN):
    abs_wave = waveform.abs().squeeze()
    if abs_wave.dim() == 0: return torch.zeros((1, target_len))
    max_amp = abs_wave.max()
    if max_amp == 0: return torch.zeros((1, target_len))
    threshold = max_amp * threshold_ratio
    active_indices = torch.where(abs_wave > threshold)[0]
    if len(active_indices) == 0: return torch.zeros((1, target_len))
    start_idx, end_idx = active_indices[0].item(), active_indices[-1].item()
    trimmed = waveform[:, start_idx:end_idx+1]
    if trimmed.size(1) > target_len: return trimmed[:, :target_len]
    else: return F.pad(trimmed, (0, target_len - trimmed.size(1)))

def van_rossum_distance(spk1, spk2, tau=0.1):
    T, N = spk1.shape
    s1, s2 = spk1.t().unsqueeze(1), spk2.t().unsqueeze(1)
    t = torch.arange(T, device=spk1.device).float()
    kernel = torch.exp(-t / (tau * T)).view(1, 1, -1)
    f1 = F.conv1d(s1, kernel, padding=T-1)[:, :, :T]
    f2 = F.conv1d(s2, kernel, padding=T-1)[:, :, :T]
    return torch.sqrt(torch.sum((f1 - f2)**2) / N).item()

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
    features =[]
    for c in coeffs: features.extend([np.mean(c), np.std(c), np.sum(c**2)])
    features = np.array(features)
    return torch.tensor(features / (np.linalg.norm(features) + 1e-9), dtype=torch.float32)

# Database construction (with caching)
def get_or_build_database(model, dataloader, device):
    if os.path.exists(DB_CACHE_PATH):
        print(f"Loading database from cache: {DB_CACHE_PATH}")
        return torch.load(DB_CACHE_PATH, weights_only=False)
        
    db = {'snn':[], 'spikes':[], 'mfcc':[], 'dwt':[], 'raw_c': [], 
          'gold_vector':[], 'labels':[], 'filenames':[]}
    
    print("Building database from model embeddings and features...")
    model.eval()
    with torch.no_grad():
        for waveforms, labels, filenames in tqdm(dataloader):
            if waveforms.dim() == 4: waveforms = waveforms.squeeze(2)
            if waveforms.dim() == 2: waveforms = waveforms.unsqueeze(1)
            waveforms_gpu = waveforms.to(device)
            
            snn_embs, _ = model.get_embedding(waveforms_gpu)
            _, spk_hid = model.backbone(waveforms_gpu)
            gold_v = extract_gold_standard(waveforms_gpu)
            
            db['snn'].append(snn_embs.cpu())
            db['spikes'].append(spk_hid.transpose(0, 1).cpu())
            db['gold_vector'].append(gold_v.cpu())
            
            for i in range(waveforms.size(0)):
                w = waveforms[i]
                db['mfcc'].append(extract_mfcc(w).unsqueeze(0).cpu())
                db['dwt'].append(extract_dwt(w).unsqueeze(0).cpu())
                raw = apply_vad_and_align(w).squeeze(0).cpu()
                db['raw_c'].append((raw - raw.mean()).unsqueeze(0))
                db['labels'].append(labels[i].item())
                db['filenames'].append(filenames[i])

    for k in['snn', 'spikes', 'mfcc', 'dwt', 'raw_c', 'gold_vector']: 
        db[k] = torch.cat(db[k], dim=0)
    db['labels'] = torch.tensor(db['labels'])
    db['filenames'] = np.array(db['filenames'])
    
    torch.save(db, DB_CACHE_PATH)
    return db

# Benchmark
def run_latent_quality_benchmark(db, k=5):
    methods =["SNN Siamese", "Van Rossum", "MFCC", "DWT", "Pearson"]
    results = {m: {'recall': [], 'factual_rmse': [], 'cf_rmse':[], 'time':[]} for m in methods}
    
    
    for i in tqdm(range(len(db['filenames']))):
        q_label = db['labels'][i].item()
        q_fname = db['filenames'][i]
        q_gold = db['gold_vector'][i].unsqueeze(0)
        
        mask_self = torch.from_numpy(db['filenames'] != q_fname)
        db_labels_self = db['labels'][mask_self]
        db_gold_self = db['gold_vector'][mask_self]
        
        mask_same_label = (db_labels_self == q_label)
        mask_diff_label = ~mask_same_label

        # Methos Eval
        for m in ["SNN Siamese", "MFCC", "DWT", "Pearson"]:
            t0 = time.perf_counter()
            
            if m == "SNN Siamese":
                dists = torch.cdist(db['snn'][i:i+1], db['snn'][mask_self], p=2).squeeze(0)
            elif m == "MFCC":
                dists = 1 - F.cosine_similarity(db['mfcc'][i:i+1], db['mfcc'][mask_self])
            elif m == "DWT":
                dists = torch.cdist(db['dwt'][i:i+1], db['dwt'][mask_self], p=2).squeeze(0)
            elif m == "Pearson":
                dists = 1 - F.cosine_similarity(db['raw_c'][i:i+1], db['raw_c'][mask_self])
            
            topk_all = torch.topk(dists, k, largest=False)[1]
            
            t1 = time.perf_counter()
            results[m]['time'].append((t1 - t0) * 1000) 
            
            # Recall
            results[m]['recall'].append((db_labels_self[topk_all] == q_label).float().mean().item())
            
            # Factual RMSE 
            dists_f = dists[mask_same_label]
            if len(dists_f) > 0:
                topk_f = torch.topk(dists_f, min(k, len(dists_f)), largest=False)[1]
                rmse_f = torch.sqrt(torch.mean((q_gold - db_gold_self[mask_same_label][topk_f])**2, dim=1)).mean().item()
                results[m]['factual_rmse'].append(rmse_f)
            
            # Contrafactual RMSE 
            dists_cf = dists[mask_diff_label]
            if len(dists_cf) > 0:
                top1_cf = torch.topk(dists_cf, 1, largest=False)[1]
                rmse_cf = torch.sqrt(torch.mean((q_gold - db_gold_self[mask_diff_label][top1_cf])**2, dim=1)).mean().item()
                results[m]['cf_rmse'].append(rmse_cf)

        # Van Rossum
        # Chrono start
        t0_vr = time.perf_counter()
        
        q_spk = db['spikes'][i]
        d_snn = torch.cdist(db['snn'][i:i+1], db['snn'][mask_self], p=2).squeeze(0)
        top20_snn = torch.topk(d_snn, min(20, len(d_snn)), largest=False)[1]
        
        vr_dists = torch.tensor([van_rossum_distance(q_spk, db['spikes'][mask_self][idx]) for idx in top20_snn])
        vr_topk_all = top20_snn[torch.topk(vr_dists, min(k, len(vr_dists)), largest=False)[1]]
        
        # Chrono end
        t1_vr = time.perf_counter()
        results["Van Rossum"]['time'].append((t1_vr - t0_vr) * 1000)
        
        # Recall RMSE
        results["Van Rossum"]['recall'].append((db_labels_self[vr_topk_all] == q_label).float().mean().item())
        
        # Factual RMSE
        mask_vr_f = (db_labels_self[top20_snn] == q_label)
        if mask_vr_f.any():
            vr_dists_f = vr_dists[mask_vr_f]
            vr_idxs_f = top20_snn[mask_vr_f]
            vr_topk_f = vr_idxs_f[torch.topk(vr_dists_f, min(k, len(vr_dists_f)), largest=False)[1]]
            results["Van Rossum"]['factual_rmse'].append(torch.sqrt(torch.mean((q_gold - db_gold_self[vr_topk_f])**2, dim=1)).mean().item())
        
        # Contrafactual RMSE
        mask_vr_cf = ~mask_vr_f
        if mask_vr_cf.any():
            vr_dists_cf = vr_dists[mask_vr_cf]
            vr_idxs_cf = top20_snn[mask_vr_cf]
            vr_top1_cf = vr_idxs_cf[torch.argmin(vr_dists_cf)]
            results["Van Rossum"]['cf_rmse'].append(torch.sqrt(torch.mean((q_gold - db_gold_self[vr_top1_cf].unsqueeze(0))**2, dim=1)).mean().item())

    # Final report
    print(f"\n{'='*120}")
    print(f"{'Method':<15} | {'TIME/Q (ms)':<15} | {'RECALL @5':<12} | {'FACTUAL RMSE (lower better)':<18} | {'CF RMSE (higher better)':<20} | {'MARGIN xAI (higher better)'}")
    print(f"{'='*120}")
    for m in methods:
        rec = np.mean(results[m]['recall']) * 100
        f_rmse = np.mean(results[m]['factual_rmse'])
        cf_rmse = np.mean(results[m]['cf_rmse'])
        margen = cf_rmse - f_rmse
        avg_time = np.mean(results[m]['time'])
        
        time_str = f"{avg_time:>9.2f}" + (" *" if m == "Van Rossum" else "  ")
        
        print(f"{m:<15} | {time_str:<15} | {rec:>10.2f}% | {f_rmse:>18.6f} | {cf_rmse:>20.6f} | {margen:>15.6f}")
    print(f"{'='*120}")
    print("* Note: The time of Van Rossum includes a pre-filtering over the top-20 to make it viable.")
    print("        If applied to the complete DB, the time would be orders of magnitude higher.")

# main execution
if __name__ == "__main__":
    model = SiameseSNN(BACKBONE_PATH).to(device)
    ckpt = torch.load(SIAMESE_SAVE_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state'] if 'model_state' in ckpt else ckpt)

    test_ds = AudioMNISTEvalDataset(f"{DATA_ROOT}/test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    db = get_or_build_database(model, test_loader, device)

    run_latent_quality_benchmark(db, k=5)