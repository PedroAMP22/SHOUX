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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "AudioMNIST_split"
BACKBONE_PATH = "src/demos/siameseSNN/models/snn_pop_audio_explainable.pth" 
SIAMESE_SAVE_PATH = "./src/demos/siameseSNN/models/snn_siamese_model.pt"
DB_CACHE_PATH = "db_cache.pt" 
TARGET_LEN = 8000 

print(f"Using {device}")
bundle = torchaudio.pipelines.WAV2VEC2_BASE
gold_model = bundle.get_model().to(device)
gold_model.eval()


def extract_gold_standard(waveform, sample_rate=8000):
    with torch.no_grad():
        resampler = torchaudio.transforms.Resample(sample_rate, 16000).to(device)
        wave_16k = resampler(waveform.to(device))
        if wave_16k.dim() == 3: wave_16k = wave_16k.squeeze(1)
        features, _ = gold_model(wave_16k)
        return features.mean(dim=1).cpu()

def apply_vad_and_align(waveform, threshold_ratio=0.05, target_len=TARGET_LEN):
    abs_wave = waveform.abs().squeeze()
    if abs_wave.dim() == 0 or abs_wave.max() == 0: return torch.zeros((1, target_len))
    threshold = abs_wave.max() * threshold_ratio
    active_indices = torch.where(abs_wave > threshold)[0]
    if len(active_indices) == 0: return torch.zeros((1, target_len))
    trimmed = waveform[:, active_indices[0].item():active_indices[-1].item()+1]
    if trimmed.size(1) > target_len: return trimmed[:, :target_len]
    return F.pad(trimmed, (0, target_len - trimmed.size(1)))

def van_rossum_distance(spk1, spk2, tau=0.1):
    T, N = spk1.shape
    s1, s2 = spk1.t().unsqueeze(1), spk2.t().unsqueeze(1)
    t = torch.arange(T, device=spk1.device).float()
    kernel = torch.exp(-t / (tau * T)).view(1, 1, -1)
    f1 = F.conv1d(s1, kernel, padding=T-1)[:, :, :T]
    f2 = F.conv1d(s2, kernel, padding=T-1)[:, :, :T]
    return torch.sqrt(torch.sum((f1 - f2)**2) / N).item()

def extract_mfcc(waveform, sample_rate=8000):
    mfcc_tr = torchaudio.transforms.MFCC(
        sample_rate=sample_rate, n_mfcc=13,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
    ).to(waveform.device)
    return mfcc_tr(apply_vad_and_align(waveform)).contiguous().view(-1)

def extract_dwt(waveform, wavelet='db4', level=4):
    wave_np = apply_vad_and_align(waveform).squeeze(0).cpu().numpy()
    coeffs = pywt.wavedec(wave_np, wavelet, level=level)
    features = np.concatenate([[np.mean(c), np.std(c), np.sum(c**2)] for c in coeffs])
    return torch.tensor(features / (np.linalg.norm(features) + 1e-9), dtype=torch.float32)

def get_or_build_database(model, dataloader, device):
    if os.path.exists(DB_CACHE_PATH):
        print(f"Loaded {DB_CACHE_PATH}")
        return torch.load(DB_CACHE_PATH, weights_only=False)
        
    db = {'snn':[], 'spikes':[], 'mfcc':[], 'dwt':[], 'raw_c':[], 'gold_vector':[], 'labels':[], 'filenames':[]}
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
                db['raw_c'].append(apply_vad_and_align(w).cpu())
                db['labels'].append(labels[i].item())
                db['filenames'].append(filenames[i])
                
    for k in['snn', 'spikes', 'mfcc', 'dwt', 'raw_c', 'gold_vector']: db[k] = torch.cat(db[k], dim=0)
    db['labels'] = torch.tensor(db['labels']); db['filenames'] = np.array(db['filenames'])
    torch.save(db, DB_CACHE_PATH)
    return db


def run_advanced_xai_benchmark(db, k_values=[1, 3, 5]):
    methods =["SNN Siamese", "Van Rossum", "MFCC", "DWT", "Pearson"]
    
    results = {k: {m: {'f_rmse':[], 'cf_rmse': [], 'f_diversity': [], 'cf_diversity':[], 'time':[]} for m in methods} for k in k_values}
    

    for i in tqdm(range(len(db['filenames'])), desc="Processing Queries"):
        q_label = db['labels'][i].item()
        q_fname = db['filenames'][i]
        q_gold = db['gold_vector'][i].unsqueeze(0)
        q_spk = db['spikes'][i]
        
        mask_self = torch.from_numpy(db['filenames'] != q_fname)
        db_labels_f = db['labels'][mask_self]
        db_gold_f = db['gold_vector'][mask_self]

        for m in methods:
            t0 = time.perf_counter()
            
            if m == "SNN Siamese":
                dists = torch.cdist(db['snn'][i:i+1], db['snn'][mask_self], p=2).squeeze(0)
            elif m == "MFCC":
                dists = torch.cdist(db['mfcc'][i:i+1], db['mfcc'][mask_self], p=2).squeeze(0)
            elif m == "DWT":
                dists = torch.cdist(db['dwt'][i:i+1], db['dwt'][mask_self], p=2).squeeze(0)
            elif m == "Pearson":
                dists = 1 - F.cosine_similarity(db['raw_c'][i:i+1].to(device), db['raw_c'][mask_self].to(device)).cpu()
            elif m == "Van Rossum":
                d_base = torch.cdist(db['snn'][i:i+1], db['snn'][mask_self], p=2).squeeze(0)
                top50_idx = torch.topk(d_base, 50, largest=False)[1]
                vr_dists_local = torch.tensor([van_rossum_distance(q_spk, db['spikes'][mask_self][idx]) for idx in top50_idx])
                dists = torch.full_like(d_base, float('inf'))
                dists[top50_idx] = vr_dists_local

            t1 = time.perf_counter()
            calc_time = (t1 - t0) * 1000 

            for k in k_values:
                results[k][m]['time'].append(calc_time)
                
                factual_rmse_val = 0
                factual_diversity_val = 0
                cf_rmses_list = []
                cf_gold_vectors =[]

                for class_id in range(10):
                    mask_class = (db_labels_f == class_id)
                    if not mask_class.any(): continue
                    
                    dists_class = dists[mask_class]
                    actual_k = min(k, len(dists_class))
                    topk_class_idx = torch.topk(dists_class, actual_k, largest=False)[1]
                    
                    neighbors_gold = db_gold_f[mask_class][topk_class_idx]
                    rmse = torch.sqrt(torch.mean((q_gold - neighbors_gold)**2, dim=1)).mean().item()
                    
                    if class_id == q_label:
                        factual_rmse_val = rmse
                        if len(neighbors_gold) > 1:
                            factual_diversity_val = torch.pdist(neighbors_gold, p=2).mean().item()
                        else:
                            factual_diversity_val = 0.0
                    else:
                        cf_rmses_list.append(rmse)
                        top1_enemy_gold = db_gold_f[mask_class][torch.argmin(dists_class)].unsqueeze(0)
                        cf_gold_vectors.append(top1_enemy_gold)
                
                results[k][m]['f_rmse'].append(factual_rmse_val)
                results[k][m]['f_diversity'].append(factual_diversity_val)
                results[k][m]['cf_rmse'].append(np.mean(cf_rmses_list))
                
                if len(cf_gold_vectors) > 1:
                    cf_tensor = torch.cat(cf_gold_vectors, dim=0)
                    diversity = torch.pdist(cf_tensor, p=2).mean().item()
                    results[k][m]['cf_diversity'].append(diversity)

    for k in k_values:
        print(f"\nResults K={k}:")
        print(f"{'Method':<15} | {'T/Q(ms)':<8} | {'F.RMSE':<10} | {'CF.RMSE':<10} | {'F.Div':<9} | {'CF.Div':<9}")
        print("-" * 90)
        for m in methods:
            t = np.mean(results[k][m]['time'])
            f = np.mean(results[k][m]['f_rmse'])
            cf = np.mean(results[k][m]['cf_rmse'])
            f_div = np.mean(results[k][m]['f_diversity'])
            cf_div = np.mean(results[k][m]['cf_diversity'])
            
            print(f"{m:<15} | {t:>8.2f} | {f:>10.5f} | {cf:>10.5f} | {f_div:>9.5f} | {cf_div:>9.5f}")

if __name__ == "__main__":
    model = SiameseSNN(BACKBONE_PATH).to(device)
    ckpt = torch.load(SIAMESE_SAVE_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state'] if 'model_state' in ckpt else ckpt)

    test_ds = AudioMNISTEvalDataset(f"{DATA_ROOT}/test")
    test_loader = DataLoader(
        test_ds,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )    
    db = get_or_build_database(model, test_loader, device)

    run_advanced_xai_benchmark(db, k_values=[1, 3, 5])