import random
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import  DataLoader
import torchaudio
import snntorch as snn
import pywt 

from classes import SiameseSNN, AudioMNISTEvalDataset

# ==========================================
# 1. GLOBAL SNN PARAMETERS
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
beta = 0.9
v_threshold = 1.0
num_steps = 50
spike_grad = snn.surrogate.fast_sigmoid(slope=15)
DATA_ROOT = "AudioMNIST_split"
BACKBONE_PATH = "src/demos/siameseSNN/models/snn_pop_audio_explainable.pth" 
SIAMESE_SAVE_PATH = "./src/demos/siameseSNN/models/snn_siamese_model.pt"

# ==========================================
# 3. BASELINE FEATURE EXTRACTORS
# ==========================================
def extract_mfcc(waveform, sample_rate=8000):
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate, n_mfcc=13,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
    )
    mfcc = mfcc_transform(waveform)
    return mfcc.mean(dim=2).squeeze(0) 

def extract_dwt(waveform, wavelet='db4'):
    wave_np = waveform.squeeze(0).numpy()
    cA, cD = pywt.dwt(wave_np, wavelet)
    return torch.tensor(np.concatenate((cA, cD)), dtype=torch.float32)

def extract_raw(waveform):
    return waveform.squeeze(0)

# ==========================================
# 4. UNIFIED DATABASE BUILDER
# ==========================================
def build_all_databases(model, dataloader, device):
    db = {
        'snn': [], 'mfcc':[], 'dwt':[], 'pearson': [],
        'labels': [], 'filenames':[]
    }
    
    print("Building unified database for all methods... Please wait.")
    model.eval()
    with torch.no_grad():
        for waveforms, labels, filenames in dataloader:
            waveforms_gpu = waveforms.to(device)
            snn_embs, _ = model.get_embedding(waveforms_gpu)
            db['snn'].append(snn_embs.cpu())
            
            for i in range(waveforms.size(0)):
                wave = waveforms[i]
                db['mfcc'].append(extract_mfcc(wave).unsqueeze(0))
                db['dwt'].append(extract_dwt(wave).unsqueeze(0))
                db['pearson'].append(extract_raw(wave).unsqueeze(0))
                
                db['labels'].append(labels[i].item())
                db['filenames'].append(filenames[i])

    db['snn'] = torch.cat(db['snn'], dim=0)
    db['mfcc'] = torch.cat(db['mfcc'], dim=0)
    db['dwt'] = torch.cat(db['dwt'], dim=0)
    db['pearson'] = torch.cat(db['pearson'], dim=0)
    
    print(f"Database built! Total samples: {len(db['labels'])}")
    return db

# ==========================================
# 5. UNIFIED K-NN RETRIEVAL & RMSE
# ==========================================
def evaluate_query(query_waveform, query_label, query_filename, model, db, k=5, device='cuda', verbose=True):
    if verbose:
        print(f"\n{'='*50}")
        print(f"QUERY: {query_filename} (True Label = {query_label})")
        print(f"{'='*50}")
    
    if query_waveform.dim() == 2:
        query_waveform = query_waveform.unsqueeze(0)
        
    model.eval()
    with torch.no_grad():
        q_snn, _ = model.get_embedding(query_waveform.to(device))
        q_snn = q_snn.cpu()
        
    q_mfcc = extract_mfcc(query_waveform.squeeze(0)).unsqueeze(0)
    q_dwt = extract_dwt(query_waveform.squeeze(0)).unsqueeze(0)
    q_pearson = extract_raw(query_waveform.squeeze(0)).unsqueeze(0)

    methods = {
        "SNN Siamese": (q_snn, db['snn'], "euclidean"),
        "MFCC": (q_mfcc, db['mfcc'], "cosine"),
        "DWT": (q_dwt, db['dwt'], "euclidean"),
        "Pearson": (q_pearson, db['pearson'], "pearson")
    }

    results_summary = {}

    for method_name, (q_feat, db_feats, dist_type) in methods.items():
        if dist_type == "euclidean":
            distances = torch.cdist(q_feat, db_feats, p=2).squeeze(0)
        elif dist_type == "cosine":
            cos_sim = F.cosine_similarity(q_feat, db_feats)
            distances = 1 - cos_sim
        elif dist_type == "pearson":
            q_centered = q_feat - q_feat.mean()
            db_centered = db_feats - db_feats.mean(dim=1, keepdim=True)
            pearson_corr = F.cosine_similarity(q_centered, db_centered)
            distances = 1 - pearson_corr

        # Get Top K+1 (to account for the query itself being in the database)
        topk_distances, topk_indices = torch.topk(distances, k + 1, largest=False)
        
        retrieved_labels =[]
        if verbose:
            print(f"\n--- {method_name} ---")
            
        valid_count = 0
        for i in range(k + 1):
            idx = topk_indices[i].item()
            fname = db['filenames'][idx]
            
            # EXCLUDE THE QUERY ITSELF
            if fname == query_filename:
                continue
                
            lbl = db['labels'][idx]
            dist = topk_distances[i].item()
            retrieved_labels.append(lbl)
            
            if verbose:
                print(f"  Rank {valid_count+1}: Label = {lbl} | Dist = {dist:.4f} | File = {fname}")
                
            valid_count += 1
            if valid_count == k:
                break

        # Calculate Label RMSE
        squared_errors =[(lbl - query_label) ** 2 for lbl in retrieved_labels]
        rmse = math.sqrt(sum(squared_errors) / k)
        results_summary[method_name] = rmse
        
        if verbose:
            print(f"  >> Label RMSE: {rmse:.4f}")

    return results_summary

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Initialize Model
    model = SiameseSNN(backbone_path=BACKBONE_PATH).to(device)
    model.load_state_dict(torch.load(SIAMESE_SAVE_PATH, map_location=device))

    # 2. Load Dataset
    dataset = AudioMNISTEvalDataset(split_dir=DATA_ROOT+"/test", sample_rate=8000)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 3. Build Unified Database
    db = build_all_databases(model, dataloader, device)

    # 4. Evaluate 100 Random Samples
    N_EVALS = 100
    # Ensure we don't try to sample more than the dataset size
    num_samples = min(N_EVALS, len(dataset))
    
    print(f"\nEvaluating {num_samples} random queries to calculate average RMSE...")
    
    # Randomly select indices without replacement
    query_indices = random.sample(range(len(dataset)), num_samples)
    
    # Dictionary to keep track of cumulative RMSEs
    avg_rmses = {
        "SNN Siamese": 0.0,
        "MFCC": 0.0,
        "DWT": 0.0,
        "Pearson": 0.0
    }

    for i, query_idx in enumerate(query_indices):
        query_waveform, query_label, query_filename = dataset[query_idx]
        
        # Print a progress update every 10 samples
        if (i + 1) % 10 == 0:
            print(f"Processing query {i + 1}/{num_samples}...")
            
        # Run evaluation (verbose=False to avoid terminal spam)
        rmses = evaluate_query(
            query_waveform=query_waveform,
            query_label=query_label,
            query_filename=query_filename,
            model=model,
            db=db,
            k=5,
            device=device,
            verbose=False
        )
        
        # Accumulate the RMSEs
        for method in avg_rmses:
            avg_rmses[method] += rmses[method]

    # 5. Calculate and Print Final Averages
    for method in avg_rmses:
        avg_rmses[method] /= num_samples

    print(f"\n{'-'*50}")
    print(f"FINAL AVERAGE RMSE COMPARISON ({num_samples} Queries)")
    print(f"{'-'*50}")
    
    # Sort methods by lowest average RMSE
    sorted_methods = sorted(avg_rmses.items(), key=lambda x: x[1])
    for rank, (method, rmse) in enumerate(sorted_methods):
        print(f"{rank+1}. {method}: {rmse:.4f}")