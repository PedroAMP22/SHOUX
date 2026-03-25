import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

DATA_ROOT = "AudioMNIST_split"
DB_CACHE_PATH = "db_cache.pt"
EXAMPLE_DIR = "example" 
K_EXAMPLES = 3

if not os.path.exists(DB_CACHE_PATH):
    raise FileNotFoundError(f"Not found {DB_CACHE_PATH}.")

os.makedirs(EXAMPLE_DIR, exist_ok=True)

db = torch.load(DB_CACHE_PATH, weights_only=False)

np.random.seed(10) 
query_idx = np.random.randint(len(db['filenames']))

q_label = db['labels'][query_idx].item()
q_fname = db['filenames'][query_idx]
q_emb = db['snn'][query_idx]
q_wave = db['raw_c'][query_idx].squeeze().numpy()

print("\n" + "="*70)
print("="*70)
print(f"QUERY:")
print(f"   File: {q_fname}")
print(f"   Class {q_label}")
print("-" * 70)

dists = torch.cdist(q_emb.unsqueeze(0), db['snn'], p=2).squeeze(0)
mask_self = torch.arange(len(db['filenames'])) != query_idx

mask_factual = mask_self & (db['labels'] == q_label)
dists_f = dists[mask_factual]
indices_f = torch.where(mask_factual)[0]

sorted_f_idx = torch.argsort(dists_f)
top_f_indices = indices_f[sorted_f_idx[:K_EXAMPLES]]
top_f_dists = dists_f[sorted_f_idx[:K_EXAMPLES]]

for i in range(len(top_f_indices)):
    idx = top_f_indices[i]
    print(f"   {i+1}. File: {db['filenames'][idx]:<15} | Distance: {top_f_dists[i]:.4f}")

print("-" * 70)

best_enemies =[]
for class_id in range(10):
    if class_id == q_label: continue
    
    mask_class = mask_self & (db['labels'] == class_id)
    if not mask_class.any(): continue
    
    dists_class = dists[mask_class]
    indices_class = torch.where(mask_class)[0]
    
    min_idx_local = torch.argmin(dists_class)
    min_dist = dists_class[min_idx_local].item()
    global_idx = indices_class[min_idx_local].item()
    
    best_enemies.append((min_dist, class_id, db['filenames'][global_idx], global_idx))

best_enemies.sort(key=lambda x: x[0])
top_cf = best_enemies[:K_EXAMPLES]

for i, (dist_val, c_label, fname, _) in enumerate(top_cf):
    print(f"   {i+1}. Class: {c_label} | File: {fname:<15} | Distance: {dist_val:.4f}")

print("="*70)

best_f_idx = top_f_indices[0].item()
best_cf_idx = top_cf[0][3]

f_label = q_label
f_fname = db['filenames'][best_f_idx]
f_emb = db['snn'][best_f_idx]

cf_label = top_cf[0][1]
cf_fname = top_cf[0][2]
cf_emb = db['snn'][best_cf_idx]

src_query = os.path.join(DATA_ROOT, "test", str(q_label), q_fname)
src_factual = os.path.join(DATA_ROOT, "test", str(f_label), f_fname)
src_cf = os.path.join(DATA_ROOT, "test", str(cf_label), cf_fname)

dst_query = os.path.join(EXAMPLE_DIR, f"1_query_digit{q_label}_{q_fname}")
dst_factual = os.path.join(EXAMPLE_DIR, f"2_factual_digit{f_label}_{f_fname}")
dst_cf = os.path.join(EXAMPLE_DIR, f"3_counterfactual_digit{cf_label}_{cf_fname}")

try:
    shutil.copy2(src_query, dst_query)
    shutil.copy2(src_factual, dst_factual)
    shutil.copy2(src_cf, dst_cf)
    print(f"\nAudios in n '{EXAMPLE_DIR}'")
except Exception as e:
    print(f"\nERROR {e}")

latent_txt_path = os.path.join(EXAMPLE_DIR, "latent_factors.txt")
with open(latent_txt_path, "w") as f:
    f.write(f"=== FACTORES LATENTES (512 Dimensiones) ===\n\n")
    f.write(f"1. QUERY (Dígito {q_label} - {q_fname}):\n")
    f.write(np.array2string(q_emb.numpy(), separator=', ') + "\n\n")
    
    f.write(f"2. FACTUAL (Dígito {f_label} - {f_fname}):\n")
    f.write(np.array2string(f_emb.numpy(), separator=', ') + "\n\n")
    
    f.write(f"3. COUNTERFACTUAL (Dígito {cf_label} - {cf_fname}):\n")
    f.write(np.array2string(cf_emb.numpy(), separator=', ') + "\n")

wave_f = db['raw_c'][best_f_idx].squeeze().numpy()
wave_cf = db['raw_c'][best_cf_idx].squeeze().numpy()

plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True, sharey=True)
fig.suptitle('xAI Audio Retrieval: Factual vs Counterfactual Waveforms', fontsize=16, fontweight='bold', y=0.95)

ax1.plot(q_wave, color='#1f77b4', linewidth=1)
ax1.set_title(f'Query: Digit {q_label}[{q_fname}]', fontsize=12, fontweight='bold', color='#1f77b4')

ax2.plot(wave_f, color='#2ca02c', linewidth=1)
ax2.set_title(f'Factual Match: Digit {q_label} [{f_fname}] (Dist: {top_f_dists[0]:.4f})', fontsize=12, fontweight='bold', color='#2ca02c')

ax3.plot(wave_cf, color='#d62728', linewidth=1)
ax3.set_title(f'Nearest Counterfactual: Digit {cf_label} [{cf_fname}] (Dist: {top_cf[0][0]:.4f})', fontsize=12, fontweight='bold', color='#d62728')
ax3.set_xlabel('Time (Samples)')

plt.tight_layout()
plt.savefig(os.path.join(EXAMPLE_DIR, "qualitative_waveforms.png"), dpi=300, bbox_inches='tight')

import torchaudio.transforms as T

mel_transform = T.MelSpectrogram(
    sample_rate=8000,
    n_fft=400,
    hop_length=160,
    n_mels=64
)
db_transform = T.AmplitudeToDB(stype='power', top_db=80)

def get_mel_heatmap(wave_np):
    wave_t = torch.tensor(wave_np).unsqueeze(0).float() 
    mel = mel_transform(wave_t)
    mel_db = db_transform(mel).squeeze(0).numpy()
    return mel_db

mel_q = get_mel_heatmap(q_wave)
mel_f = get_mel_heatmap(wave_f)
mel_cf = get_mel_heatmap(wave_cf)

def get_active_frames(mel_db):
    col_max = mel_db.max(axis=0)
    active_cols = np.where(col_max > mel_db.min() + 1e-3)[0]
    return active_cols[-1] if len(active_cols) > 0 else mel_db.shape[1]

max_q = get_active_frames(mel_q)
max_f = get_active_frames(mel_f)
max_cf = get_active_frames(mel_cf)

max_frame = max(max_q, max_f, max_cf)
max_frame = min(mel_q.shape[1], max_frame + 5)

mel_q = mel_q[:, :max_frame]
mel_f = mel_f[:, :max_frame]
mel_cf = mel_cf[:, :max_frame]

fig3, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
fig3.suptitle('Acoustic Representation: Mel-Spectrogram Heatmaps', fontsize=24, fontweight='bold', y=1.05)

cmap_audio = 'magma'

im1 = ax1.imshow(mel_q, aspect='auto', origin='lower', cmap=cmap_audio)
ax1.set_title(f'Query (Digit {q_label})', fontsize=20, fontweight='bold')
ax1.set_ylabel('Mel Bins (Frequency)', fontsize=16)
ax1.set_xlabel('Time Frames', fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=14)

im2 = ax2.imshow(mel_f, aspect='auto', origin='lower', cmap=cmap_audio)
ax2.set_title(f'Factual Match (Digit {f_label})', fontsize=20, fontweight='bold', color='#2ca02c')
ax2.set_xlabel('Time Frames', fontsize=16)
ax2.tick_params(axis='both', which='major', labelsize=14)

im3 = ax3.imshow(mel_cf, aspect='auto', origin='lower', cmap=cmap_audio)
ax3.set_title(f'Nearest Counterfactual (Digit {cf_label})', fontsize=20, fontweight='bold', color='#d62728')
ax3.set_xlabel('Time Frames', fontsize=16)
ax3.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()

cbar = fig3.colorbar(im3, ax=[ax1, ax2, ax3], format='%+2.0f dB', location='right', shrink=0.85, pad=0.02)
cbar.set_label('Magnitude (dB)', fontsize=16)
cbar.ax.tick_params(labelsize=14)

mel_path = os.path.join(EXAMPLE_DIR, "mel_spectrograms_horizontal.png")
plt.savefig(mel_path, dpi=300, bbox_inches='tight')
plt.show()