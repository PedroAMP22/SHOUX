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

q_mat = q_emb.numpy().reshape(16, 32)
f_mat = f_emb.numpy().reshape(16, 32)
cf_mat = cf_emb.numpy().reshape(16, 32)

diff_f = np.abs(q_mat - f_mat)
diff_cf = np.abs(q_mat - cf_mat)

fig2 = plt.figure(figsize=(16, 8))
fig2.suptitle('Latent Space "Semantic Fingerprints" (512d reshaped to 16x32)', fontsize=18, fontweight='bold', y=0.98)

vmin_emb = min(q_mat.min(), f_mat.min(), cf_mat.min())
vmax_emb = max(q_mat.max(), f_mat.max(), cf_mat.max())

ax1 = plt.subplot(2, 3, 1)
im1 = ax1.imshow(q_mat, cmap='viridis', vmin=vmin_emb, vmax=vmax_emb, aspect='auto')
ax1.set_title(f'Query (Digit {q_label})', fontsize=14, fontweight='bold')
ax1.axis('off')

ax2 = plt.subplot(2, 3, 2)
im2 = ax2.imshow(f_mat, cmap='viridis', vmin=vmin_emb, vmax=vmax_emb, aspect='auto')
ax2.set_title(f'Factual Match (Digit {f_label})', fontsize=14, fontweight='bold')
ax2.axis('off')

ax3 = plt.subplot(2, 3, 3)
im3 = ax3.imshow(cf_mat, cmap='viridis', vmin=vmin_emb, vmax=vmax_emb, aspect='auto')
ax3.set_title(f'Counterfactual (Digit {cf_label})', fontsize=14, fontweight='bold')
ax3.axis('off')

cbar_ax1 = fig2.add_axes([0.92, 0.55, 0.015, 0.35])
fig2.colorbar(im3, cax=cbar_ax1, label='Activation Value')

vmax_diff = max(diff_f.max(), diff_cf.max())

ax5 = plt.subplot(2, 3, 5)
im5 = ax5.imshow(diff_f, cmap='Reds', vmin=0, vmax=vmax_diff, aspect='auto')
ax5.set_title(f'Error: Query vs Factual\n(Mean Error: {diff_f.mean():.4f})', fontsize=12, fontweight='bold', color='darkred')
ax5.axis('off')

ax6 = plt.subplot(2, 3, 6)
im6 = ax6.imshow(diff_cf, cmap='Reds', vmin=0, vmax=vmax_diff, aspect='auto')
ax6.set_title(f'Error: Query vs Counterfactual\n(Mean Error: {diff_cf.mean():.4f})', fontsize=12, fontweight='bold', color='darkred')
ax6.axis('off')

cbar_ax2 = fig2.add_axes([0.92, 0.1, 0.015, 0.35])
fig2.colorbar(im6, cax=cbar_ax2, label='Absolute Difference')

plt.subplots_adjust(wspace=0.1, hspace=0.3, right=0.9)
plt.savefig(os.path.join(EXAMPLE_DIR, "semantic_fingerprints.png"), dpi=300, bbox_inches='tight')
plt.show()

import torchaudio.transforms as T


mel_transform = T.MelSpectrogram(
    sample_rate=8000,
    n_fft=400,
    hop_length=160,
    n_mels=64
)
db_transform = T.AmplitudeToDB(stype='power', top_db=80)

def get_mel_heatmap(wave_np):
    wave_t = torch.tensor(wave_np).unsqueeze(0).float() # [1, T]
    mel = mel_transform(wave_t)
    mel_db = db_transform(mel).squeeze(0).numpy()
    return mel_db

mel_q = get_mel_heatmap(q_wave)
mel_f = get_mel_heatmap(wave_f)
mel_cf = get_mel_heatmap(wave_cf)

fig3, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, sharey=True)
fig3.suptitle('Acoustic Representation: Mel-Spectrogram Heatmaps', fontsize=18, fontweight='bold', y=0.96)

cmap_audio = 'magma'

im1 = ax1.imshow(mel_q, aspect='auto', origin='lower', cmap=cmap_audio)
ax1.set_title(f'Query (Digit {q_label})', fontsize=14, fontweight='bold', color='#333333')
ax1.set_ylabel('Mel Bins (Frequency)')

im2 = ax2.imshow(mel_f, aspect='auto', origin='lower', cmap=cmap_audio)
ax2.set_title(f'Factual Match (Digit {f_label})', fontsize=14, fontweight='bold', color='#2ca02c')
ax2.set_ylabel('Mel Bins (Frequency)')

im3 = ax3.imshow(mel_cf, aspect='auto', origin='lower', cmap=cmap_audio)
ax3.set_title(f'Nearest Counterfactual (Digit {cf_label})', fontsize=14, fontweight='bold', color='#d62728')
ax3.set_ylabel('Mel Bins (Frequency)')
ax3.set_xlabel('Time Frames')

fig3.subplots_adjust(right=0.85, hspace=0.3)
cbar_ax = fig3.add_axes([0.88, 0.15, 0.02, 0.7])
fig3.colorbar(im3, cax=cbar_ax, format='%+2.0f dB', label='Magnitude (dB)')

mel_path = os.path.join(EXAMPLE_DIR, "mel_spectrograms_heatmap.png")
plt.savefig(mel_path, dpi=300, bbox_inches='tight')
plt.show()