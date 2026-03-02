import os
import shutil
import random
from tqdm import tqdm

src_root = "AudioMNIST/data"
dst_root = "AudioMNIST_split"
train_perc, val_perc, test_perc = 0.8, 0.1, 0.1

all_data = []
for subject in os.listdir(src_root):
    subj_path = os.path.join(src_root, subject)
    if os.path.isdir(subj_path):
        for file in os.listdir(subj_path):
            if file.endswith(".wav"):
                label = file.split("_")[0] 
                all_data.append((os.path.join(subj_path, file), label, file))

random.seed(42) 
random.shuffle(all_data)

total = len(all_data)
tr_end = int(total * train_perc)
vl_end = tr_end + int(total * val_perc)

splits = {
    'train': all_data[:tr_end],
    'val': all_data[tr_end:vl_end],
    'test': all_data[vl_end:]
}

print(f"Dividiendo {total} archivos...")
for split_name, files in splits.items():
    for src_path, label, filename in tqdm(files, desc=f"Procesando {split_name}"):
        # Estructura: AudioMNIST_split/train/0/archivo.wav
        dest_dir = os.path.join(dst_root, split_name, label)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(src_path, os.path.join(dest_dir, filename))

