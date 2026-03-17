"""
    Code to divide the original AudioMNIST dataset into train/val/test splits based on subjects,
    ensuring no subject appears in more than one split.

"""

import os
import shutil
import random
from tqdm import tqdm

src_root = "AudioMNIST/data"
dst_root = "AudioMNIST_split"


subjects =[d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
subjects.sort()

random.seed(42) 
random.shuffle(subjects)

train_subs = subjects[:48]
val_subs = subjects[48:54]
test_subs = subjects[54:]

print(f"Sujetos en Train: {len(train_subs)}")
print(f"Sujetos en Val: {len(val_subs)}")
print(f"Sujetos en Test: {len(test_subs)}")

if os.path.exists(dst_root):
    shutil.rmtree(dst_root)

def process_split(split_name, split_subjects):
    for subject in tqdm(split_subjects, desc=f"{split_name}"):
        subj_path = os.path.join(src_root, subject)
        for file in os.listdir(subj_path):
            if file.endswith(".wav"):
                label = file.split("_")[0] 
                
                src_path = os.path.join(subj_path, file)
                dest_dir = os.path.join(dst_root, split_name, label)
                
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy2(src_path, os.path.join(dest_dir, file))

process_split('train', train_subs)
process_split('val', val_subs)
process_split('test', test_subs)

