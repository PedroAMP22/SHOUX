import os
import shutil
import random
from tqdm import tqdm

src_root = "AudioMNIST/data"
dst_root = "AudioMNIST_split"

# 1. Obtener la lista de carpetas de sujetos (ej. '01', '02', ..., '60')
subjects =[d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]
subjects.sort() # Ordenamos para que sea reproducible

# Mezclamos los sujetos aleatoriamente
random.seed(42) 
random.shuffle(subjects)

# 2. Dividimos los SUJETOS (AudioMNIST tiene 60 sujetos)
# 48 sujetos para Train | 6 para Val | 6 para Test
train_subs = subjects[:48]
val_subs = subjects[48:54]
test_subs = subjects[54:]

print(f"Sujetos en Train: {len(train_subs)}")
print(f"Sujetos en Val: {len(val_subs)}")
print(f"Sujetos en Test: {len(test_subs)}")

# Limpiamos el directorio destino si ya existe para evitar mezclar datos viejos
if os.path.exists(dst_root):
    shutil.rmtree(dst_root)

# 3. Función para procesar y copiar manteniendo tu estructura de carpetas por dígitos
def process_split(split_name, split_subjects):
    for subject in tqdm(split_subjects, desc=f"Procesando {split_name}"):
        subj_path = os.path.join(src_root, subject)
        for file in os.listdir(subj_path):
            if file.endswith(".wav"):
                # El nombre del archivo es ej: "0_01_0.wav" -> el primer elemento es el label (dígito)
                label = file.split("_")[0] 
                
                src_path = os.path.join(subj_path, file)
                # Estructura final: AudioMNIST_split/train/0/archivo.wav
                dest_dir = os.path.join(dst_root, split_name, label)
                
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy2(src_path, os.path.join(dest_dir, file))

# 4. Ejecutar la copia
process_split('train', train_subs)
process_split('val', val_subs)
process_split('test', test_subs)

print("\n¡División por locutores completada con éxito!")