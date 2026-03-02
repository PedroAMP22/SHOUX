import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate, utils
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset
import os
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_steps = 50
beta = 0.9
spike_grad = surrogate.atan()
beta = 0.9
v_threshold = 1.2
sparsity_coeff = 0.05 
spike_grad = surrogate.atan() 


# siamese class
class SiameseSNN(nn.Module):
    def __init__(self, backbone_path):
        super().__init__()
        checkpoint = torch.load(backbone_path, map_location=device)
        self.backbone = PopNetAudio().to(device)
        self.backbone.load_state_dict(checkpoint['model_state'])
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.fc_siamese = nn.Linear(256, 128)

    def get_embedding(self, x):
        self.backbone.eval() 
        # Obtenemos los spikes de la capa oculta
        _, spk_hid = self.backbone(x) 
        
        # pathway_activity: suma de spikes en el tiempo [batch, 256]
        # Esto es lo que el script de test necesita para la explicabilidad
        pathway_activity = spk_hid.sum(dim=0) 
        
        # Generamos el embedding normalizado
        out = self.fc_siamese(pathway_activity)
        embedding = F.normalize(out, p=2, dim=1)
        
        # CAMBIO CLAVE: Devolvemos AMBOS valores
        return embedding, pathway_activity

    def forward(self, x1, x2):
        # Durante el entrenamiento solo nos interesan los embeddings
        emb1, _ = self.get_embedding(x1)
        emb2, _ = self.get_embedding(x2)
        return emb1, emb2

# dataset for siamese
class SiameseAudioMNIST(Dataset):
    def __init__(self, split_dir, sample_rate=8000, duration=1.0):
        self.split_dir = split_dir
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.file_list = []
        self.label_to_indices = {i: [] for i in range(10)}
        
        current_idx = 0
        for label_dir in sorted(os.listdir(split_dir)):
            label_path = os.path.join(split_dir, label_dir)
            if os.path.isdir(label_path):
                for file in os.listdir(label_path):
                    if file.endswith(".wav"):
                        self.file_list.append((os.path.join(label_path, file), int(label_dir), file))
                        self.label_to_indices[int(label_dir)].append(current_idx)
                        current_idx += 1

    def __len__(self): return len(self.file_list)

    def _get_audio(self, idx):
        path, label, _ = self.file_list[idx]
        waveform, sr = torchaudio.load(path, backend="soundfile")
        waveform = waveform - waveform.mean()
        waveform = waveform / (waveform.abs().max() + 1e-8)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        if waveform.size(1) > self.num_samples:
            waveform = waveform[:, :self.num_samples]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.num_samples - waveform.size(1)))
        return waveform, label

    def __getitem__(self, idx):
        x1, label1 = self._get_audio(idx)
        is_same = random.randint(0, 1)
        if is_same:
            idx2 = random.choice(self.label_to_indices[label1])
            x2, _ = self._get_audio(idx2)
        else:
            label2 = random.choice([l for l in range(10) if l != label1])
            idx2 = random.choice(self.label_to_indices[label2])
            x2, _ = self._get_audio(idx2)
        return x1, x2, torch.tensor(is_same, dtype=torch.float32)
     
 
class AudioMNISTEvalDataset(Dataset):
    def __init__(self, split_dir, sample_rate=8000, duration=1.0):
        self.split_dir = split_dir
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.file_list = []
        for label_dir in sorted(os.listdir(split_dir)):
            label_path = os.path.join(split_dir, label_dir)
            if os.path.isdir(label_path):
                for file in os.listdir(label_path):
                    if file.endswith(".wav"):
                        self.file_list.append((os.path.join(label_path, file), int(label_dir), file))

    def __len__(self): return len(self.file_list)

    def __getitem__(self, idx):
        path, label, filename = self.file_list[idx]
        waveform, sr = torchaudio.load(path, backend="soundfile")
        
        # Normalización biológica
        waveform = waveform - waveform.mean()
        waveform = waveform / (waveform.abs().max() + 1e-8)

        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        if waveform.size(1) > self.num_samples:
            waveform = waveform[:, :self.num_samples]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.num_samples - waveform.size(1)))
        return waveform, label, filename
    

class PopNetAudio(nn.Module):
    def __init__(self):
        super().__init__()
        # Extractor de características (Equivalente al encoding)
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=80, stride=4),
            nn.BatchNorm1d(16),
            snn.Leaky(beta=beta, threshold=v_threshold, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            snn.Leaky(beta=beta, threshold=v_threshold, spike_grad=spike_grad, init_hidden=True),
            nn.Flatten()
        )

        self.dropout = nn.Dropout(0.5)

        # Capa de Población Oculta (256 neuronas)
        self.fc1 = nn.Linear(15776, 256)
        self.lif_hid = snn.Leaky(beta=beta, threshold=v_threshold, spike_grad=spike_grad, init_hidden=True)
        
        # Capa de Salida (10 clases)
        self.fc2 = nn.Linear(256, 10)
        self.lif_out = snn.Leaky(beta=beta, threshold=v_threshold, spike_grad=spike_grad, init_hidden=True, output=True)

    def forward(self, x):
        utils.reset(self.conv_block); utils.reset(self.lif_hid); utils.reset(self.lif_out)
        spk_hid_rec, spk_out_rec = [], []
        
        for step in range(num_steps):
            x_feat = self.conv_block(x * 3.0)
            
            x_feat = self.dropout(x_feat)
            
            spk_hid = self.lif_hid(self.fc1(x_feat))
            
            spk_out, _ = self.lif_out(self.fc2(spk_hid))
            
            spk_hid_rec.append(spk_hid)
            spk_out_rec.append(spk_out)
            
        return torch.stack(spk_out_rec), torch.stack(spk_hid_rec)
    
class AudioMNISTSplitDataset(Dataset):
    def __init__(self, split_dir, sample_rate=8000, duration=1.0):
        self.split_dir = split_dir
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.file_list = []
        for label_dir in sorted(os.listdir(split_dir)):
            label_path = os.path.join(split_dir, label_dir)
            if os.path.isdir(label_path):
                for file in os.listdir(label_path):
                    if file.endswith(".wav"):
                        self.file_list.append((os.path.join(label_path, file), int(label_dir)))
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        waveform, sr = torchaudio.load(path, backend="soundfile")
        waveform = waveform - waveform.mean()
        waveform = waveform / (waveform.abs().max() + 1e-8)
        if sr != self.sample_rate: waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        if waveform.size(1) > self.num_samples: waveform = waveform[:, :self.num_samples]
        else: waveform = torch.nn.functional.pad(waveform, (0, self.num_samples - waveform.size(1)))
        return waveform, label