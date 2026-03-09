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

# siamese class
class SiameseSNN(nn.Module):
    def __init__(self, backbone_path):
        super().__init__()
        checkpoint = torch.load(backbone_path, map_location=device)
        state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
        
        num_neurons_hid = state_dict['fc1.weight'].shape[0]
        
        self.backbone = PopNetAudio(num_neurons_hid=num_neurons_hid, num_classes=10).to(device)
        
        self.backbone.load_state_dict(state_dict, strict=True)
            
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.backbone.eval()
        
        input_dim = self.backbone.num_neurons_hid * 2 
        self.fc_siamese = nn.Linear(input_dim, 128).to(device)
    
    def get_embedding(self, x):
        self.backbone.eval()
        
        # spk_hid_rec es [num_steps, batch, num_neurons_hid]
        _, spk_hid_rec = self.backbone(x) 
        
        # 1. Tasa de disparo (lo que ya tenías)
        firing_rate = spk_hid_rec.sum(dim=0) # [batch, 250]
        
        # 2. Latencia (Time-to-First-Spike)
        # Buscamos el índice del primer paso donde ocurre un spike
        # Si no hay spikes, ponemos num_steps (latencia máxima)
        first_spike_idx = torch.argmax(spk_hid_rec, dim=0).float()
        # Normalizamos entre 0 y 1
        latency = first_spike_idx / num_steps 
        
        # 3. Combinamos: [batch, 250 + 250] = [batch, 500]
        # Ahora el embedding tiene información de frecuencia Y de tiempo
        combined_features = torch.cat([firing_rate, latency], dim=1)
        
        # Proyección (Asegúrate de que fc_siamese acepte 500 entradas)
        out = self.fc_siamese(combined_features)
        
        embedding = F.normalize(out, p=2, dim=1)
        
        return embedding, spk_hid_rec.sum(dim=0) # Mantenemos la compatibilidad

    def forward(self, x1, x2):
        emb1, _ = self.get_embedding(x1)
        emb2, _ = self.get_embedding(x2)
        return emb1, emb2

    def forward_full(self, x1, x2):
        emb1, spk_hid1 = self.get_embedding(x1)
        emb2, spk_hid2 = self.get_embedding(x2)
        return emb1, emb2, spk_hid1, spk_hid2

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
    


beta = 0.9
v_threshold = 1.0
num_steps = 50
spike_grad = snn.surrogate.fast_sigmoid(slope=15) 

class PopNetAudio(nn.Module):
    def __init__(self, num_neurons_hid=250, num_classes=10): 
        super().__init__()
        
        self.num_classes = num_classes
        self.num_neurons_hid = num_neurons_hid
        self.neurons_per_class = num_neurons_hid // num_classes
        
        # Extractor de características
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=80, stride=4),
            nn.BatchNorm1d(16),
            snn.Leaky(beta=beta, threshold=v_threshold, spike_grad=spike_grad, init_hidden=True),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            snn.Leaky(beta=beta, threshold=v_threshold, spike_grad=spike_grad, init_hidden=True),
            nn.Flatten()
        ) # -> Lo transforma a Spikes



        self.dropout = nn.Dropout(0.4)
        # los Expertos
        # idealmente, si hay 250 neuronas, 25 neuronas por clase
        self.fc1 = nn.Linear(15776, num_neurons_hid)
        self.lif_hid = snn.Leaky(beta=beta, threshold=v_threshold, spike_grad=spike_grad, init_hidden=True, output=True)
        
        # Máscaras para la función de pérdida
        masks = torch.zeros(num_classes, num_neurons_hid)
        for i in range(num_classes):
            masks[i, i*self.neurons_per_class : (i+1)*self.neurons_per_class] = 1.0
        self.register_buffer('expert_masks', masks)

    def forward(self, x):
        utils.reset(self.conv_block); utils.reset(self.lif_hid)
        spk_hid_rec =[]
        
        for step in range(num_steps):
            x_feat = self.conv_block(x)
            x_feat = self.dropout(x_feat)
            
            # Capa de expertos
            cur_hid = self.fc1(x_feat)
            spk_hid, _ = self.lif_hid(cur_hid)
            spk_hid_rec.append(spk_hid)
            
        # Apilamos los spikes: [num_steps, batch_size, num_neurons_hid]
        spk_hid_rec = torch.stack(spk_hid_rec)
        
        # --- EXPLICABILIDAD POR DISEÑO (Direct Pooling) ---
        # Redimensionamos para agrupar las neuronas por clase: 
        #[num_steps, batch_size, num_classes, neurons_per_class]
        spk_grouped = spk_hid_rec.view(num_steps, x.size(0), self.num_classes, self.neurons_per_class)
        
        # Sumamos los spikes de las 25 neuronas de cada experto para obtener la "energía" de la clase
        # Resultado: [num_steps, batch_size, num_classes]
        spk_out_rec = spk_grouped.sum(dim=3) 
        
        return spk_out_rec, spk_hid_rec
    
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