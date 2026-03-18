import torch
import torch.nn as nn
import snntorch as snn
from snntorch import utils
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

        self.temporal_filter = nn.Conv1d(
            in_channels=num_neurons_hid, 
            out_channels=num_neurons_hid, 
            kernel_size=25,
            padding=12, 
            groups=num_neurons_hid 
        ).to(device)
        
        self.avg_pool = nn.AdaptiveAvgPool1d(16)
        self.max_pool = nn.AdaptiveMaxPool1d(16)

        input_dim = num_neurons_hid * 32 
        
        self.fc_siamese = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(1024, 512) 
        ).to(device)
    
    def get_embedding(self, x):
        self.backbone.eval()
        with torch.no_grad():
            _, spk_hid_rec = self.backbone(x) 
        
        x_temp = spk_hid_rec.permute(1, 2, 0) 
        x_temp = F.leaky_relu(self.temporal_filter(x_temp))
        
        x_avg = self.avg_pool(x_temp)
        x_max = self.max_pool(x_temp)
        combined_features = torch.cat([x_avg, x_max], dim=1).reshape(x_temp.size(0), -1)
        
        out = self.fc_siamese(combined_features)
        embedding = F.normalize(out, p=2, dim=1)
        return embedding, spk_hid_rec.sum(dim=0)

    def forward(self, x1, x2):
        emb1, spk1 = self.get_embedding(x1)
        emb2, spk2 = self.get_embedding(x2)
        return emb1, emb2, spk1, spk2

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
v_threshold = 1.5
num_steps = 50
spike_grad = snn.surrogate.fast_sigmoid(slope=15) 

class PopNetAudio(nn.Module):
    def __init__(self, num_neurons_hid=250, num_classes=10): 
        super().__init__()
        
        self.num_classes = num_classes
        self.num_neurons_hid = num_neurons_hid
        self.neurons_per_class = num_neurons_hid // num_classes
        
        # Characteristic convolutional layers to extract features from the raw audio
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



        self.dropout = nn.Dropout(0.4)
        # Experts layer, 250 in total, 25 per class
        self.fc1 = nn.Linear(15776, num_neurons_hid)
        self.lif_hid = snn.Leaky(beta=beta, threshold=v_threshold, spike_grad=spike_grad, init_hidden=True, output=True)
        
        # mask for loss function, to ensure that only the spikes of the expert neurons of the correct class contribute to the loss
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
            
            # experts
            cur_hid = self.fc1(x_feat)
            spk_hid, _ = self.lif_hid(cur_hid)
            spk_hid_rec.append(spk_hid)
            
        # [num_steps, batch_size, num_neurons_hid]
        spk_hid_rec = torch.stack(spk_hid_rec)
        
        # xAi in architecture - we group the spikes of the hidden layer into 10 groups of 25 neurons (each group is an "expert" for a class)
        spk_grouped = spk_hid_rec.view(num_steps, x.size(0), self.num_classes, self.neurons_per_class)
        
        # We sum the spikes of each group to get the output spikes for each class, this is what will be used for the loss and the embeddings
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