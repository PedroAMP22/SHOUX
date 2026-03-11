"""
We use the Audio MNist, 
Sören Becker, Johanna Vielhaben, Marcel Ackermann, Klaus-Robert Müller, Sebastian Lapuschkin, Wojciech Samek,
AudioMNIST: Exploring Explainable Artificial Intelligence for audio analysis on a simple benchmark,
Journal of the Franklin Institute,
Volume 361, Issue 1,
2024,
Pages 418-428,
ISSN 0016-0032,
https://doi.org/10.1016/j.jfranklin.2023.11.038.
(https://www.sciencedirect.com/science/article/pii/S0016003223007536)
Keywords: Deep learning; Neural networks; Interpretability; Explainable artificial intelligence; Audio classification; Speech recognition
"""
import torch
import os
import snntorch.functional as SF
from tqdm import tqdm
from classes import AudioMNISTSplitDataset, PopNetAudio
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = "AudioMNIST_split"
save_path = "./src/demos/siameseSNN/models"
os.makedirs(save_path, exist_ok=True)

batch_size = 64
num_epochs = 15

train_loader = DataLoader(AudioMNISTSplitDataset(f"{data_root}/train"), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(AudioMNISTSplitDataset(f"{data_root}/val"), batch_size=batch_size, shuffle=False)

model = PopNetAudio(num_neurons_hid=250, num_classes=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = SF.ce_rate_loss()

print(f"Starting training on {device}...")
for epoch in range(num_epochs):
    model.train()
    
    # penalizer
    ortho_coeff = 0.1  # intrusion penalty
    sparsity_coeff = 0.2 # sparsity regularization
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for data, targets in pbar:
        data, targets = data.to(device), targets.to(device).long()
        
        # reduce amplitude to avoid saturation
        spk_out, spk_hid = model(data * 3.0) 
        
        # CrossEntropy to make sure experts learn how to clasifiy
        loss_cls = loss_fn(spk_out, targets)

        # Add of all spikes
        total_spikes_hid = spk_hid.sum(dim=0) 
        
        # Masks for teh assignment of experts, each class has a group of 25 experts, the rest are intruders
        batch_masks = model.expert_masks[targets]
        intruder_masks = 1.0 - batch_masks
        
        # We penalize the itrusions of other experts
        intruder_spikes = total_spikes_hid * intruder_masks
        loss_ortho = ortho_coeff * intruder_spikes.mean()

        # Regularization, we want the exeperts to fire but not to sature the network
        expert_spikes = total_spikes_hid * batch_masks
        target_spikes_per_neuron = 7.5 
        avg_expert_spikes = expert_spikes.sum(dim=1) / model.neurons_per_class
        loss_sparsity = sparsity_coeff * torch.mean((avg_expert_spikes - target_spikes_per_neuron)**2)
        
        # Total loss
        loss_total = loss_cls + loss_ortho + loss_sparsity
        
        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        pbar.set_postfix({
            'Cls': f"{loss_cls.item():.2f}", 
            'Orth': f"{loss_ortho.item():.2f}",
            'Spar': f"{loss_sparsity.item():.2f}"
        })

    # Vlaidation
    model.eval()
    correct, total, total_spikes, total_neurons_time = 0, 0, 0, 0
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            spk_out, spk_hid = model(data * 3.0)
            
            # Predict: wich expert group has the most spikes?
            _, predicted = spk_out.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            total_spikes += spk_hid.sum().item()
            total_neurons_time += spk_hid.numel()
            
    val_acc = 100 * correct / total
    firing_rate = 100 * total_spikes / total_neurons_time
    print(f" -> Val Acc: {val_acc:.2f}% | Firing Rate: {firing_rate:.2f}%")

# Save
class_experts = {i: list(range(i*model.neurons_per_class, (i+1)*model.neurons_per_class)) for i in range(10)}
torch.save({
    'model_state': model.state_dict(),
    'class_experts': class_experts,
    'neurons_per_class': model.neurons_per_class
}, os.path.join(save_path, "snn_pop_audio_explainable.pth"))

print("\nEnding, saving...")