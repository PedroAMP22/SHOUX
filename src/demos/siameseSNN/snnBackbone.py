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

print(f"Iniciando entrenamiento en {device}...")
for epoch in range(num_epochs):
    model.train()
    
    # Aumentamos drásticamente la penalización de ortogonalidad
    ortho_coeff = 0.1  # Antes era 0.001. Ahora sí dolerá equivocarse.
    sparsity_coeff = 0.05
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for data, targets in pbar:
        data, targets = data.to(device), targets.to(device).long()
        
        # Reducimos un poco la ganancia para evitar saturación masiva
        spk_out, spk_hid = model(data * 3.0) 
        
        # 1. Loss de Clasificación (CrossEntropy sobre la suma de los expertos)
        loss_cls = loss_fn(spk_out, targets)

        # Suma total de spikes en el tiempo:[batch_size, 250]
        total_spikes_hid = spk_hid.sum(dim=0) 
        
        # Máscaras
        batch_masks = model.expert_masks[targets]
        intruder_masks = 1.0 - batch_masks
        
        # 2. PENALIZACIÓN DE INTRUSOS (Ortogonalidad estricta)
        # Castigamos cualquier spike que ocurra fuera del grupo experto asignado
        intruder_spikes = total_spikes_hid * intruder_masks
        loss_ortho = ortho_coeff * intruder_spikes.mean()

        # 3. REGULARIZACIÓN DE SPARSITY (Evitar saturación)
        # Queremos que el experto correcto dispare, pero no al 100% (evitar los "50" fijos)
        # Un buen target es ~15% de firing rate (aprox 7.5 spikes por neurona en 50 steps)
        expert_spikes = total_spikes_hid * batch_masks
        target_spikes_per_neuron = 7.5 
        avg_expert_spikes = expert_spikes.sum(dim=1) / model.neurons_per_class
        loss_sparsity = sparsity_coeff * torch.mean((avg_expert_spikes - target_spikes_per_neuron)**2)
        
        # Loss Total
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

    # --- VALIDACIÓN ---
    model.eval()
    correct, total, total_spikes, total_neurons_time = 0, 0, 0, 0
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            spk_out, spk_hid = model(data * 3.0)
            
            # Predicción: ¿Qué grupo de expertos sumó más spikes en total?
            _, predicted = spk_out.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            total_spikes += spk_hid.sum().item()
            total_neurons_time += spk_hid.numel()
            
    val_acc = 100 * correct / total
    firing_rate = 100 * total_spikes / total_neurons_time
    print(f" -> Val Acc: {val_acc:.2f}% | Firing Rate: {firing_rate:.2f}%")

# Guardado
class_experts = {i: list(range(i*model.neurons_per_class, (i+1)*model.neurons_per_class)) for i in range(10)}
torch.save({
    'model_state': model.state_dict(),
    'class_experts': class_experts,
    'neurons_per_class': model.neurons_per_class
}, os.path.join(save_path, "snn_pop_audio_explainable.pth"))

print("\nEntrenamiento finalizado. Modelo 100% explicable guardado.")