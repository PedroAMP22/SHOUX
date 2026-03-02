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
from torch.utils.data import DataLoader
import os
import snntorch.functional as SF
from tqdm import tqdm
from classes import AudioMNISTSplitDataset, PopNetAudio

# --- CONFIGURACIÓN ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = "AudioMNIST_split"
save_path = "./src/demos/siameseSNN/models"
os.makedirs(save_path, exist_ok=True)

batch_size = 64
num_epochs = 15

# --- 1. DATA LOADERS ---
train_loader = DataLoader(AudioMNISTSplitDataset(f"{data_root}/train"), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(AudioMNISTSplitDataset(f"{data_root}/val"), batch_size=batch_size, shuffle=False)

model = PopNetAudio().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = SF.ce_rate_loss()

# --- 2. ENTRENAMIENTO DEL BACKBONE ---
print("Entrenando Backbone SNN con regularización biológica (Sparsity)...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    # Progreso del entrenamiento
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    for batch_idx, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device)
        
        # Forward pass (SNNTorch devuelve típicamente:[tiempo, batch, neuronas])
        spk_out, spk_hid = model(data)
        
        # 1. Loss de Clasificación (Cross Entropy sobre la tasa de disparo)
        loss_cls = loss_fn(spk_out, targets)

        # 2. Penalización L1 sobre los pesos (Promueve expertos puros)
        l1_weight_loss = 0.001 * torch.norm(model.fc2.weight, p=1)

        # 3. Penalización de actividad (Sparsity)
        # Promedia sobre el tiempo (dim 0) y el batch (dim 1)
        reg_loss = 0.01 * torch.norm(spk_hid.mean(dim=(0, 1)), p=1) 
        
        # Loss total
        loss_val = loss_cls + l1_weight_loss + reg_loss
        
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        
        total_loss += loss_val.item()

        # Actualizar barra de progreso
        pbar.set_postfix({'Loss': f"{loss_val.item():.4f}", 'Sparsity Penalty': f"{reg_loss.item():.4f}"})

    # --- 3. VALIDACIÓN (OBLIGATORIO PARA EL PAPER) ---
    model.eval()
    correct = 0
    total = 0
    total_spikes = 0
    total_neurons_time = 0
    
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            spk_out, spk_hid = model(data)
            
            # Precisión (Accuracy): Sumar spikes en el tiempo y elegir la clase con más disparos
            _, predicted = spk_out.sum(dim=0).max(1) 
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Calcular Firing Rate (Sparsity real) para el paper
            total_spikes += spk_hid.sum().item()
            total_neurons_time += spk_hid.numel() # Total de oportunidades de disparo
            
    val_acc = 100 * correct / total
    firing_rate = 100 * total_spikes / total_neurons_time
    
    print(f" -> Val Accuracy: {val_acc:.2f}% | Firing Rate Oculta: {firing_rate:.2f}% (Sparsity: {100-firing_rate:.2f}%)\n")

# --- 4. IDENTIFICACIÓN DE EXPERTOS (MÉTODO IRIS / XAI) ---
print("Mapeando expertos de clase...")
weights = model.fc2.weight.data.cpu() # Forma:[10 clases, 256 neuronas]
class_experts = {i:[] for i in range(10)}

for n_idx in range(weights.shape[1]):
    best_class = torch.argmax(weights[:, n_idx]).item()
    class_experts[best_class].append(n_idx)

# Guardar modelo y mapa de expertos
torch.save({
    'model_state': model.state_dict(),
    'class_experts': class_experts
}, os.path.join(save_path, "snn_pop_audio.pth"))

print("Entrenamiento del Backbone SNN completado con éxito.")