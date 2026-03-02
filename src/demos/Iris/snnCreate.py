# 
# Basado en el proyecto de: 
# Kastger. (s.f.). spiking-neural-network. GitHub. 
# Recuperado el 26 de febrero de 2025, de https://github.com/kastger/spiking-neural-network
# 


import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate, utils
import snntorch.functional as SF
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# --- 1. POPULATION ENCODER ---
def population_encode(batch, bins=15):
    """ Converts 4 floats into 60 binary spikes (15 per feature) """
    batch_size, num_features = batch.shape
    encoded = torch.zeros(batch_size, num_features * bins)
    for i in range(num_features):
        # Map value 0.0-1.0 to bin 0-14
        vals = (batch[:, i] * (bins - 1)).long()
        for j, v in enumerate(vals):
            encoded[j, i * bins + v] = 1.0
    return encoded

# --- 2. DATA PREPARATION ---
iris = load_iris()
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(iris.data)
X_train, X_test, y_train, y_test = train_test_split(
    torch.tensor(X_scaled).float(), 
    torch.tensor(iris.target).long(), 
    test_size=0.2, random_state=42
)

# --- 3. MODEL DEFINITION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
spike_grad = surrogate.atan()

class PopNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 60 inputs (4 features * 15 bins) -> 64 hidden neurons
        self.fc1 = nn.Linear(60, 64)
        self.lif1 = snn.Leaky(beta=0.8, spike_grad=spike_grad)
        self.fc2 = nn.Linear(64, 3)
        self.lif2 = snn.Leaky(beta=0.8, spike_grad=spike_grad)

    def forward(self, x):
        # Convert float input to spikes once per sample
        x_pop = population_encode(x).to(device)
        mem1, mem2 = self.lif1.init_leaky(), self.lif2.init_leaky()
        spk2_rec, spk1_rec = [], []
        
        for step in range(25): # 25 steps is enough for this encoding
            cur1 = self.fc1(x_pop)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)
        return torch.stack(spk2_rec), torch.stack(spk1_rec), x_pop

net = PopNet().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)
loss_fn = SF.ce_rate_loss()

# --- 4. TRAINING LOOP ---
print("Training started...")
for epoch in range(601):
    spk_out, spk_hid, _ = net(X_train.to(device))
    
    # Loss = Classification error + Sparsity penalty (keeps network clean)
    loss_val = loss_fn(spk_out, y_train.to(device))
    loss_val += 0.05 * spk_hid.mean() 
    
    optimizer.zero_grad()
    loss_val.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        _, idx = spk_out.sum(dim=0).max(1)
        acc = (idx == y_train.to(device)).float().mean()
        print(f"Epoch {epoch:4d} | Loss: {loss_val.item():.4f} | Acc: {acc:.2f}")

# --- 5. EXPERT IDENTIFICATION ---
# Map hidden neurons to the feature block they listen to most
weights = net.fc1.weight.data.abs().cpu() 
feature_experts = {i: [] for i in range(4)}
for n_idx in range(64):
    # Sum weights from each 15-bin block
    sums = [weights[n_idx, i*15 : (i+1)*15].sum() for i in range(4)]
    feature_experts[np.argmax(sums)].append(n_idx)

# --- 6. SAVE ---
torch.save({
    'model_state': net.state_dict(),
    'scaler': scaler,
    'feature_experts': feature_experts,
    'test_data': (X_test, y_test),
    'target_names': iris.target_names
}, "./src/demos/Iris/snn_pop_model.pth")
print("\nSuccess! Model saved.")