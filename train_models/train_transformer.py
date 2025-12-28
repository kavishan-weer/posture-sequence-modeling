import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import time
import math

# 1. Load Data
X = np.load('data/processed/X.npy') # (Batch, 20, 15)
y = np.load('data/processed/y.npy')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)

# 2. Positional Encoding (Crucial for Transformers)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# 3. Define Tiny Transformer
class TransformerBaseline(nn.Module):
    def __init__(self, input_dim=15, d_model=32, nhead=4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Tiny encoder: 1 layer is enough for this task
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=1)
        
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x) # Transform 15 sensors to 32 dimensions
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Use the mean of all timesteps as the final representation
        x = x.mean(dim=1) 
        return self.sigmoid(self.fc(x))

# 4. Training
model = TransformerBaseline()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("✨ Training Tiny Transformer...")
for epoch in range(50):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_y)
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")

# 5. Evaluation
model.eval()
with torch.no_grad():
    start_time = time.time()
    test_outputs = model(X_test)
    end_time = time.time()
    
    latency = (end_time - start_time) / len(X_test)
    predictions = (test_outputs > 0.5).float()
    accuracy = (predictions == y_test).float().mean()

print(f"\n✅ Transformer Accuracy: {accuracy.item()*100:.2f}%")
print(f"⏱️ Inference Latency: {latency*1000:.4f} ms per window")

torch.save(model.state_dict(), 'models/transformer_weights.pth')