import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import time
import os

# 1. Load the processed data
X = np.load('data/processed/X.npy')
y = np.load('data/processed/y.npy')

# PyTorch expects (Batch, Channels, Length) for 1D Conv
# Our data is (Batch, Length, Channels) -> we need to swap axes
X = np.transpose(X, (0, 2, 1)) 

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch Tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)

# 3. Define the CNN Architecture
class CNNBaseline(nn.Module):
    def __init__(self):
        super(CNNBaseline, self).__init__()
        # Conv1d(in_channels=15 sensors, out_channels=32 filters, kernel_size=3)
        self.conv1 = nn.Conv1d(15, 32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        # The linear layer size depends on your window size (20)
        # after Conv(3) and Pool(2), the length becomes 9. 32 * 9 = 288.
        self.fc1 = nn.Linear(32 * 9, 16)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# 4. Training Loop
model = CNNBaseline()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("🏋️ Training CNN Baseline...")
for epoch in range(50):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")

# 5. Evaluation & Latency Check
model.eval()
with torch.no_grad():
    start_time = time.time()
    test_outputs = model(X_test)
    end_time = time.time()
    
    latency = (end_time - start_time) / len(X_test)
    predictions = (test_outputs > 0.5).float()
    accuracy = (predictions == y_test).float().mean()

print(f"\n✅ CNN Accuracy: {accuracy.item()*100:.2f}%")
print(f"⏱️ Inference Latency: {latency*1000:.4f} ms per window")

# 6. Save Model
torch.save(model.state_dict(), 'models/cnn_weights.pth')
print("Model saved to models/cnn_weights.pth")