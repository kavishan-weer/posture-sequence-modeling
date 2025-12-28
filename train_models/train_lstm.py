import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import time

# 1. Load the processed data
X = np.load('data/processed/X.npy') # (Batch, 20, 15)
y = np.load('data/processed/y.npy')

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)

# 3. Define the LSTM Architecture
class LSTMBaseline(nn.Module):
    def __init__(self, input_size=15, hidden_size=32, num_layers=1):
        super(LSTMBaseline, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # h_n is the hidden state after the last timestep
        _, (h_n, c_n) = self.lstm(x)
        # Use the last hidden state to predict the class
        out = self.fc(h_n[-1])
        return self.sigmoid(out)

# 4. Training
model = LSTMBaseline()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("🧠 Training LSTM Baseline...")
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

print(f"\n✅ LSTM Accuracy: {accuracy.item()*100:.2f}%")
print(f"⏱️ Inference Latency: {latency*1000:.4f} ms per window")

torch.save(model.state_dict(), 'models/lstm_weights.pth')