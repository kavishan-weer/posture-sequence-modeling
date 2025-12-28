import torch
import numpy as np
from train_cnn import CNNBaseline
from train_lstm import LSTMBaseline
from train_transformer import TransformerBaseline # Use the correct class name

# 1. Load Data
X_test = torch.tensor(np.load('data/processed/X.npy'), dtype=torch.float32)
y_test = torch.tensor(np.load('data/processed/y.npy'), dtype=torch.float32).unsqueeze(1)

# 2. Load Models
cnn = CNNBaseline(); cnn.load_state_dict(torch.load('models/cnn_weights.pth'))
lstm = LSTMBaseline(); lstm.load_state_dict(torch.load('models/lstm_weights.pth'))
tf = TransformerBaseline(); tf.load_state_dict(torch.load('models/transformer_weights.pth'))

models = {"CNN": cnn, "LSTM": lstm, "Transformer": tf}
target_noise = [0.0, 1.0, 3.0]

print(f"{'Model':<12} | {'Sigma=0.0':<10} | {'Sigma=1.0':<10} | {'Sigma=3.0':<10}")
print("-" * 50)

for name, model in models.items():
    model.eval()
    row = [name]
    for sigma in target_noise:
        with torch.no_grad():
            # Add noise
            noisy_X = X_test + torch.randn_like(X_test) * sigma
            # CNN needs axis swap
            input_X = noisy_X.transpose(1, 2) if name == "CNN" else noisy_X
            
            outputs = model(input_X)
            preds = (outputs > 0.5).float()
            acc = (preds == y_test).float().mean().item()
            row.append(f"{acc*100:.1f}%")
    
    print(f"{row[0]:<12} | {row[1]:<10} | {row[2]:<10} | {row[3]:<10}")