import torch
import numpy as np
import matplotlib.pyplot as plt
from train_cnn import CNNBaseline
from train_lstm import LSTMBaseline
from train_transformer import TransformerBaseline

# 1. Load Data
X_test = torch.tensor(np.load('data/processed/X.npy'), dtype=torch.float32)
y_test = torch.tensor(np.load('data/processed/y.npy'), dtype=torch.float32).unsqueeze(1)

# 2. Initialize and Load Models
cnn = CNNBaseline(); cnn.load_state_dict(torch.load('models/cnn_weights.pth'))
lstm = LSTMBaseline(); lstm.load_state_dict(torch.load('models/lstm_weights.pth'))
tf = TransformerBaseline(); tf.load_state_dict(torch.load('models/transformer_weights.pth'))

models = {"CNN": cnn, "LSTM": lstm, "Transformer": tf}
noise_levels = [0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0]
results = {name: [] for name in models.keys()}

print("🧪 Running Stress Test...")

for sigma in noise_levels:
    # Add Gaussian Noise: x_noisy = x + N(0, sigma)
    noisy_X = X_test + torch.randn_like(X_test) * sigma
    
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            # Adjust shape for CNN if necessary
            input_X = noisy_X.transpose(1, 2) if name == "CNN" else noisy_X
            
            outputs = model(input_X)
            preds = (outputs > 0.5).float()
            acc = (preds == y_test).float().mean().item()
            results[name].append(acc)

# 3. Plot the "Breaking Point"
plt.figure(figsize=(10, 6))
for name, accs in results.items():
    plt.plot(noise_levels, accs, label=name, marker='o')

plt.title("Model Robustness: Accuracy vs. Sensor Noise")
plt.xlabel("Noise Level (Standard Deviation)")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig('results/plots/robustness_results.png')
print("✅ Results saved to results/plots/robustness_results.png")
plt.show()