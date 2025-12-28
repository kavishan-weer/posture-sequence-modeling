import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Create folders
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("Starting Data Preprocessing...")

# 1. Load Data
# Note: We drop Timestamp because the model learns from the relative sequence, not the wall-clock time.
good_df = pd.read_csv('data/raw/sit-good-data.csv').drop(columns=['Timestamp'])
bad_df = pd.read_csv('data/raw/sit-bad-data.csv').drop(columns=['Timestamp'])

# 2. Standardization (The Math: z = (x - mean) / std)
# This is critical. It ensures all 15 sensor axes are on the same scale.
scaler = StandardScaler()
all_data = pd.concat([good_df, bad_df])
scaler.fit(all_data)

# Save the scaler. 
joblib.dump(scaler, 'models/scaler.pkl')

good_scaled = scaler.transform(good_df)
bad_scaled = scaler.transform(bad_df)

# 3. Windowing Function (Creating the Sequences)
# We take 20 samples (~2 seconds of data) to make one "prediction"
def create_sequences(data, window_size=20, overlap=0.5):
    sequences = []
    step = int(window_size * (1 - overlap)) # 50% overlap for better density
    for i in range(0, len(data) - window_size + 1, step):
        window = data[i : i + window_size]
        sequences.append(window)
    return np.array(sequences)

X_good = create_sequences(good_scaled)
X_bad = create_sequences(bad_scaled)

# 4. Labeling
# 0 = Good Posture, 1 = Bad Posture
X = np.concatenate([X_good, X_bad], axis=0)
y = np.concatenate([np.zeros(len(X_good)), np.ones(len(X_bad))], axis=0)

# 5. Final Save
np.save('data/processed/X.npy', X.astype(np.float32))
np.save('data/processed/y.npy', y.astype(np.float32))

print(f"✅ Success!")
print(f"Total sequences created: {len(X)}")
print(f"Each sequence shape: {X[0].shape} (20 timesteps x 15 features)")