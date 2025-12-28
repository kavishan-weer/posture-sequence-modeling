# Comparative Study of Sequence Modeling Paradigms for Wearable Posture Intelligence

## 🎯 Research Objective

To investigate how different neural architectures (CNN, LSTM, and Transformer) learn human posture from high-dimensional IMU time-series data, and to quantify the trade-offs between temporal memory, real-time latency, and robustness to sensor noise in wearable intelligence.

## 📂 Project Structure

```
posture-sequence-modeling/
├── data/
│   ├── raw/                    # Raw CSV data files
│   └── processed/              # Preprocessed numpy arrays
├── models/                     # Trained model weights and scaler
│   ├── cnn_weights.pth
│   ├── lstm_weights.pth
│   ├── transformer_weights.pth
│   └── scaler.pkl
├── train_models/               # Training scripts
│   ├── train_cnn.py
│   ├── train_lstm.py
│   └── train_transformer.py
├── results/
│   └── plots/                  # Visualization outputs
├── preprocess.py               # Data preprocessing pipeline
├── get_metrics.py             # Model evaluation metrics
├── data_visualization.py       # Data exploration visualizations
├── stress_test.py             # Robustness testing with noise injection
└── requirements.txt           # Python dependencies
```

## 📊 1. Methodology & Data Engineering

The dataset consists of **15-channel time-series data** from 5 MPU6050 sensors (Yaw, Pitch, Roll per sensor) sampled at **~9Hz**.

- **Standardization**: Features were normalized using a global StandardScaler ($\mu=0, \sigma=1$) to ensure equal weight across all sensor axes.
- **Temporal Windowing**: Raw data was segmented into **20-timestep windows** (~2.2s) with **50% overlap**. This allows the models to capture the "geometry of a movement" rather than just static snapshots.
- **Inductive Bias Setup**: Models were designed with roughly similar parameter counts to ensure a fair comparison of their inherent architectural strengths.

## 🧠 2. The Three Paradigms

I implemented and compared three fundamentally different ways of "understanding" motion:

1. **CNN (Local Feature Extractor)**: Focuses on local spatial patterns between sensors.
2. **LSTM (Sequential Memory)**: Uses recursive gates to maintain a hidden state over the 2.2s window.
3. **Tiny Transformer (Global Attention)**: Uses Self-Attention and Positional Encoding to weight the importance of every timestep in relation to every other timestep.

## 🧪 3. Empirical Results & Stress Testing

While all models achieved **100% accuracy** on clean data, I conducted a **Gaussian Noise Injection study** to find the "Breaking Point" of each paradigm.

### Robustness–Latency Benchmark

| Model Paradigm | Clean Acc | σ=1.0 Acc | σ=3.0 Acc | Latency (ms) |
|---------------|-----------|-----------|-----------|--------------|
| **CNN**       | 100%      | 100%      | 98.2%     | 0.0163 ms 🥇 |
| **LSTM**      | 100%      | 100%      | 98.2%     | 0.0232 ms 🥈 |
| **Transformer** | 100%    | 100%      | **100.0%** 🏆 | 0.1040 ms 🥉 |

## 🔍 4. Key Research Insights

### The "Attention" Advantage (Why the Transformer Won Robustness)

The Transformer was the only model to maintain **100% accuracy at $\sigma=3.0$**. This confirms that Self-Attention acts as a powerful denoising mechanism. Unlike the LSTM, which can suffer from "error accumulation" if early steps are noisy, the Transformer looks at the entire window globally and can simply "ignore" noisy timesteps by assigning them lower attention weights.

### The Real-Time Constraint (Why the CNN Wins Deployment)

Despite the Transformer's robustness, it is **~6x slower** than the CNN. In a wearable haptic feedback loop, low latency is critical to prevent user "feedback lag." The CNN provides a nearly perfect balance: **98.2% robustness** with industry-leading inference speed (**0.0163 ms**).

### Inductive Bias vs. Data

For posture detection, spatial relationships (captured by CNN filters) are often more invariant than temporal order (captured by LSTMs). The LSTM's performance match with the CNN at $\sigma=3.0$ suggests that for short 2-second windows, the recursive memory doesn't provide a significant advantage over local spatial patterns.

Read More ----> Analyzing Inductive Biases in Sequence Modeling for Wearable Posture Intelligence.pdf

## 🚀 5. Scaling Hypothesis (Future Work)

To achieve "Frontier" levels of intelligence, the next step is **Self-Supervised Masked Pre-training**. By training a Transformer to reconstruct masked sensor sequences (predicting missing IMU data), we can build a model that understands the "physics of the human body" before it ever sees a label.

## 🛠️ Reproduction

### Installation

```bash
pip install -r requirements.txt
```

### Data Preprocessing

```bash
python preprocess.py
```

This will:
- Load raw CSV data from `data/raw/`
- Apply standardization and temporal windowing
- Save processed arrays to `data/processed/`

### Training Models

Train individual models:

```bash
python train_models/train_cnn.py
python train_models/train_lstm.py
python train_models/train_transformer.py
```

### Evaluation

```bash
python get_metrics.py
```

### Visualization

Generate data visualizations:

```bash
python data_visualization.py
```

### Robustness Testing

Run stress tests with noise injection:

```bash
python stress_test.py
```

## 📦 Dependencies

See `requirements.txt` for the complete list of dependencies. Key libraries include:
- PyTorch (Deep Learning)
- scikit-learn (Preprocessing & Metrics)
- NumPy & Pandas (Data Manipulation)
- Matplotlib & Seaborn (Visualization)


## 👤 Author

Kavishan Weerasinghe

