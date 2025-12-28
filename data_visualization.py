import matplotlib.pyplot as plt
import pandas as pd

# Load the data
good_df = pd.read_csv('data/raw/sit-good-data.csv')
bad_df = pd.read_csv('data/raw/sit-bad-data.csv')

# Plotting specific features to see the difference
# Focus on Pitch and Roll of Sensor 1 and 2 
features_to_plot = ['Pitch1', 'Roll1', 'Pitch2', 'Roll2']

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

# Plot Good Posture
for feat in features_to_plot:
    axes[0].plot(good_df[feat].values[:200], label=feat)
axes[0].set_title('Good Posture - Signal Patterns (First 200 samples)')
axes[0].set_ylabel('Degrees')
axes[0].legend()
axes[0].grid(True)

# Plot Bad Posture
for feat in features_to_plot:
    axes[1].plot(bad_df[feat].values[:200], label=feat)
axes[1].set_title('Bad Posture - Signal Patterns (First 200 samples)')
axes[1].set_ylabel('Degrees')
axes[1].set_xlabel('Sample Index')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('data_visualization.png')
plt.close()
