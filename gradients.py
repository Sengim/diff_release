import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from utils import *
import matplotlib.pyplot as plt

target_byte = 2
dataset_root_path = "Datasets"
dataset = load_dataset("ascadv2", dataset_root_path, target_byte=target_byte, traces_dim=2000, leakage_model="ID", n_prof=10000)
dif_folder = ""


model = tf.keras.models.load_model(f"Results/{dif_folder}/trained_model.keras")
dataset.x_profiling, _ = scale_dataset(dataset.x_profiling, None, StandardScaler())
snrs = snr_fast(dataset.x_profiling[:5000], dataset.share3_profiling[ :5000, target_byte])

index = np.argsort(snrs)[::-1][:50]
print(index[0])
temp = tf.Variable(dataset.x_profiling[0:500])
with tf.GradientTape() as tape:
   pred = model([temp,  np.zeros(500)], training=False)
   #print(pred)
   loss = tf.math.reduce_mean(pred[:, index[0]])
   #print(loss)
  
grads = tape.gradient(loss, temp)
#print(grads)
dgrad_abs = tf.math.abs(grads)
temp = dgrad_abs.numpy()
graddex = np.argsort(temp[0])[::-1][:50]
print(index, graddex)

fig, ax1 = plt.subplots()
ax1.set_xlabel("Samples")
ax1.set_ylabel("SNR")

ax1.plot(snrs, label="SNR", alpha=1, color='orange')

ax2 = ax1.twinx()

plt.plot(dgrad_abs[0], label="Gradients", alpha=0.5)
ax2.set_ylabel("Gradients")
handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
fig.legend(handles, labels)
fig.tight_layout()
plt.grid()
plt.show()
