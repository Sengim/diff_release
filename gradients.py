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













exit()
print(index)
print(model.layers[0])
newvar = model.layers[-1].get_weights()
print(newvar[1].shape)
for i in range(dataset.x_profiling.shape[1]):
  if i in index:
     print(f"Haha{i}")
     continue
  newvar[0][:, i] = newvar[0][:, i] *0
  newvar[1][i] =  0

model.layers[-1].set_weights(newvar)

#print(model.layers[0].get_weights())

print("here")

analyzer = innvestigate.create_analyzer("lrp.z", model)#, neuron_selection_mode="all")

dataset.x_profiling, _= scale_dataset(dataset.x_profiling, None, StandardScaler())
broad_dataset = HackyDatasetObject()

temp = np.zeros(dataset.x_profiling.shape[1])
ts = np.zeros(500) + 0
indices = [[i, index] for i in range(500)]
# for i in range( 500):
#     analysis = analyzer.analyze([[dataset.x_profiling[i]], [0], [[0, index]]], index)
#     temp += np.abs(analysis[0][0])
#     indices
temp = np.zeros((50, dataset.x_profiling.shape[1]))
analysis = analyzer.analyze([dataset.x_profiling[:500], ts])
temp = np.average(np.abs(analysis[0]), axis=0)
plt.plot(snrs/snrs[index[0]], label= "SNR")
plt.plot(temp/np.max(np.abs(temp)), label="LRP")
plt.legend()
plt.show()

# for j in range(len(index)):

#   ts = np.zeros(5000)
#   indices = [[i, index[j]] for i in range(5000)]  

#   temp[j]= np.sum(np.abs(analysis[0]), axis=0)

# plt.plot(snrs/snrs[index[0]], label= "SNR")
# plt.plot(np.average(temp,axis=0 )/np.max(np.abs(np.average(temp, axis=0))), label="LRP")
# plt.legend()
# plt.show()
