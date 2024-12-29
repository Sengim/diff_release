import numpy as np
import numpy as np
from utils import *
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.decomposition import PCA

def winres_convolve(trace, window=20):

    kernel = np.ones(window) / window  # Create a moving average kernel
    convolved = np.convolve(trace, kernel, mode='same')  # Keep the same size
    # Downsample the convolved result to match the step size
    return convolved

SHARES = 2
TARGET_BYTE = 2
PATH = "/mnt/d/Datasets"
RESULTS = "Results"
N_ATT = 2000


dataset = load_dataset("ascadv2", PATH, TARGET_BYTE, 2000, leakage_model="ID", n_prof=20000)
print(max(snr_fast(dataset.x_profiling, dataset.share1_profiling)))
dataset.x_profiling, dataset.x_attack = scale_dataset(dataset.x_profiling, dataset.x_attack, StandardScaler())
# for i in range(2,30):
#     kernel = np.ones(i) / i  # Create a moving average kernel
#     convolved= np.zeros_like(dataset.x_profiling)
#     for j in range(20000):
#         convolved[j] = np.convolve(dataset.x_profiling[j], kernel, mode='same') 
#     print(i, max(snr_fast(convolved, dataset.share1_profiling)), max(snr_fast(convolved, dataset.share3_profiling[:, TARGET_BYTE])))

# for j in range(2, 20):
#     pca = PCA(j)
#     pcad_traces = pca.inverse_transform(pca.fit_transform(dataset.x_profiling))
#     print(j, max(snr_fast(pcad_traces, dataset.share1_profiling)), max(snr_fast(pcad_traces, dataset.share3_profiling[:, TARGET_BYTE])))

for j in range(50, 100):
    pca = PCA(j)
    pcad_traces = pca.inverse_transform(pca.fit_transform(dataset.x_profiling))
    print(j, max(snr_fast(pcad_traces, dataset.share1_profiling)), max(snr_fast(pcad_traces, dataset.share3_profiling[:, TARGET_BYTE])))