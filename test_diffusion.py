import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from utils import *
from scripts.attack import *
import matplotlib.pyplot as plt



target_byte = 2
dataset_root_path = "Datasets"
results_root_path = "Results"

dataset = load_dataset("ascadv2", dataset_root_path, target_byte=target_byte, traces_dim=2000, n_prof=400000)
#dataset = load_dataset("eshard", dataset_root_path, target_byte=target_byte, traces_dim=1400, n_prof=5000)
#dataset = load_dataset("ASCAD", dataset_root_path, target_byte=target_byte, traces_dim=700, n_prof=5000)
dif_folder = "diffusion_ASCAD_23_01_2024_07_13_47_5155170"
# dif_folder = "diffusion_eshard_24_10_2023_14_58_35_4847917"
# # dif_folder = "diffusion_aes_hd_25_10_2023_15_47_34_3475741"
dif_folder ="diffusion_ascadv2_22_11_2023_07_22_04_1057297"
model = tf.keras.models.load_model(f"{results_root_path}/{dif_folder}/trained_model.keras")

dataset.x_profiling, dataset.x_attack= scale_dataset(dataset.x_profiling, dataset.x_attack, StandardScaler())
orig_x_prof= dataset.x_profiling.copy()
orig_x_att = dataset.x_attack.copy()
dataset.x_profiling = model.predict([dataset.x_profiling, np.ones(dataset.x_profiling.shape[0])*0])
dataset.x_attack = model.predict([dataset.x_attack, np.ones(dataset.x_attack.shape[0])*0])

ge = []
nt = []
order =3 if dataset.name =="ascadv2" else 2
pi = []
if dataset.name == "ASCAD" or dataset.name == "eshard":

    dataset.share1_profiling = dataset.share1_profiling[target_byte, :]
    dataset.share2_profiling = dataset.share2_profiling[target_byte, :] 
    # dataset.share1_attack = dataset.share1_attack[target_byte, :]
    # dataset.share2_attack = dataset.share2_attack[target_byte, :] 
    pass
if dataset.name == "ascadv2":

    dataset.share1_profiling = dataset.share1_profiling
    dataset.share3_profiling = dataset.share3_profiling[:, target_byte]
    dataset.share2_profiling = dataset.share2_profiling 
    # dataset.share1_attack = dataset.share1_attack[target_byte, :]
    # dataset.share2_attack = dataset.share2_attack[target_byte, :] 
    pass
old_snrs3 = [0, 1]
old_snrs1 = [0,1]
old_snrs2 = [0, 1]
old_snrs1 = snr_fast(dataset.x_profiling, dataset.share1_profiling)
old_snrs2 = snr_fast(dataset.x_profiling, dataset.share2_profiling)
plt.plot(old_snrs1, label="Share 1")
plt.plot(old_snrs2, label="Share 2")
if order == 3:
    old_snrs3 = snr_fast(dataset.x_profiling, dataset.share3_profiling)

    plt.plot(old_snrs3, label="Share 3")

print(f"Orig: {np.max(old_snrs1)}, {np.max(old_snrs2)}, {np.max(old_snrs3)}")
plt.ylabel("SNR")
plt.xlabel("Samples")

plt.legend()
plt.show()
#plt.savefig(f"{results_root_path}/{dif_folder}/{dif_folder}/snr_orig.png")
plt.close()

t = 5
for i in range(0, 16):

    dataset.x_profiling= model.predict([orig_x_prof, np.ones(dataset.x_profiling.shape[0])*i])

    old_snrs1 = snr_fast(dataset.x_profiling, dataset.share1_profiling)
    old_snrs2 = snr_fast(dataset.x_profiling, dataset.share2_profiling)
    plt.plot(old_snrs1, label="Share 1")
    plt.plot(old_snrs2, label="Share 2")
    if order == 3:
        old_snrs3 = snr_fast(dataset.x_profiling, dataset.share3_profiling)
        # old_snrs3 = snr_fast(dataset.x_attack, dataset.share3_attack)
        plt.plot(old_snrs3, label="Share 3")
    print(f"{i}:{np.max(old_snrs1)}, {np.max(old_snrs2)}, {np.max(old_snrs3)}")



    plt.ylabel("SNR")
    plt.xlabel("Samples")
    plt.legend()    
    plt.show()
    #plt.savefig(f"{results_root_path}/{dif_folder}/{dif_folder}/snr_t{i}.png")
    plt.close()

