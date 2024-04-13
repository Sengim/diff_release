import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from utils import *
import matplotlib.pyplot as plt

def hw(input):

    result = np.zeros_like(input)
    for i in range(8):
        result += (input >> i) & 0x1
    return result

target_byte = 2


dataset_root_path = "Datasets"
dataset = load_dataset("ascadv2", dataset_root_path, target_byte=target_byte, traces_dim=2000, leakage_model="ID", n_prof=50000)

dif_folder ="diffusion_ascadv2_22_02_2024_07_22_04_1057297"
model = tf.keras.models.load_model(f"Results/{dif_folder}/trained_model.keras")


dataset.x_profiling, dataset.x_attack= scale_dataset(dataset.x_profiling, dataset.x_attack, StandardScaler())

"""
This part can be altered to any other secret intermediate value. 
"""
# hw_share_2 = hw(dataset.share2_profiling[target_byte].astype(np.uint8))

#hw_share_2 = hw(dataset.share3_profiling[:, target_byte].astype(np.uint8))
hw_share_2 = hw(dataset.share1_profiling[:].astype(np.uint8))

snrs = snr_fast(dataset.x_profiling[:20000], hw_share_2[ :20000])

#Only use the minimum number of instances per class.
minimum_hw_indices = min(len(np.where(hw_share_2==1)[0]),len(np.where(hw_share_2==7)[0])) 
index = np.argsort(snrs)[::-1][0]
bins=np.linspace(-3, 3, 20)
print(minimum_hw_indices)
for i in [1,4, 7]:
    hw_indices = np.where(hw_share_2 == i)

    
    temp = dataset.x_profiling[hw_indices, index]
    #Hacky reshaping to make sure dimensions are appropriate
    print(temp.shape, temp.reshape(-1,1).shape)
    plt.hist(temp.reshape(-1,1)[:minimum_hw_indices],alpha=0.5, bins=bins, label=f"HW = {i}")# ,range=(np.mean(temp)-np.var(temp), np.mean(temp)+np.var(temp)))


    print(f"Original distribution{i},  {np.mean(temp)}, {np.var(temp)}")

plt.legend()
plt.grid()
plt.ylabel("Count")
plt.savefig("orig_hist.png")
plt.close()
dataset.x_profiling = model.predict([dataset.x_profiling, np.zeros_like(hw_share_2) ])
#temp = dataset.x_profiling[hw_indices, index]
minimum_hw_indices = min(len(np.where(hw_share_2==1)[0]),len(np.where(hw_share_2==7)[0])) 
for i in [1,4,7]:
    hw_indices = np.where(hw_share_2 == i)


    temp = dataset.x_profiling[hw_indices, index]
    plt.hist(temp.reshape(-1,1)[ :minimum_hw_indices], alpha=0.5,bins=bins, label=f"HW = {i}")# ,range=(np.mean(temp)-np.var(temp), np.mean(temp)+np.var(temp)))

    print(f"New distribution{i},  {np.mean(temp)}, {np.var(temp)}")

plt.legend()
plt.grid()
plt.ylabel("Count")
plt.savefig("new_hist.png")

