import numpy as np
from utils import *
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

SHARES = 2
TARGET_BYTE = 2
PATH = "Datasets"
RESULTS = "Results"
N_ATT = 2000


def create_dataset_ASCADf():

    #indices for Max snr peak of byte 2
    indices = [156, 517]
    dif_folder = "diffusion_ASCAD_06_02_2024_11_18_10_8996285"
    model = tf.keras.models.load_model(f"{RESULTS}/{dif_folder}/trained_model.keras") 
    dataset = load_dataset("ASCAD", PATH, TARGET_BYTE, 700, leakage_model="HW")
    dataset.x_profiling, dataset.x_attack = scale_dataset(dataset.x_profiling, dataset.x_attack, StandardScaler())
    dataset.x_attack = model.predict([dataset.x_attack, np.ones(dataset.x_attack.shape[0])*15])
    dataset.x_attack = dataset.x_attack[:, indices]
    return dataset

def create_dataset_ESHARD():
    #indices for Max snr peak of byte 2
    indices = [840, 43] 
    dif_folder = "diffusion_eshard_08_02_2024_21_25_39_6496171"
    model = tf.keras.models.load_model(f"{RESULTS}/{dif_folder}/trained_model.keras") 
    dataset = load_dataset("eshard", PATH, TARGET_BYTE, 1400, leakage_model="HW")
    dataset.x_profiling, dataset.x_attack = scale_dataset(dataset.x_profiling, dataset.x_attack, StandardScaler())
    dataset.x_attack = model.predict([dataset.x_attack, np.ones(dataset.x_attack.shape[0]*15)])
    dataset.x_attack = dataset.x_attack[:, indices]
    return dataset

def hw(input: np.uint8):
    out = 0
    temp = input
    for i in range(8):
        if temp % 2 == 1:
            out = out + 1
        temp = temp >> 1
    return out



def online_coeffs(traces, leakage_table: np.ndarray):
    """
    Calculates coefficients using online calculation method.

    :param leakage_table: the leakage table used.
    :return: the coefficients, calculated in an online manner.
    """
    results = np.zeros(N_ATT)
    ht = 0
    h = 0
    t = 0
    hh = 0
    tt = 0
    for d in range(N_ATT):
        h += leakage_table[d]
        hh += np.power(leakage_table[d], 2)
        t += traces[d]
        tt = np.power(traces[d], 2)
        ht += leakage_table[d] * traces[d]
        denominator = (np.power(h, 2) - (d * hh)) * (np.power(t, 2) - (d * tt))
        numerator = d * ht - h * t
        results[d] = np.abs(numerator/np.sqrt(np.abs(denominator)))
    return results
        

rank = 0
dataset = create_dataset_ASCADf()
leakage = np.abs(dataset.x_attack[:, 0] - dataset.x_attack[:, 1])
hyps = dataset.labels_key_hypothesis_attack.T
correct_key = dataset.attack_keys[1, TARGET_BYTE]
results = np.zeros((256, N_ATT))
ranks = np.zeros(N_ATT)
for i in range(50):
    results = np.zeros((256, N_ATT))
    leakage, hyps = shuffle(leakage, hyps)
    #correct_key = 0
    dataset.attack_plaintexts
    for k in range(256):
        results[k] = online_coeffs(leakage, hyps[:, k]) 
    for n in range(N_ATT):
        keys = np.argsort(results[:, n])[::-1]
        for j in range(256):
            if keys[j] == correct_key:
                ranks[n] += j
                break 

    print(i, ranks)  
    

print(ranks/50)
np.savez("ranks_dif_ASCAD.npz", ranks = ranks/50)






