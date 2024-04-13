import numpy as np
from datetime import datetime
import os
from src.datasets.paths import *
from src.datasets.load_ascadr import *
from src.datasets.load_ascadf import *
from src.datasets.load_dpav42 import *
from src.datasets.load_eshard import *
from src.datasets.load_chesctf import *
from src.datasets.load_aes_hd import *
from src.datasets.load_aes_hd_mm import *
from src.datasets.load_ascadv2 import *
from src.datasets.load_spook_sw3 import *
from src.datasets.simulate_higher_order import *
from src.datasets.load_cswap_arith import *
from src.datasets.load_cswap_pointer import *
from src.datasets.load_ascon_unprotec import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from os.path import exists
from sklearn.decomposition import PCA

def snr_fast(x, y):
    ns = x.shape[1]
    unique = np.unique(y)
    means = np.zeros((len(unique), ns))
    variances = np.zeros((len(unique), ns))

    for i, u in enumerate(unique):
        new_x = x[np.argwhere(y == int(u))]
        means[i] = np.mean(new_x, axis=0)
        variances[i] = np.var(new_x, axis=0)
    return np.var(means, axis=0) / np.mean(variances, axis=0)

def create_directory_results(args, path):
    now = datetime.now()
    now_str = f"{now.strftime('%d_%m_%Y_%H_%M_%S')}_{np.random.randint(1000000, 10000000)}"
    dir_results = f"{path}/diffusion_{args['dataset']}_{now_str}"
    if not os.path.exists(dir_results):
        os.mkdir(dir_results)
    return dir_results


def load_dataset(identifier: str, path: str, target_byte: int, traces_dim: int, leakage_model="ID", num_features=-1, n_prof=None, args=None):
    
    dataset_file = get_dataset_filepath(path, identifier, traces_dim, leakage_model=leakage_model) if not identifier == "sim" else None
    if identifier == "eshard":
        dataset = ReadEshard(70000 if n_prof is None else n_prof, 0, 10000, target_byte, leakage_model, dataset_file, number_of_samples=traces_dim)
    if identifier == "ascad-variable":
        dataset = ReadASCADr(200000 if n_prof is None else n_prof, 0, 10000, target_byte, leakage_model,
                                                dataset_file,
                                                number_of_samples=traces_dim)
    if identifier == "ASCAD":
        dataset = ReadASCADf(50000 if n_prof is None else n_prof, 0, 10000, target_byte, leakage_model,
                                                dataset_file,
                                                number_of_samples=traces_dim)
    if identifier == "dpa_v42":
        dataset = ReadDPAV42(70000 if n_prof is None else n_prof, 0, 5000, target_byte, leakage_model,
                                                dataset_file,
                                                number_of_samples=traces_dim)
    if identifier == "ches_ctf":
        dataset = ReadCHESCTF(45000 if n_prof is None else n_prof, 0, 5000, target_byte, leakage_model,
                                                         dataset_file,
                                                         number_of_samples=traces_dim)
    if identifier == "aes_hd":
        dataset = ReadAESHD(200000 if n_prof is None else n_prof, 0, 20000, target_byte, leakage_model,
                                                         dataset_file,
                                                         number_of_samples=traces_dim)
    if identifier == "aes_hd_mm":
        dataset = ReadAESHDMM(200000 if n_prof is None else n_prof, 0, 20000, target_byte, leakage_model,
                                                         dataset_file,
                                                         number_of_samples=traces_dim)
        
    if identifier == "spook_sw3":
                 return ReadSpookSW3(200000 if n_prof is None else n_prof, 0,0, target_byte, leakage_model, dataset_file,
                                   number_of_samples=traces_dim)
    if identifier == "ascadv2":
                 return ReadASCADv2(200000 if n_prof is None else n_prof, 0,10000, target_byte, leakage_model, dataset_file,
                                   number_of_samples=traces_dim)
    
    if identifier == "ascon":
                 return ReadASCON(50000 if n_prof is None else n_prof, 0,50000, target_byte, leakage_model, dataset_file,
                                   number_of_samples=traces_dim)
    if identifier == "sim":
         return SimulateHigherOrder(args, 1, n_prof, 1, num_informative_features=num_features, num_features=traces_dim)
    if identifier == "cswap_pointer":
         return ReadCswapPointer(dataset_file)
    if identifier == "cswap_arithmetic":
         return ReadCswapArith(dataset_file)
         
    return dataset



def scale_dataset(prof_set, attack_set, scaler):
        prof_new = scaler.fit_transform(prof_set)
        if attack_set is not None:
            attack_new = scaler.transform(attack_set)
        else:
            attack_new = None
        return prof_new, attack_new

def get_lda_features(dataset, target_byte: int, n_components=10):
    order = 1 if not  (dataset.name=="simulate" or dataset.name =="spook_sw3") else dataset.order
    order = order + 1
    n_poi_snr = min(100, dataset.ns//order)
    x_prof, x_att = get_features(dataset, target_byte, n_poi=n_poi_snr*order)

    result_prof, result_att = None, None
    for i in range(order):
        lda = LinearDiscriminantAnalysis(n_components=n_components//(order))
        if dataset.name == "spook_sw3":
            lda.fit(x_prof[:20000, i*n_poi_snr:(i+1)*n_poi_snr], dataset.profiling_shares[ :20000,target_byte,  i])
        elif dataset.name == "simulate":
            lda.fit(x_prof[:20000, i*n_poi_snr:(i+1)*n_poi_snr], dataset.profiling_shares[ :20000,  i])
        else:
             temp = dataset.share1_profiling[target_byte, :20000] if i == 0 else dataset.share2_profiling[target_byte, :20000]
             lda.fit(x_prof[:20000, i*n_poi_snr:(i+1)*n_poi_snr], temp)
        
        s1_prof = lda.transform(x_prof[:, i*n_poi_snr:(i+1)*n_poi_snr] )
        s1_att = lda.transform(x_att[:, i*n_poi_snr:(i+1)*n_poi_snr] )
        result_prof = s1_prof if i == 0 else np.append(result_prof, s1_prof, axis=1)
        result_att = s1_att if i == 0 else np.append(result_att, s1_att, axis=1)
    return result_prof, result_att

def get_pca_features(dataset, target_byte: int, n_components=10):

    order = 1 if not  (dataset.name=="simulate" or dataset.name =="spook_sw3") else dataset.order
    order = order + 1
    n_poi_snr = min(100, dataset.ns//order)
    x_prof, x_att = get_features(dataset, target_byte, n_poi=n_poi_snr*order)

    result_prof, result_att = None, None
    for i in range(order):
        pca = PCA(n_components=n_components//(order))
        pca.fit(x_prof[:20000, i*n_poi_snr:(i+1)*n_poi_snr])
        s1_prof = pca.transform(x_prof[:, i*n_poi_snr:(i+1)*n_poi_snr] )
        s1_att = pca.transform(x_att[:, i*n_poi_snr:(i+1)*n_poi_snr] )
        result_prof = s1_prof if i == 0 else np.append(result_prof, s1_prof, axis=1)
        result_att = s1_att if i == 0 else np.append(result_att, s1_att, axis=1)
    return result_prof, result_att


def get_features(dataset, target_byte: int, n_poi=100):
    snr_arr = get_snr_shares(dataset, target_byte)
    if snr_arr.shape[0]>2:
         return get_3_features(dataset, snr_arr, n_poi)
    snr_val_share_1 = snr_arr[0]
    snr_val_share_2 = snr_arr[1]
    snr_val_share_1[np.isnan(snr_val_share_1)] = 0
    snr_val_share_2[np.isnan(snr_val_share_2)] = 0
    
    ind_snr_masks_poi_sm = np.argsort(snr_val_share_1)[::-1][:int(n_poi / 2)]
    sorted_poi_masks_sm = np.argsort(snr_val_share_1)[::-1]
    ind_snr_masks_poi_sm_sorted = np.sort(ind_snr_masks_poi_sm)

    ind_snr_masks_poi_r2 = np.argsort(snr_val_share_2)[::-1][:int(n_poi / 2)]
    sorted_poi_masks_r2 = np.argsort(snr_val_share_2)[::-1]
    ind_snr_masks_poi_r2_sorted = np.sort(ind_snr_masks_poi_r2)

    poi_profiling = np.concatenate((ind_snr_masks_poi_sm_sorted, ind_snr_masks_poi_r2_sorted), axis=0)

    return dataset.x_profiling[:, poi_profiling], dataset.x_attack[:, poi_profiling]


def get_3_features(dataset, snr_arr, n_poi=100):
    n_poi = 15
    snr_val_share_1 = snr_arr[0]
    snr_val_share_2 = snr_arr[1]
    snr_val_share_1[np.isnan(snr_val_share_1)] = 0
    snr_val_share_2[np.isnan(snr_val_share_2)] = 0
    
    ind_snr_masks_poi_sm = np.argsort(snr_val_share_1)[::-1][:int(n_poi / 3)]
    sorted_poi_masks_sm = np.argsort(snr_val_share_1)[::-1]
    ind_snr_masks_poi_sm_sorted = np.sort(ind_snr_masks_poi_sm)
    poi_profiling = ind_snr_masks_poi_sm_sorted
    for i in range(1,snr_arr.shape[0]):
        ind_snr_masks_poi_r2 = np.argsort(snr_val_share_2)[::-1][:int(n_poi / 3)]
        sorted_poi_masks_r2 = np.argsort(snr_val_share_2)[::-1]
        ind_snr_masks_poi_r2_sorted = np.sort(ind_snr_masks_poi_r2)

        poi_profiling = np.concatenate((poi_profiling, ind_snr_masks_poi_r2_sorted), axis=0)

    return dataset.x_profiling[:, poi_profiling], dataset.x_attack[:, poi_profiling]


def get_snr_shares(dataset, target_byte):
    if dataset.name == "simulate":
        return get_snr_shares_sim(dataset)
    elif dataset.name =="spook_sw3":
        return get_snr_shares_spook(dataset, target_byte)
    elif dataset.name == "ascadv2":
         return get_snr_shares_ascadv2(dataset)
    result_arr = np.zeros((2, dataset.ns))
    print(dataset.ns, dataset.x_profiling.shape)
    result_arr[0, :]= snr_fast(np.array(dataset.x_profiling[:min(dataset.x_profiling.shape[0], 20000)], dtype=np.int16), np.asarray(dataset.share1_profiling[target_byte, :min(dataset.x_profiling.shape[0], 20000)]))
    result_arr[1, :] = snr_fast(np.array(dataset.x_profiling[:min(dataset.x_profiling.shape[0], 20000)], dtype=np.int16), np.asarray(dataset.share2_profiling[target_byte, :min(dataset.x_profiling.shape[0], 20000)]))
    return result_arr

def get_snr_shares_sim(dataset):
    result_arr = np.zeros((dataset.order+ 1, dataset.ns))
    order = dataset.order + 1
    for i in range(order):
        result_arr[i, :]= snr_fast(np.array(dataset.x_profiling[:min(dataset.x_profiling.shape[0], 20000)], dtype=np.int16), np.asarray(dataset.profiling_shares[ :min(dataset.x_profiling.shape[0], 20000), i]))
    return result_arr





def get_snr_shares_spook(dataset, target_byte):
    result_arr = np.zeros((dataset.order+1, dataset.ns))
    order = dataset.order+1
    for i in range(order):
        result_arr[i, :]= snr_fast(np.array(dataset.x_profiling[:min(dataset.x_profiling.shape[0], 20000)], dtype=np.int16), np.asarray(dataset.prof_shares[ :min(dataset.x_profiling.shape[0], 20000), target_byte, i]))
    return result_arr
def get_snr_shares_ascadv2(dataset):
    result_arr = np.zeros((3, dataset.ns))
    order = 3

    result_arr[0, :]= snr_fast(np.array(dataset.x_profiling[:min(dataset.x_profiling.shape[0], 20000)], dtype=np.int16), np.asarray(dataset.share1_profiling[ :min(dataset.x_profiling.shape[0], 20000)]))
    result_arr[1, :]= snr_fast(np.array(dataset.x_profiling[:min(dataset.x_profiling.shape[0], 20000)], dtype=np.int16), np.asarray(dataset.share2_profiling[ :min(dataset.x_profiling.shape[0], 20000)]))
    result_arr[2, :]= snr_fast(np.array(dataset.x_profiling[:min(dataset.x_profiling.shape[0], 20000)], dtype=np.int16), np.asarray(dataset.share3_profiling[ :min(dataset.x_profiling.shape[0], 20000), dataset.target_byte]))
    return result_arr

def sost_fast(x, y):
    ns = x.shape[1]
    unique = np.unique(y)
    means = np.zeros((len(unique), ns))
    variances = np.zeros((len(unique), ns))
    counts = np.zeros(len(unique))

    for i, u in enumerate(unique):
        new_x = x[np.argwhere(y == int(u))]
        means[i] = np.mean(new_x, axis=0)
        variances[i] = np.var(new_x, axis=0)
        counts[i] = len(new_x)


    temp = (means[0] - means[1])**2
    return temp

from numba import njit

aes_sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])

"""
Two Tables to process a field multiplication over GF(256): a*b = alog (log(a) + log(b) mod 255)
"""
log_table = np.array([0, 0, 25, 1, 50, 2, 26, 198, 75, 199, 27, 104, 51, 238, 223, 3,
                      100, 4, 224, 14, 52, 141, 129, 239, 76, 113, 8, 200, 248, 105, 28, 193,
                      125, 194, 29, 181, 249, 185, 39, 106, 77, 228, 166, 114, 154, 201, 9, 120,
                      101, 47, 138, 5, 33, 15, 225, 36, 18, 240, 130, 69, 53, 147, 218, 142,
                      150, 143, 219, 189, 54, 208, 206, 148, 19, 92, 210, 241, 64, 70, 131, 56,
                      102, 221, 253, 48, 191, 6, 139, 98, 179, 37, 226, 152, 34, 136, 145, 16,
                      126, 110, 72, 195, 163, 182, 30, 66, 58, 107, 40, 84, 250, 133, 61, 186,
                      43, 121, 10, 21, 155, 159, 94, 202, 78, 212, 172, 229, 243, 115, 167, 87,
                      175, 88, 168, 80, 244, 234, 214, 116, 79, 174, 233, 213, 231, 230, 173, 232,
                      44, 215, 117, 122, 235, 22, 11, 245, 89, 203, 95, 176, 156, 169, 81, 160,
                      127, 12, 246, 111, 23, 196, 73, 236, 216, 67, 31, 45, 164, 118, 123, 183,
                      204, 187, 62, 90, 251, 96, 177, 134, 59, 82, 161, 108, 170, 85, 41, 157,
                      151, 178, 135, 144, 97, 190, 220, 252, 188, 149, 207, 205, 55, 63, 91, 209,
                      83, 57, 132, 60, 65, 162, 109, 71, 20, 42, 158, 93, 86, 242, 211, 171,
                      68, 17, 146, 217, 35, 32, 46, 137, 180, 124, 184, 38, 119, 153, 227, 165,
                      103, 74, 237, 222, 197, 49, 254, 24, 13, 99, 140, 128, 192, 247, 112, 7])

alog_table = np.array([1, 3, 5, 15, 17, 51, 85, 255, 26, 46, 114, 150, 161, 248, 19, 53,
                       95, 225, 56, 72, 216, 115, 149, 164, 247, 2, 6, 10, 30, 34, 102, 170,
                       229, 52, 92, 228, 55, 89, 235, 38, 106, 190, 217, 112, 144, 171, 230, 49,
                       83, 245, 4, 12, 20, 60, 68, 204, 79, 209, 104, 184, 211, 110, 178, 205,
                       76, 212, 103, 169, 224, 59, 77, 215, 98, 166, 241, 8, 24, 40, 120, 136,
                       131, 158, 185, 208, 107, 189, 220, 127, 129, 152, 179, 206, 73, 219, 118, 154,
                       181, 196, 87, 249, 16, 48, 80, 240, 11, 29, 39, 105, 187, 214, 97, 163,
                       254, 25, 43, 125, 135, 146, 173, 236, 47, 113, 147, 174, 233, 32, 96, 160,
                       251, 22, 58, 78, 210, 109, 183, 194, 93, 231, 50, 86, 250, 21, 63, 65,
                       195, 94, 226, 61, 71, 201, 64, 192, 91, 237, 44, 116, 156, 191, 218, 117,
                       159, 186, 213, 100, 172, 239, 42, 126, 130, 157, 188, 223, 122, 142, 137, 128,
                       155, 182, 193, 88, 232, 35, 101, 175, 234, 37, 111, 177, 200, 67, 197, 84,
                       252, 31, 33, 99, 165, 244, 7, 9, 27, 45, 119, 153, 176, 203, 70, 202,
                       69, 207, 74, 222, 121, 139, 134, 145, 168, 227, 62, 66, 198, 81, 243, 14,
                       18, 54, 90, 238, 41, 123, 141, 140, 143, 138, 133, 148, 167, 242, 13, 23,
                       57, 75, 221, 124, 132, 151, 162, 253, 28, 36, 108, 180, 199, 82, 246, 1])

@njit
def multGF256(a, b):
    """ Multiplication function in GF(2^8) """
    if (a == 0) or (b == 0):
        return 0
    else:
        return alog_table[(log_table[a] + log_table[b]) % 255]

