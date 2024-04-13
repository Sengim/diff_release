import numpy as np
from datetime import datetime
import os
from src.datasets.paths import *
from src.datasets.load_ascadr import *
from src.datasets.load_ascadf import *
from src.datasets.load_eshard import *
from src.datasets.load_aes_hd import *
from src.datasets.load_aes_hd_mm import *
from src.datasets.load_ascadv2 import *
from src.datasets.simulate_higher_order import *
from src.datasets.load_cswap_arith import *
from src.datasets.load_cswap_pointer import *
from src.datasets.load_ascon_unprotec import *
from os.path import exists

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
    if identifier == "aes_hd":
        dataset = ReadAESHD(200000 if n_prof is None else n_prof, 0, 20000, target_byte, leakage_model,
                                                         dataset_file,
                                                         number_of_samples=traces_dim)
    if identifier == "aes_hd_mm":
        dataset = ReadAESHDMM(200000 if n_prof is None else n_prof, 0, 20000, target_byte, leakage_model,
                                                         dataset_file,
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