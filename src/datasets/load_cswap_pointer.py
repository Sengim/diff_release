import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt


class ReadCswapPointer:

    def __init__(self, file_path, n_profiling=31875*2, n_validation=0, n_attack=12750):
        self.name = "cswap_pointer"
        self.n_profiling = n_profiling
        self.n_validation = n_validation
        self.n_attack = n_attack
        self.file_path = file_path
        self.target_params = {
            "name": "cswap_pointer",
            "data_length": 2,
            "first_sample": 0,
            "number_of_samples": 1000,
            "n_set1": n_profiling,
            "n_set2": n_validation,
            "n_attack": n_attack,
            "classes": 2,
            "epochs": 25,
            "mini-batch": 64
        }
        self.x_profiling, self.x_validation, self.profiling_labels, self.validation_labels = self.train_sets()
        self.x_attack, self.attack_labels = self.test_set()
        print(self.x_profiling.shape)
        print(self.profiling_labels.shape, self.profiling_labels[0])
        print( self.profiling_labels[0:10])
        
        #self.rescale(False)


    def train_sets(self):
        hf = h5py.File("{}".format(self.file_path), 'r')
        profiling_samples = np.array(hf.get('profiling_traces'))
        profiling_data = np.array(hf.get('profiling_data'))

        fs = self.target_params["first_sample"]
        ns = self.target_params["number_of_samples"]

        profiling_samples = profiling_samples[:, fs: fs + ns]
        training_dataset_reshaped = profiling_samples.reshape((profiling_samples.shape[0], profiling_samples.shape[1], 1))

        y_set1 = profiling_data[0:self.target_params["n_set1"]]
        y_set2 = profiling_data[self.target_params["n_set1"]:self.target_params["n_set1"] + self.target_params["n_set2"]]
        set1_samples = profiling_samples[0:self.target_params["n_set1"]]
        set2_samples = profiling_samples[self.target_params["n_set1"]:self.target_params["n_set1"] + self.target_params["n_set2"]]

        return  set1_samples, set2_samples, y_set1, y_set2

    def test_set(self):
        hf = h5py.File("{}".format(self.file_path), 'r')
        test_samples = np.array(hf.get('attacking_traces'))
        test_data = np.array(hf.get('attacking_data'))

        fs = self.target_params["first_sample"]
        ns = self.target_params["number_of_samples"]
        test_samples = test_samples[:, fs: fs + ns]

        test_samples = test_samples[0:self.target_params["n_attack"]]
        y_test = test_data[0:self.target_params["n_attack"]]

        x_test = test_samples.reshape((test_samples.shape[0], test_samples.shape[1], 1))
        return test_samples, y_test

    def rescale(self, reshape_to_cnn):
        self.x_profiling = np.array(self.x_profiling)
        self.x_validation = np.array(self.x_validation)
        self.x_attack = np.array(self.x_attack)

        scaler = StandardScaler()
        self.x_profiling = scaler.fit_transform(self.x_profiling)
        if self.n_validation >0:
            self.x_validation = scaler.transform(self.x_validation)
        self.x_attack = scaler.transform(self.x_attack)

        if reshape_to_cnn:
            print("reshaping to 3 dims")
            self.x_profiling = self.x_profiling.reshape((self.x_profiling.shape[0], self.x_profiling.shape[1], 1))
            if self.n_validation >0:
                self.x_validation = self.x_validation.reshape((self.x_validation.shape[0], self.x_validation.shape[1], 1))
            self.x_attack = self.x_attack.reshape((self.x_attack.shape[0], self.x_attack.shape[1], 1))
