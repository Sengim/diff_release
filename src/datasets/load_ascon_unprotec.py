import numpy as np
import h5py
from tensorflow.keras.utils import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class ReadASCON:

    def __init__(self, n_profiling, n_validation, n_attack, target_byte, leakage_model, file_path, first_sample=0, number_of_samples=772):
        self.name = "ASCON"
        self.n_profiling = n_profiling
        self.n_validation = n_validation
        self.n_attack = n_attack
        self.target_byte = target_byte
        self.leakage_model = leakage_model
        self.file_path = file_path
        self.fs = first_sample
        self.ns = number_of_samples
        self.classes = 9 if leakage_model == "HW" else 256

        self.x_profiling = []
        self.x_validation = []
        self.x_attack = []

        self.y_profiling = []
        self.y_validation = []
        self.y_attack = []

        self.profiling_labels = []
        self.validation_labels = []
        self.attack_labels = []

        self.load_dataset()

    def load_dataset(self):
        in_file = h5py.File(self.file_path, "r")

        profiling_samples = np.array(in_file['random_keys/traces'][:self.n_profiling])
        attack_samples = np.array(in_file['fixed_keys/traces'][:self.n_attack + self.n_validation])
        profiling_plaintext = in_file['random_keys/metadata']['plaintext']
        attack_plaintext = in_file['fixed_keys/metadata']['plaintext']
        profiling_key = in_file['random_keys/metadata']['key']
        attack_key = in_file['fixed_keys/metadata']['key']
        profiling_mask = in_file['random_keys/metadata']["nonce"]
        #print(profiling_mask)
        attack_mask = in_file['fixed_keys/metadata']

        profiling_labels= in_file['random_keys/labels']
        #print(profiling_labels[:10])
        #print
        attack_labels= in_file['fixed_keys/labels']

        profiling_plaintexts = profiling_plaintext[:self.n_profiling]
        profiling_keys = profiling_key[:self.n_profiling]
        profiling_masks = profiling_mask[:self.n_profiling]
        validation_plaintexts = attack_plaintext[:self.n_validation]
        validation_keys = attack_key[:self.n_validation]
        validation_masks = attack_mask[:self.n_validation]
        attack_plaintexts = attack_plaintext[self.n_validation:self.n_validation + self.n_attack]
        attack_keys = attack_key[self.n_validation:self.n_validation + self.n_attack]
        attack_masks = attack_mask[self.n_validation:self.n_validation + self.n_attack]

        self.profiling_keys = profiling_keys
        self.profiling_plaintexts = profiling_plaintexts
        self.profiling_masks = profiling_masks
        self.attack_plaintexts = attack_plaintexts
        self.attack_masks = attack_masks
        self.attack_keys = attack_keys

        self.x_profiling = profiling_samples[:, self.fs:self.fs + self.ns]
        self.x_validation = attack_samples[:self.n_validation, self.fs:self.fs + self.ns]
        self.x_attack = attack_samples[self.n_validation:self.n_validation + self.n_attack, self.fs:self.fs + self.ns]
        ASCON_128_IV = (0x80400c0600000000).to_bytes(16, 'big')
        

        print(ASCON_128_IV)
        self.x_1 = profiling_key[:self.n_profiling, :8]
        self.x_0 = np.array(list(ASCON_128_IV), dtype=np.uint8).repeat(self.x_1.shape[0],axis=0).reshape((self.x_1.shape[0], 16)) 
        #self.x_1 = profiling_key[self.n_profiling:, :8]
        
        self.x_2 = profiling_key[:self.n_profiling, 8:]
        self.x_2[:, 7] = self.x_2[:, 7]^ 0xf0
        self.x_0 = self.x_0[:, 8:]
        self.x_3 = profiling_mask[:self.n_profiling, :8]
        self.x_4 = profiling_mask[:self.n_profiling, 8:]

        self.sbox()
        if self.leakage_model == "HW":
            self.profiling_labels = np.array([bin(j[0]).count("1") + bin(j[1]).count("1") + bin(j[2]).count("1") + bin(j[3]).count("1") for j in self.profiling_labels ], dtype=np.uint16)
        self.validation_labels = attack_labels[:self.n_validation, self.target_byte*8:(self.target_byte+1)*8]
        self.attack_labels = attack_labels[self.n_validation:self.n_validation + self.n_attack, self.target_byte] + attack_labels[self.n_validation:self.n_validation + self.n_attack, self.target_byte+1]*2\
                            + attack_labels[self.n_validation:self.n_validation+ self.n_attack, self.target_byte+2] * 4

    def sbox(self):
        #Any point in Sbox can be picked as desired leakage model
        self.x_0 ^= self.x_4;   
        #self.profiling_labels = self.x_0[:, self.target_byte] 
        self.x_4 ^= self.x_3;
        #self.profiling_labels = self.x_4[:, self.target_byte]
        self.x_2 ^= self.x_1;
        self.profiling_labels = self.x_2[:, ]
        
        t0  = self.x_0;    
        t1  = self.x_1;    
        t2  = self.x_2;    
        t3  = self.x_3;    
        t4  = self.x_4;
        t0 =~ t0;    
        t1 =~ t1;    
        t2 =~ t2;    
        t3 =~ t3;    
        t4 =~ t4;
        t0 &= self.x_1; 
        #self.profiling_labels= t0[:, self.target_byte]   
        t1 &= self.x_2;    
        #self.profiling_labels= t1[:, self.target_byte]  
        t2 &= self.x_3;    
        t3 &= self.x_4;    
        t4 &= self.x_0;
        
        self.x_0 ^= t1;    
        self.x_1 ^= t2;    
        self.x_2 ^= t3;    
        self.x_3 ^= t4;    
        self.x_4 ^= t0;
        
        self.x_1 ^= self.x_0;    
        self.x_0 ^= self.x_4;    
        self.x_3 ^= self.x_2;  
        #self.profiling_labels = self.x_0[:, self.target_byte]  
        self.x_2 =~ self.x_2;
        #self.profiling_labels= (self.x_3^self.x_2)[:, 0:4] 
        

    def rescale(self, reshape_to_cnn):
        self.x_profiling = np.array(self.x_profiling)
        self.x_validation = np.array(self.x_validation)
        self.x_attack = np.array(self.x_attack)

        scaler = StandardScaler()
        self.x_profiling = scaler.fit_transform(self.x_profiling)
        self.x_validation = scaler.transform(self.x_validation)
        self.x_attack = scaler.transform(self.x_attack)

        if reshape_to_cnn:
            print("reshaping to 3 dims")
            self.x_profiling = self.x_profiling.reshape((self.x_profiling.shape[0], self.x_profiling.shape[1], 1))
            self.x_validation = self.x_validation.reshape((self.x_validation.shape[0], self.x_validation.shape[1], 1))
            self.x_attack = self.x_attack.reshape((self.x_attack.shape[0], self.x_attack.shape[1], 1))

