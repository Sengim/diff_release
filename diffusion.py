from train_diffusion import *
from models import *
import numpy as np
from utils import create_directory_results

class Diffussion:

    def __init__(self, **kwargs):
        self.args = kwargs['args']
        self.main_path = self.args["results_root_path"]
        self.dir_results = create_directory_results(self.args, self.main_path)
        self.models = DiffusionModel(self.args, self.dir_results)
        
        

    def train_cgan(self):
        
        np.savez(f"{self.dir_results}/args.npz", args=self.args)
        train_cgan = TrainDiffusion(self.args, self.models, self.dir_results)
        train_cgan.train()
