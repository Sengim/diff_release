import numpy as np
import random

def hw(input: np.uint32):
    out = 0
    temp = input
    for i in range(32):
        if temp % 2 == 1:
            out = out + 1
        temp = temp >> 1
    return out
vec_hw = np.vectorize(hw)

class SimulateHigherOrder():

    def __init__(self, args, order, num_traces, num_attack_traces,  num_informative_features, num_features) -> None:

        add_noise = args["add_noise"]
        leakage_model = "ID"
        self.name = "sim"
        self.order = 0
        self.uni_noise=False
        
        self.num_traces = num_traces
        self.n_profiling = num_traces
        self.n_attack = num_attack_traces
        self.num_features = num_features
        self.num_informative_features = num_informative_features
        self.num_leakage_regions = 1
        self.rsm_mask = True
        self.noise = add_noise

        self.create_pattern(num_informative_features//self.num_leakage_regions, 20)
        indices = [20 * np.random.randint(num_features//20, size=self.num_leakage_regions * (self.order+1))]
        indices = [25]
        self.x_profiling, self.profiling_masks, self.profiling_shares  = self.generate_traces(num_traces, indices)
        self.x_attack, self.attack_masks, self.attack_shares  = self.generate_traces(num_attack_traces, indices)

        self.profiling_labels = self.profiling_masks[:, order] if leakage_model == "ID" else vec_hw(self.profiling_masks[:, order])
        self.attack_labels = self.attack_masks[:, order] if leakage_model == "ID" else vec_hw(self.attack_masks[:, order])


    def generate_traces(self, num_traces, leakage_region_indices):

        masks = np.random.randint(256, size=(num_traces, self.order + 1), dtype =np.uint8)
        shares = np.zeros((num_traces, self.order + 1), dtype=np.uint8)

        for i in range(self.order):
            shares[:, i] = masks[:, i]
        temp = masks[:, 0]
        for i in range(1, self.order+1):
            temp = temp ^ masks[:, i]
        shares[:, self.order] = temp

        leakage_values = self.leakage_spread_hw(shares, self.num_features, num_traces)

        
        traces = np.random.normal(0, self.noise, size=(num_traces, self.num_features))
        for i in range(self.order + 1):
            for j in range(self.num_leakage_regions):
                
                traces = self.include_leakage_around_index(traces, leakage_region_indices[i *self.num_leakage_regions + j], i, leakage_values)
        return traces, masks, shares 
    
   
    def leakage_spread_hw(self, shares, num_points, num_traces):
        leakage_spread = np.zeros((self.order +1, num_points, num_traces))
        for share in range(self.order+1):
            value = shares[:, share].copy()
            for i in range(num_points):
            
                bits = [j for j in range(8)]
                leakage = np.zeros_like(value)
                for j in bits:
                    leakage = leakage + ((value >> j) & 1)
                leakage_spread[share, i, :] = leakage
        return leakage_spread
    

    def include_leakage_around_index(self, traces, index, share, leakage_values):
        
        for j in range(len(self.pattern)):
            traces[:, index + self.pattern[j]] += leakage_values[share, j, :]
        
        return traces

    def create_pattern(self, num_points, spread):
        self.pattern = np.random.default_rng().choice(spread*2, size=num_points, replace=False) - spreads
    