import numpy as np
from utils import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from sklearn.utils import shuffle


class TrainDiffusion:

    def __init__(self, args, model, results_dir):

        self.dataset = load_dataset(args["dataset"], args["dataset_root_path"], args["target_byte"], args["dataset_dim"], args["leakage_model"], n_prof=args["n_profiling"], num_features=args["n_inf_feat"], args=args)
        self.result_dir = results_dir
        self.intermediate_eval = False
        if self.dataset.name.__contains__("cswap"):
            self.dataset.rescale(False)
        else:
            self.dataset.x_profiling, self.dataset.x_attack  = scale_dataset(self.dataset.x_profiling, self.dataset.x_attack, StandardScaler())

        self.model = model
        self.epochs = args["epochs"]
        self.batch_size = args["batch_size"]
        self.tsteps = args["tsteps"]    # how many steps for a noisy image into clear
        self.t_bar = np.linspace(0, 1.0, self.tsteps + 1)
        self.snr_ref_label = self.select_label()
        self.pre_noise_snr = None
        if args["add_noise"] > 0.0 and not args["dataset"] =="sim" :
            self.pre_noise_snr = np.max(snr_fast(self.dataset.x_profiling[:20000], self.snr_ref_label[:20000]))
            self.dataset.x_profiling += np.random.normal(0, args["add_noise"], self.dataset.x_profiling.shape)

        snr = snr_fast(self.dataset.x_profiling[:20000], self.snr_ref_label[:20000])
        self.orig_snr = np.max(snr)
        self.max_snr0 = []
        self.max_snrt = []
        self.loss = []

    def train(self):

        num_batches = self.dataset.x_profiling.shape[0]//self.batch_size

        for i in range(self.epochs):
            self.loss.append(0)
            for j in (temp := tqdm(range(num_batches))):
                trace_a, trace_b, ts = self.get_batch()
                self.loss[-1] += self.model.model.train_on_batch([trace_a, ts], trace_b)
                temp.set_description(f"Epoch {i}, batch {j},cl_loss:{-0} , loss: {self.loss[-1]/(j+1)}")
            self.loss[-1] /= num_batches

            #Shuffling the traces and reference labels to not mess with SNR computation in eval
            #self.dataset.x_profiling, self.snr_ref_label = shuffle(self.dataset.x_profiling, self.snr_ref_label)
            self.evaluate(i)
            np.savez(f"{self.result_dir}/metrics.npz", snr_0=np.array(self.max_snr0), snr_t=np.array(self.max_snrt))

        self.model.model.save(f"{self.result_dir}/trained_model.keras")
        np.savez(f"{self.result_dir}/metrics.npz", snr_0=np.array(self.max_snr0), snr_t=np.array(self.max_snrt))


    def forward_noise(self, x, t):
        a = self.t_bar[t]      # base on t
        b = self.t_bar[t + 1]  # image for t + 1
        #print(x.shape)
        noise = np.random.normal(size=x.shape)  # noise mask
        a = a.reshape((-1, 1))
        b = b.reshape((-1, 1))
        sample_a = x * (1 - a) + noise * a
        sample_b = x * (1 - b) + noise * b
        return sample_a, sample_b
    

    def generate_ts(self, num):
        return np.random.randint(0, self.tsteps, size=num)

    def get_batch(self):

        rnd = np.random.randint(0, self.dataset.n_profiling - self.batch_size)
        traces = self.dataset.x_profiling[rnd:rnd + self.batch_size].copy()
        #self.batch_shifts(traces)
        ts = self.generate_ts(self.batch_size)
        trace_a, trace_b = self.forward_noise(traces, ts)
        return trace_a, trace_b, ts

    def select_label(self):
        if self.dataset.name == "ascadv2":
            return self.dataset.share3_profiling[:, self.dataset.target_byte]
        elif self.dataset.name == "aes_hd" or self.dataset.name=="ASCON":
            return self.dataset.profiling_labels
        elif self.dataset.name.__contains__("cswap"):
            return self.dataset.profiling_labels[:, 1]
        elif self.dataset.name == "eshard" or self.dataset.name=="aes_hd_mm":
            print(self.dataset.share2_profiling.shape)
            return self.dataset.share2_profiling[self.dataset.target_byte]
        elif self.dataset.name == "sim":
            return self.dataset.profiling_shares[:, 0]
        return self.dataset.share1_profiling[self.dataset.target_byte]
    
    def evaluate(self, e):
        temp2 = np.array(self.dataset.x_profiling[:20000])
        temp= self.model.model.predict([temp2, np.ones(temp2.shape[0])*0])
        snr =  snr_fast(temp, self.snr_ref_label[:20000])
        snro = snr_fast(temp2, self.snr_ref_label[:20000])

        max0 = np.max(snr)
        temp= self.model.model.predict([temp2, np.ones(temp2.shape[0])*(self.tsteps-1)])
        snrt =  snr_fast(temp, self.snr_ref_label[:20000])
        maxt = np.max(snrt)
        plt.plot(snr, alpha=0.5, label = "Diff_0")
        plt.plot(snro,alpha=0.5,  label="OG")
        plt.plot(snrt, alpha=0.5, label = "Attacs")
        plt.ylabel("SNR")
        plt.xlabel("Sample")
        plt.legend()
        plt.grid()
        plt.savefig(f"{self.result_dir}/snr.png")
        plt.close()
        self.max_snr0.append(max0)
        self.max_snrt.append(maxt)
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("SNR")
        ax1.plot(self.max_snr0, label="Max SNR with t")
        ax1.plot(self.max_snrt, label=f"Max SNR with t={self.tsteps-1}")
        ax1.plot((np.ones(e+1)* self.orig_snr), linestyle='--', label="orig_snr", color='r')
        if not self.pre_noise_snr is None:
            ax1.plot((np.ones(e+1)* self.pre_noise_snr), linestyle='--', label="pre_noise_snr", color='r')


        ax2 = ax1.twinx()
        ax2.set_ylabel("loss")
        ax2.plot(self.loss,linestyle='dotted', label="Loss")
        handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
        fig.legend(handles, labels)
        fig.tight_layout()
        
        plt.savefig(f"{self.result_dir}/max_snr_share_2.png")
        plt.close() 
