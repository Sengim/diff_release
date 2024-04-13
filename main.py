import argparse
from diffusion import *

## Fill these in with desired 
dataset_root_path = "/data1/karayalcins/datasets"
results_root_path = "/data1/karayalcins/datasets/paper_9_gan_results"



def get_arguments():
    parser = argparse.ArgumentParser(add_help=False)

    """ root path for datasets """
    parser.add_argument("-dataset_root_path", "--dataset_root_path", default=dataset_root_path)

    """ root path for results """
    parser.add_argument("-results_root_path", "--results_root_path", default=results_root_path)


    """ dataset_target: name of target dataset (possible values: ascad-variable, ASCAD, dpa_v42, eshard, aes_hd_mm) """
    parser.add_argument("-dataset", "--dataset", default="ASCAD")

    """ dataset_target_dim: number of features (samples) in target dataset """
    parser.add_argument("-dataset_dim", "--dataset_dim", default=700)

    """ n_profiling_target: number of profiling traces from the target dataset """
    parser.add_argument("-n_profiling", "--n_profiling", default=50000)


    """ target_byte_target: key byte index in the target dataset """
    parser.add_argument("-target_byte_target", "--target_byte_target", default=2)

    """ leakage_model: leakage model type (ID or HW) """
    parser.add_argument("-leakage_model", "--leakage_model", default="ID")

    """ batch_size: batch size for training the Diffusion """
    parser.add_argument("-batch_size", "--batch_size", default=200)
    
    """ epochs: number of training epochs for Diffusion """
    parser.add_argument("-epochs", "--epochs", default=200)

    """ tsteps: number of steps for Diffusion """
    parser.add_argument("-tsteps", "--tsteps", default=16)

    """ add_noise: noise to add to original traces"""
    parser.add_argument("-add_noise", "--add_noise", default=0.0)
    
    """ learning_rate: initial learning rate """
    parser.add_argument("-lr", "--lr", default=0.001)
    
    """n_inf_feat: The number of leaky sample points per share for simulations"""
    parser.add_argument("-n_inf_feat", "--n_inf_feat", default=5)

    return parser.parse_args()


if __name__ == "__main__":
    arg_list = get_arguments()

    arguments = {
        "dataset_root_path": arg_list.dataset_root_path,
        "results_root_path": arg_list.results_root_path,
        "dataset": arg_list.dataset_target,
        "dataset_dim": int(arg_list.dataset_target_dim),
        "n_profiling": int(arg_list.n_profiling_target),
        "target_byte": int(arg_list.target_byte_target),
        "leakage_model": arg_list.leakage_model,
        "epochs": int(arg_list.epochs),
        "batch_size": int(arg_list.batch_size),
        "tsteps": int(arg_list.tsteps), 
        "lr": float(arg_list.lr),
        "add_noise": float(arg_list.add_noise),
        "n_inf_feat": int(arg_list.n_inf_feat),
    }

    dif = Diffussion(args=arguments)
    dif.train_cgan()