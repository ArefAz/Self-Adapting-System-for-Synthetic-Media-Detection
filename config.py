prefix = "fsd"
# prefix = "transformed2"

data_sources = [
    "real", # 0
    "stylegan-", # 1
    "stylegan2", # 2
    "stylegan3", # 3

    # "self-adapt-glide", # 12
    # "self-adapt-mjv5", # 12
    # "self-adapt-porjected_gan", # 12
    # "self-adapt-sd14", # 12
    # "self-adapt-sd15", # 12
    # "self-adapt-sdxl", # 12

    "progan", # 4
    "dalle3", # 5 
    "mjv6", # 6
    "firefly", # 7
    "sd3", # 8
    "tam_trans", # 9
    # "self-adapt-gd", # 10
    # "self-adapt-gigagan", # 11

]

initial_n_known = 4

VARIANCE_THRESHOLD = 0.9
VARIANCE_ADAPTIVE_COEFF = 0.1
SIZE_THRESHOLD = 100
SIZE_ADAPTIVE_COEFF = 2
RANDOM_SAMPLES_SIZE = 500
N_TRIALS = 5

training_kwargs = {
    "num_epochs": 50,
    "alpha": 1.0,
    "beta": 5.0,
    "batch_size": 64,
    "lr": 2.5e-5,
    "ft_lr_factor": 1.0,
    "pl_patience": 0,
    "es_patience": 3,
    "weight_decay": 1e-2,
    "top_k": 3,
    "device": "cuda",
    "n_components": 5,
    "cov_type": "tied",
}

random_seed = 42
use_autoencoder = True
do_nothing = False
do_retrain = False
refine_factor = 1.0
ood_threshold_factor = 1.2
min_ood_tpr = 0.85