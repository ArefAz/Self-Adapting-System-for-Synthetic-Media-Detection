prefix = "fsd"
# prefix = "transformed2"

data_sources = [
    "real", # 0
    "stylegan-", # 1
    "stylegan2", # 2
    "stylegan3", # 3

    "progan", # 4
    "dalle3", # 5 
    "mjv6", # 6
    "firefly", # 7
    "sd3", # 8
    "tam_trans", # 9
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
    "pl_patience": 0,
    "es_patience": 3,
}

FixedTransform = False
use_autoencoder = True
refine_factor = 1.5