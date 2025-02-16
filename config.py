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
    "ld", # 7
    "firefly", # 8
    "sd3", # 9
    "tam_trans", # 10
]

initial_n_known = 5

VARIANCE_THRESHOLD = 0.9
VARIANCE_ADAPTIVE_COEFF = 0.1
SIZE_THRESHOLD = 100
SIZE_ADAPTIVE_COEFF = 2
RANDOM_SAMPLES_SIZE = 250
N_TRIALS = 5

FixedTransform = False