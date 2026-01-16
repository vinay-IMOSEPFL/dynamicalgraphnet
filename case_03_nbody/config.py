import os

# Base paths
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "case_03_nbody", "data_321")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "case_03_nbody", "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "case_03_nbody", "results")

# Ensure directories exist
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Settings derived from the notebook and your requirements
MODEL_SETTINGS = {
    "batch_size": 200,
    "epochs": 500,
    "lr": 5e-4,
    "nf": 64,  # Latent size
    "model": "dgn",
    "n_layers": 2,
    "max_testing_samples": 600,
    "max_training_samples": 500,
    "data_dir": DATA_DIR,
    "finite_diff": True,
    
    # Time steps
    "delta_frame": 10,       # Used for finite diff calculation
    "time_step": 1,
    # N-Body specific configuration
    "n_isolated": 3,
    "n_stick": 2,
    "n_hinge": 1,
    
    "results_dir": RESULTS_DIR,
}

SEED = 90
DEVICE_ID = "1"