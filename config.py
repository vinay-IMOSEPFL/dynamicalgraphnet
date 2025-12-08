import os

# Base paths
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")  # Ensure motion.pkl is here
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Settings derived from the notebook
MODEL_SETTINGS ={
        "batch_size": 100,
        "epochs": 1500,
        "lr": 5e-4,
        "nf": 64,
        "model": "dgn",
        "attention": 0,
        "n_layers":3,
        "max_testing_samples": 600,
        "max_training_samples": 200,
        "data_dir": DATA_DIR,
        "norm_diff": False,
        "weight_decay": 1e-10,
        "tanh": False,
        "learnable": False,
        "finite_diff":True,
        "time_step":1.0,
        "end_time_step": 30.0,
}

SEED = 90
DEVICE_ID = "1" # From notebook: os.environ['CUDA_VISIBLE_DEVICES'] = '1'