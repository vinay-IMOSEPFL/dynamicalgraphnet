import os

# Base paths
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "case_01_human_walk", "data")  # Ensure motion.pkl is here
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "case_01_human_walk", "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "case_01_human_walk", "results")

# Settings derived from the notebook
MODEL_SETTINGS ={
        "batch_size": 100,
        "epochs": 4000,
        "lr": 5e-4,
        "nf": 64,
        "model": "dgn",
        "n_layers":2,
        "max_testing_samples": 600,
        "max_training_samples": 200,
        "data_dir": DATA_DIR,
        "weight_decay": 1e-10,
        "finite_diff":True,
        "time_step":1.0,
        "step_interval": 30.0,
        "results_dir": RESULTS_DIR,
}

SEED = 45
DEVICE_ID = "1" # From notebook: os.environ['CUDA_VISIBLE_DEVICES'] = '1'