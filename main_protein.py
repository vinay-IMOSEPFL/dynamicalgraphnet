import os
import argparse
import glob
import re
import torch
import torch.optim as optim
from tqdm import tqdm
from torch_geometric.loader import DataLoader


from case_02_protein.config import MODEL_SETTINGS, SEED, SAVED_MODELS_DIR, DEVICE_ID
from utils.utils import set_seed, evaluate, evaluate_rollout
from case_02_protein.dataset import MDAnalysisDataset, calculate_min_max_edge
from model.model_hist import DynamicsSolver
from utils.trainer import Trainer
#from case_01_human_walk.visualization import visualize_multi_step, create_gif

def find_best_model(model_dir):
    pattern = os.path.join(model_dir, "Val_Loss_*.pth")
    files = glob.glob(pattern)
    if not files: return None
    def extract_loss(fn):
        m = re.search(r'GenLoss_([\d\.]+)mm', fn)
        return float(m.group(1)) if m else float('inf')
    return min(files, key=extract_loss)


def train_one_epoch(trainer, loader, pbar_desc="", smooth=0.98):
    """
    Train for one epoch and return the epoch-average loss.

    Parameters
    ----------
    trainer : your Trainer object
    loader  : DataLoader
    pbar_desc : str                     – label for the progress bar
    smooth     : float in (0,1)         – exponential smoothing for running-avg
    """
    avg = None          # exponential moving average
    true_sum, n = 0.0, 0

    with tqdm(loader, desc=pbar_desc, leave=False, mininterval=0.25) as bar:
        for batch in bar:
            trainer.train(batch)
            loss = float(trainer.loss)

            # running arithmetic mean (for the final return value)
            true_sum += loss
            n += 1

            # running exponential mean (looks nicer on the bar)
            avg = loss if avg is None else (smooth * avg + (1 - smooth) * loss)

            bar.set_postfix({
                'batch': f'{loss:8.3e}',
                'avg':   f'{avg:8.3e}'
            })

    return true_sum / n if n else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'visual'])
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_ID
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Mode: {args.mode}")

    # Data Setup
    ds_train = MDAnalysisDataset('adk', partition='train', tmp_dir=MODEL_SETTINGS["data_dir"],
                                   delta_frame=15, load_cached=True)
    ds_val = MDAnalysisDataset('adk', partition='val', tmp_dir=MODEL_SETTINGS["data_dir"],
                                   delta_frame=15, load_cached=True)
    ds_test = MDAnalysisDataset('adk', partition='test', tmp_dir=MODEL_SETTINGS["data_dir"],
                                   delta_frame=15, load_cached=True)

    train_loader = DataLoader(ds_train, MODEL_SETTINGS["batch_size"], shuffle=True) 
    val_loader = DataLoader(ds_val, MODEL_SETTINGS["batch_size"], shuffle=False)
    test_loader = DataLoader(ds_test, MODEL_SETTINGS["batch_size"], shuffle=False)

    # Stats
    raw_stats = calculate_min_max_edge(train_loader)
    def to_dev(stat):
        return (stat[0].to(device), stat[1].to(device)) if isinstance(stat, tuple) else stat.to(device)
    train_stats = tuple(to_dev(s) for s in raw_stats)

    step_interval = 15

    # Model
    t_step = step_interval * MODEL_SETTINGS["time_step"] if MODEL_SETTINGS["finite_diff"] else step_interval * 1.0
    node_in_f = 5
    edge_in_f = 1
    model = DynamicsSolver(node_in_f, edge_in_f, t_step, train_stats, num_msgs=5, latent_size=128, mlp_layers=MODEL_SETTINGS["n_layers"]).to(device) # num_msgs=4 from cell 22 output/code
    
    optimizer = optim.Adam(model.parameters(), lr=MODEL_SETTINGS["lr"])
    trainer = Trainer(model, optimizer, device, train_stats, step_interval, SAVED_MODELS_DIR)

    if args.mode == 'train':
        print(f"Training for {MODEL_SETTINGS['epochs']} epochs...")
        with tqdm(range(1, MODEL_SETTINGS["epochs"]+1)) as pbar:
            for epoch in pbar:

                avg_train = train_one_epoch(trainer, train_loader,pbar_desc=f"Epoch {epoch}/{MODEL_SETTINGS['epochs']}")
                for batch in train_loader:
                    trainer.train(batch)
                
                # Validation every 5 epochs as per notebook cell 54
                if epoch % 1 == 0:
                    trainer.test(val_loader, mode='val', epoch=epoch)
                    trainer.test(test_loader, mode='test')
                
                pbar.set_postfix({
                    'Loss': f'{trainer.loss:.4e}',
                    'Val': f'{trainer.best_val_loss:.4f} (Ep{trainer.best_epoch})'
                })
        
        best_path = trainer.best_model_path or find_best_model(SAVED_MODELS_DIR)
    elif args.mode=='test':
        best_path = find_best_model(SAVED_MODELS_DIR)
        print(f"Loading {best_path}...")
        model.load_state_dict(torch.load(best_path, map_location=device))
        print('\n EVALUATING...')

        for nstep in [1,2,3,4]:
            
            dataset_eval = HumanDataset(partition='test', max_samples=MODEL_SETTINGS["max_testing_samples"], data_dir=MODEL_SETTINGS["data_dir"],nsteps=nstep)
            
            dataloader_eval = create_dataloaders_from_raw(dataset_eval,200,shuffle=False)
            
            eval_error = evaluate_rollout(dataloader_eval, trainer.model, device, nsteps=nstep)
            
            print(f'loss for rollout {nstep} steps {eval_error}')
        
            print("DONE.")


    elif args.mode=='visual':
        best_path = find_best_model(SAVED_MODELS_DIR)
        print(f"Loading {best_path}...")
        model.load_state_dict(torch.load(best_path, map_location=device))
        print('\n EVALUATING...')
        dataset_eval = HumanDatasetSeq(partition='test', max_samples=MODEL_SETTINGS["max_testing_samples"], data_dir=MODEL_SETTINGS["data_dir"],nsteps=4)
        loader = create_dataloaders_from_raw(dataset_eval,200,shuffle=False)
        visualize_multi_step(
                            loader,
                            MODEL_SETTINGS["results_dir"],
                            trainer.model,
                            device,
                            steps=[1,2,3,4],
                            num_graphs=5
                            )
        create_gif(MODEL_SETTINGS["results_dir"])        

    else:
        print("not a valid mode.")

if __name__ == "__main__":
    main()