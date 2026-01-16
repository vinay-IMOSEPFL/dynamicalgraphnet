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
from model.model import DynamicsSolver
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
                                   delta_frame=MODEL_SETTINGS["delta_frame"], load_cached=True)
    ds_val = MDAnalysisDataset('adk', partition='val', tmp_dir=MODEL_SETTINGS["data_dir"],
                                   delta_frame=MODEL_SETTINGS["delta_frame"], load_cached=True)
    ds_test = MDAnalysisDataset('adk', partition='test', tmp_dir=MODEL_SETTINGS["data_dir"],
                                   delta_frame=MODEL_SETTINGS["delta_frame"], load_cached=True)

    train_loader = DataLoader(ds_train, MODEL_SETTINGS["batch_size"], shuffle=True) 
    val_loader = DataLoader(ds_val, MODEL_SETTINGS["batch_size"], shuffle=False)
    test_loader = DataLoader(ds_test, MODEL_SETTINGS["batch_size"], shuffle=False)

    # Stats
    raw_stats = calculate_min_max_edge(train_loader)
    def to_dev(stat):
        return (stat[0].to(device), stat[1].to(device)) if isinstance(stat, tuple) else stat.to(device)
    train_stats = tuple(to_dev(s) for s in raw_stats)

    step_interval = MODEL_SETTINGS["delta_frame"]

    # Model
    t_step = step_interval * MODEL_SETTINGS["time_step"]
    node_in_f = 4 # one hot encoded not types (atom type N, Cl etc.)
    edge_in_f = 1
    model = DynamicsSolver(node_in_f, edge_in_f, t_step, train_stats, num_msgs=5, latent_size=128, mlp_layers=MODEL_SETTINGS["n_layers"]).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=MODEL_SETTINGS["lr"])
    trainer = Trainer(model, optimizer, device, train_stats, step_interval, SAVED_MODELS_DIR)

    if args.mode == 'train':
        print(f"Training for {MODEL_SETTINGS['epochs']} epochs...")
        with tqdm(range(1, MODEL_SETTINGS["epochs"]+1)) as pbar:
            for epoch in pbar:

                avg_train = train_one_epoch(trainer, train_loader,pbar_desc=f"Epoch {epoch}/{MODEL_SETTINGS['epochs']}")
                for batch in train_loader:
                    trainer.train(batch)
                
                # Validation every 5 epochs
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
            
            dataset_eval = MDAnalysisDataset(
                'adk', 
                partition='test', 
                tmp_dir=MODEL_SETTINGS["data_dir"],
                load_cached=True, delta_frame=MODEL_SETTINGS["delta_frame"],
                nsteps=nstep
                )
            
            
            dataloader_eval = DataLoader(dataset_eval, batch_size=64, shuffle=False)  
            
            eval_error = evaluate_rollout(dataloader_eval, trainer.model, device, nsteps=nstep)
            
            print(f'loss for rollout {nstep} steps {eval_error}')
        
            print("DONE.")


    elif args.mode == 'visual':
        best_path = find_best_model(SAVED_MODELS_DIR)
        print(f"Loading {best_path}...")
        model.load_state_dict(torch.load(best_path, map_location=device))
        print('\n EVALUATING...')
        
        # Load Dataset
        dataset_eval = MDAnalysisDataset('adk', partition='test', tmp_dir=MODEL_SETTINGS["data_dir"],
                                      load_cached=True, delta_frame=MODEL_SETTINGS["delta_frame"], nsteps=4)
        
        indices = random.sample(range(len(dataset_eval)), 5)
        selected_graphs = [dataset_eval[i] for i in indices]
        print(f"Selected indices: {indices}")

        # Define Topology Paths (Adjust as needed)
        psf_path = "case_02_protein/mdanalysis/dataset/adk_equilibrium/adk4AKE.psf"
        dcd_path = "case_02_protein/mdanalysis/dataset/adk_equilibrium/1ake_007-nowater-core-dt240ps.dcd"
        
        # 2) Rollout Loop
        for graph_idx, vis_graph in enumerate(selected_graphs):
            print(f"\n--- Processing Graph {indices[graph_idx]} ---")
            
            # Run Rollout (Returns list of Numpy arrays)
            mask_body = (vis_graph.node_type[:,-1:] != 2) # mask reflected nodes if present

            pos_pred_list = evaluate_rollout_vis(vis_graph, trainer.model, device, nsteps=4)
            
            # Retrieve Ground Truth
            pos_gt_list = vis_graph.gt_seq if hasattr(vis_graph, 'gt_seq') else vis_graph.seq
            
            for t, (pred_np, gt_t) in enumerate(zip(pos_pred_list, pos_gt_list)):
                
                pred_np_body = pred_np[mask_body]
                gt_np_body = gt_t[mask_body]
                
                # === FIX START: Force Conversion to Numpy for File Writing ===
                if torch.is_tensor(pred_np_body):
                    pred_np_body = pred_np_body.detach().cpu().numpy()
                
                if torch.is_tensor(gt_t):
                    gt_np_body = gt_np_body.detach().cpu().numpy()

                # --- B. Calculate Loss (Re-convert to tensor if needed or use existing) ---
                gt_tensor = torch.tensor(gt_np_body)
                pred_tensor = torch.tensor(pred_np_body) # Now safe to convert back for loss
                
                mse_val = F.mse_loss(pred_tensor, gt_tensor).item()
                print(f"Step {t}: MSE = {mse_val:.6f}")

                # --- C. Write PDBs ---
                # Save Ground Truth
                fpath = os.path.join('case_02_protein','results')
                os.makedirs(fpath, exist_ok=True)

                gt_fname = f"graph{indices[graph_idx]}_step{t}_gt.pdb"
                gt_fpath = os.path.join(fpath,gt_fname)
                backbone_to_pdb_file(
                    gt_np_body,
                    psf_path, dcd_path,
                    gt_fpath
                )
                
                # Save Prediction
                pred_fname = f"graph{indices[graph_idx]}_step{t}_pred_mse{mse_val:.4f}.pdb"
                pred_fpath = os.path.join(fpath,pred_fname)
                backbone_to_pdb_file(
                    pred_np_body,
                    psf_path, dcd_path,
                    pred_fpath
                )

    else:
        print("not a valid mode.")

if __name__ == "__main__":
    main()