import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed=42):
    """Sets the seed for reproducibility as defined in Cell 1."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_mlp_d(in_size, hidden_size, out_size, num_layers=1, lay_norm=True, use_sigmoid=False, use_softmax=False):
    """
    Args:
        lay_norm (bool): If True, applies LayerNorm to all HIDDEN layers.
        final_lay_norm (bool): If True, applies LayerNorm to the OUTPUT layer. 
                               (Default False: usually better for physics/ResNets).
    """
    if use_sigmoid and use_softmax:
        raise ValueError("Only one of use_sigmoid or use_softmax can be true.")
    
    layers = []
    
    # --- 1. Input Block ---
    layers.append(nn.Linear(in_size, hidden_size))
    if lay_norm:
        layers.append(nn.LayerNorm(hidden_size)) # Internal Norm
    layers.append(nn.ReLU())
    
    # --- 2. Intermediate Hidden Blocks ---
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        if lay_norm:
            layers.append(nn.LayerNorm(hidden_size)) # Internal Norm
        layers.append(nn.ReLU())

    # --- 3. Output Layer ---
    layers.append(nn.Linear(hidden_size, out_size))

    # --- 4. Final Activations ---
    if use_sigmoid:
        layers.append(nn.Sigmoid())
    if use_softmax:
        layers.append(nn.Softmax(dim=-1))

    return nn.Sequential(*layers)


def evaluate(test_loader, model, device, mode='test', plot=False, frequency=1, model_name='PIGNN', experiment_name='Val'):
    with torch.no_grad():
        res = 0.
        res_counter = 0

        for test_batch in test_loader:
            for i in range(1):  # Iterates 10 times: 0 -> predicts 31, ..., 9 -> predicts 40
                if i == 0:
                    test_graph = test_batch.to(device)
                    graph_t0 = test_graph
                    # Clone to ensure ground truth remains unchanged
                    end_pos = test_graph.end_pos.clone()  

                node_dv,node_dx,_= model(graph_t0.detach())
                new_vel =  graph_t0.vel + node_dv
                new_pos =  graph_t0.pos + node_dx

                graph_t0.prev_pos = graph_t0.pos.clone()
                graph_t0.prev_vel = graph_t0.vel.clone()
                graph_t0.pos = new_pos.clone()
                graph_t0.vel = new_vel.clone()

            loss = F.mse_loss(new_pos, end_pos)
            batch_size = test_graph.num_graphs
            res += loss.item() * batch_size
            res_counter += batch_size

        mean_pos_error = res / res_counter
    return mean_pos_error

def evaluate_rollout(test_loader, model, device, nsteps=1):
    model.eval()
    with torch.no_grad():
        res = 0.
        res_counter = 0

        for test_batch in test_loader:
            for i in range(nsteps):  # Iterates 10 times: 0 -> predicts 31, ..., 9 -> predicts 40
                if i == 0:
                    test_graph = test_batch.to(device)
                    graph_t0 = test_graph
                    # Clone to ensure ground truth remains unchanged
                    end_pos = test_graph.end_pos.clone()  

                node_dv,node_dx,_= model(graph_t0.detach())
                new_vel =  graph_t0.vel + node_dv
                new_pos =  graph_t0.pos + node_dx

                graph_t0.prev_pos = graph_t0.pos.clone()
                graph_t0.prev_vel = graph_t0.vel.clone()
                graph_t0.pos = new_pos.clone()
                graph_t0.vel = new_vel.clone()

            loss = F.mse_loss(new_pos, end_pos)
            batch_size = test_graph.num_graphs
            res += loss.item() * batch_size
            res_counter += batch_size

        mean_pos_error = res / res_counter
    return mean_pos_error