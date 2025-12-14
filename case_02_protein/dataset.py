import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data

def augment_graph(graph):
    """
    Augments a single PyG Data object with a global node.
    - Global Pos/Vel: Mean of existing nodes.
    - Global Targets (y_): Mean of existing node targets.
    - Node Type: -1
    - Edge Type: -1
    - Connectivity: Bidirectional (All nodes <-> Global Node)
    """
    # --- 1. Calculate Global Properties (Mean of current graph) ---
    # Using dim=0 to average across all nodes in this single graph
    
    # Inputs
    g_pos = graph.pos.mean(dim=0, keepdim=True)       # (1, 3)
    g_vel = graph.vel.mean(dim=0, keepdim=True)       # (1, 3)
    g_prev = graph.prev_vel.mean(dim=0, keepdim=True) # (1, 3)
    g_type = torch.tensor([[-1.0]])                   # (1, 1)

    # Targets (y_)
    g_y_dv = graph.y_dv.mean(dim=0, keepdim=True)          # (1, 3)
    g_y_dx = graph.y_dx.mean(dim=0, keepdim=True)          # (1, 3)
    g_end_pos = graph.end_pos.mean(dim=0, keepdim=True)    # (1, 3)
    g_end_vel = graph.end_vel.mean(dim=0, keepdim=True)    # (1, 3)

    # --- 2. Append Global Node to Attributes ---
    
    # Inputs
    graph.pos = torch.cat([graph.pos, g_pos], dim=0)
    graph.vel = torch.cat([graph.vel, g_vel], dim=0)
    graph.prev_vel = torch.cat([graph.prev_vel, g_prev], dim=0)
    graph.node_type = torch.cat([graph.node_type, g_type], dim=0)
    
    # Targets
    graph.y_dv = torch.cat([graph.y_dv, g_y_dv], dim=0)
    graph.y_dx = torch.cat([graph.y_dx, g_y_dx], dim=0)
    graph.end_pos = torch.cat([graph.end_pos, g_end_pos], dim=0)
    graph.end_vel = torch.cat([graph.end_vel, g_end_vel], dim=0)

    # Handle Ground Truth Sequence (gt_seq) if it exists
    # gt_seq is typically a list of (N, 3) arrays/tensors
    if hasattr(graph, 'gt_seq') and graph.gt_seq is not None:
        new_gt_seq = []
        for step_data in graph.gt_seq:
            # Check if it's numpy or torch
            if isinstance(step_data, torch.Tensor):
                g_step = step_data.mean(dim=0, keepdim=True)
                step_data = torch.cat([step_data, g_step], dim=0)
            else: # assume numpy
                g_step = np.mean(step_data, axis=0, keepdims=True)
                step_data = np.concatenate([step_data, g_step], axis=0)
            new_gt_seq.append(step_data)
        graph.gt_seq = new_gt_seq

    # --- 3. Create New Edges ---
    num_original_nodes = graph.num_nodes - 1 # We just added 1
    
    # Indices: 0 to N-1 are original, N is global
    original_indices = torch.arange(num_original_nodes, dtype=torch.long)
    global_idx = torch.full((num_original_nodes,), num_original_nodes, dtype=torch.long)
    
    # Edges: Original -> Global
    src1, dst1 = original_indices, global_idx
    # Edges: Global -> Original
    src2, dst2 = global_idx, original_indices
    
    new_src = torch.cat([src1, src2])
    new_dst = torch.cat([dst1, dst2])
    new_edges = torch.stack([new_src, new_dst], dim=0)
    
    # Append to Edge Index
    graph.edge_index = torch.cat([graph.edge_index, new_edges], dim=1)
    
    # --- 4. Create New Edge Attributes (Type -1) ---
    # Detect existing feature dimension (usually 1 based on your code)
    feat_dim = graph.edge_attr.shape[1] 
    new_attr = torch.full((new_edges.shape[1], feat_dim), -1.0)
    
    graph.edge_attr = torch.cat([graph.edge_attr, new_attr], dim=0)
    
    return graph


class MDAnalysisDataset(Dataset):
    """
    Simplified MDAnalysisDataset (cached version
    using raw loc and vel from preprocessed .pkl files.
    """

    def __init__(
        self,
        dataset_name: str,
        partition: str = 'train',
        tmp_dir: str = None,
        delta_frame: int = 1,
        train_valid_test_ratio=None,
        test_rot: bool = False,
        test_trans: bool = False,
        load_cached: bool = True,
        nsteps=1
    ):
        super().__init__()
        assert load_cached, "This version only supports load_cached=True"
        self.delta_frame = delta_frame
        self.delta_total = nsteps*self.delta_frame
        self.nsteps = nsteps
        self.time_step = 1
        self.dataset = dataset_name
        self.partition = partition
        self.test_rot = test_rot
        self.test_trans = test_trans

        # Adjust tmp_dir to point at cached folder
        tmp_dir = os.path.join(tmp_dir, 'adk_backbone_processed')
        self.tmp_dir = tmp_dir

        # Default split ratio if not provided
        if train_valid_test_ratio is None:
            train_valid_test_ratio = [0.6, 0.2, 0.2]
        assert sum(train_valid_test_ratio) <= 1
        self.train_valid_test_ratio = train_valid_test_ratio

        # 1) Load the single '<dataset_name>.pkl' file
        edges_list, self.edge_attr, self.node_feat, self.n_frames = torch.load(
            os.path.join(tmp_dir, f'{dataset_name}.pkl')
        )
        self.edges = torch.stack(edges_list, dim=0)

        # 2) Precompute split-indices over usable frames
        usable = self.n_frames - self.delta_total-self.time_step
        n_train = int(self.train_valid_test_ratio[0] * usable)
        n_valid = int(self.train_valid_test_ratio[1] * usable)
        n_test  = usable - n_train - n_valid
        self.train_start = self.time_step
        self.valid_start = self.time_step + n_train
        self.test_start  = self.time_step + n_train + n_valid

        self.n_train = n_train
        self.n_valid = n_valid
        self.n_test  = n_test

        print(f"[MDAnalysisDataset] partition='{self.partition}'  "
              f"usable={usable}  train={n_train}  valid={n_valid}  test={n_test}")

    def __len__(self):
        if self.partition == 'train':
            return self.n_train
        elif self.partition in ('valid', 'val'):
            return self.n_valid
        else:
            return self.n_test

    def __getitem__(self, i):
        # Determine frame_0 index for this partition
        if self.partition == 'train':
            frame_0 = self.train_start + i
        elif self.partition in ('valid', 'val'):
            frame_0 = self.valid_start + i
        else:
            frame_0 = self.test_start + i
        
        frame_tm1 = frame_0 - self.time_step
        frame_tp1 = frame_0 + self.time_step
        frame_end = frame_0 + self.delta_total

        # Load raw loc and vel from cached files
        loc_0, vel_0, edge_global, edge_global_attr = torch.load(
            os.path.join(self.tmp_dir, f'{self.dataset}_{frame_0}.pkl')
        )
        loc_tp, vel_tp, _, _ = torch.load(
            os.path.join(self.tmp_dir, f'{self.dataset}_{frame_tp1}.pkl')
        )
        loc_end, vel_end, _, _ = torch.load(
            os.path.join(self.tmp_dir, f'{self.dataset}_{frame_end}.pkl')
        )
        loc_tm, vel_tm, _, _ = torch.load(
            os.path.join(self.tmp_dir, f'{self.dataset}_{frame_tm1}.pkl')
        )

        seq = []
        for step in range(self.nsteps + 1):
            idx = frame_0 + step * self.delta_frame
            loc_seq, _, _, _ = torch.load(
            os.path.join(self.tmp_dir, f'{self.dataset}_{idx}.pkl')
        )
            seq.append(loc_seq)

        # Stack global edges
        edge_global = torch.stack(edge_global, dim=0)
        if edge_global_attr.dim() == 1:
            edge_global_attr = edge_global_attr.unsqueeze(-1)

        # Combine local and global edges
        edge_index_combined = torch.cat([self.edges, edge_global], dim=1)

        # Build combined edge_attr = [distance, type_flag]
        ea = self.edge_attr
        if ea.dim() == 1:
            ea = ea.unsqueeze(-1)
        E_l = ea.size(0)
        E_g = edge_global_attr.size(0)

        dist_combined = torch.vstack((ea, edge_global_attr))
        zero_block = torch.zeros(E_l, 1)
        one_block  = torch.ones(E_g,  1)
        type_flag  = torch.vstack((zero_block, one_block))

        senders,receivers = edge_index_combined

        charges = self.node_feat[:,1:2]

        charges_ij = (charges[senders]*charges[receivers]).view(-1,1)


        edge_attr_combined = type_flag#torch.hstack((type_flag, charges_ij))



        # Build PyG Data object
        graph = Data(
            edge_index=edge_index_combined,
            edge_attr=edge_attr_combined,
        )
        graph.node_type = self.node_feat
        graph.pos     = loc_0
        graph.vel     = vel_0
        graph.prev_vel  = vel_tm
        graph.y_dv    = vel_end - vel_0
        graph.y_dx    = loc_end - loc_0
        graph.end_pos   = loc_end
        graph.end_vel   = vel_end
        graph.seq     = seq
        return graph


def calculate_min_max_edge(train_loader):
    """
    Calculate min/max statistics for graph properties using the ConsecutiveGraphDataset.
    
    Parameters:
    ----------
    timestep_dict : dict
        Dictionary containing graphs with keys as timesteps.
    time_step_increment : int, optional
        Time step increment to use (default: 1).
        
    Returns:
    -------
    tuple:
        Min/max statistics for various physical properties.
    """
    # Initialize lists to collect data
    all_edge_dx = []
    all_node_v_t = []
    all_node_v_tm1 = []
    all_node_dv = []
    all_node_dx = []
    
    # Process all valid timesteps
    for batch in train_loader:
        batched_graph = batch
        
        senders,receivers = batched_graph.edge_index

        edge_dx = batched_graph.pos[receivers] - batched_graph.pos[senders]
        
        # Extract positions and velocities
        node_vel_t = batched_graph.vel.float()
        node_vel_tm1 = batched_graph.prev_vel.float()
        
        # Calculate displacements and acceleration changes
        mask_body = (batched_graph.node_type!=2).squeeze()
        node_dv = (batched_graph.y_dv).float()
        node_dx = (batched_graph.y_dx).float()
        
        # Collect data
        all_edge_dx.append(edge_dx)
        all_node_v_t.append(node_vel_t)
        all_node_v_tm1.append(node_vel_tm1)
        all_node_dv.append(node_dv)
        all_node_dx.append(node_dx)
    
    # Concatenate all collected data
    all_edge_dx = torch.cat(all_edge_dx, dim=0)
    all_node_v_t = torch.cat(all_node_v_t, dim=0)
    all_node_v_tm1 = torch.cat(all_node_v_tm1, dim=0)
    all_node_dv= torch.cat(all_node_dv, dim=0)
    all_node_dx = torch.cat(all_node_dx,dim=0)
    
    # Compute norms
    norm_edge_dx = all_edge_dx.norm(dim=1)
    norm_node_v_t = all_node_v_t.norm(dim=1)
    norm_node_v_tm1 = all_node_v_tm1.norm(dim=1)
    norm_node_dv = all_node_dv.norm(dim=1)
    norm_node_dx = all_node_dx.norm(dim=1)
    
    # Compute min and max values of the norms
    min_edge_dx = norm_edge_dx.min()
    max_edge_dx = norm_edge_dx.max()

    min_node_v_t = norm_node_v_t.min()
    max_node_v_t = norm_node_v_t.max()

    min_node_v_tm1 = norm_node_v_tm1.min()
    max_node_v_tm1 = norm_node_v_tm1.max()

    mean_node_dv = norm_node_dv.mean()
    std_node_dv = norm_node_dv.std()

    mean_node_dx = norm_node_dx.mean()
    std_node_dx = norm_node_dx.std()

    # Collect statistics in tuples
    stat_edge_dx = (min_edge_dx, max_edge_dx)
    stat_node_v_t = (min_node_v_t, max_node_v_t)
    stat_node_v_tm1 = (min_node_v_tm1, max_node_v_tm1)
    stat_node_dv = (mean_node_dv, std_node_dv)
    stat_node_dx = (mean_node_dx, std_node_dx)
    
    return stat_edge_dx, stat_node_v_t, stat_node_dv, stat_node_dx