import os
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

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

# class HumanDataset(torch.utils.data.Dataset):
#     def __init__(self, partition='train', max_samples=600, data_dir='', nsteps=1,augment=True):
#         self.partition = partition
#         self.data_dir = data_dir
#         self.nsteps = nsteps
#         self.augment = augment
#         # Define the step interval (delta t)
#         self.step_interval = 30  # e.g., 30 frames
#         self.time_step = 1
#         # --- load raw data --------------------------------------
#         with open(os.path.join(data_dir, 'motion.pkl'), 'rb') as f:
#             edges, X = pkl.load(f)

#         # your smoothing / central_diff code here...
#         Ps, Vs, As = self.central_diff(X)

#         # trial IDs must match exactly
#         train_case_id = [20,1,17,13,14,9,4,2,7,5,16]
#         val_case_id   = [3,8,11,12,15,18]
#         test_case_id  = [6,19,21,0,22,10]

#         # --- load or create competitor splits (fixed for central_diff) ----------
#         split_path = os.path.join(data_dir, f'split_n{self.nsteps}.pkl')
#         try:
#             with open(split_path, 'rb') as f:
#                 train_mapping, val_mapping, test_mapping = pkl.load(f)
#                 print("Loaded competitor split!")
#         except FileNotFoundError:
#             print("Generating competitor split…")

#             def make_map(case_ids):
#                 mapping = {}
#                 for i in case_ids:
#                     core_len = Ps[i].shape[0]                    # <<— use length after central_diff
#                     max_start_index = core_len - self.nsteps*self.step_interval - 1
#                     min_start_index = self.step_interval
#                     if max_start_index< min_start_index:
#                         raise ValueError(f"Trial {i} too short for look-ahead of {self.nsteps} steps.")
#                     # competitor caps at 300
#                     itv = min(300, max_start_idx - min_start_idx + 1)               # +1 because j in [0..safe_max]
#                     pool =np.arange(min_start_idx, min_start_idx + pool_size)                      # j ∈ [0..itv-1]
#                     mapping[i] = np.random.choice(pool, size=min(100, len(pool)), replace=False)
#                 return mapping

#             train_mapping = make_map(train_case_id)
#             val_mapping   = make_map(val_case_id)
#             test_mapping  = make_map(test_case_id)

#             with open(split_path, 'wb') as f:
#                 pkl.dump((train_mapping, val_mapping, test_mapping), f)
#             print("Saved competitor split!")

#         # pick the mapping you need
#         if   partition == 'train': mapping = train_mapping
#         elif partition == 'val'  : mapping = val_mapping
#         elif partition == 'test' : mapping = test_mapping
#         else: raise ValueError(f"Unknown partition {partition!r}")

#         # now proceed exactly as before, using `mapping` instead of your make_mapping
#         each_len = max_samples // len(mapping)
#         in_graphs = []
#         for i, pool in mapping.items():
#             for j in pool[:each_len]:
#                 # note: they use delta_frame; you have nsteps*30, so this is identical
#                 cur_x_t   = Ps[i][j]
#                 cur_v_t   = Vs[i][j]
#                 cur_v_tm1 = Vs[i][j-self.step_interval]
#                 y_pos_end = Ps[i][j + self.nsteps*self.step_interval]
#                 y_vel_end = Vs[i][j + self.nsteps*self.step_interval]

#                 y_dv      = y_vel_end - cur_v_t
#                 y_dx      = y_pos_end - cur_x_t


#                 in_graphs.append(self.create_in_graph(
#                     edges,
#                     x=(cur_x_t, cur_v_t, cur_v_tm1),
#                     y=(y_dv, y_dx, y_pos_end, y_vel_end)
#                 ))

#         self.in_graphs = in_graphs
#         print(f"[HumanDataset:{partition}] built {len(in_graphs)} samples")

#     def central_diff(self, Xs, dt: float = 1.0, window_length: int = 41):
#         Ps, Vs, As = [], [], []
#         for x in Xs:
#             v      = (x[2:] - x[:-2]) / (2*dt)
#             a      = (x[2:] - 2*x[1:-1] + x[:-2]) / (dt**2)
#             p      = x[1:-1]                      # align to v,a
#             Ps.append(p)
#             Vs.append(v)
#             As.append(a)
#         return Ps, Vs, As

        
#     def get_foot_nodes(self, nodes):
#         foot_indices = np.argsort(nodes[:,1])[:4]
#         foot_pos = nodes[foot_indices]
#         return foot_pos, foot_indices
    
#     def reflected_nodes(self, nodes, z0=0, epsilon=1e-3):
#         reflected = nodes.copy()
#         reflected[:,1] = 2*z0 - nodes[:,1] - epsilon
#         distances = reflected[:,1] - nodes[:,1]
#         return reflected, distances
    
#     def find_min(self, nodes):
#         return np.min(nodes, axis=0)
    

#     def create_edges(self, N, edges):
#         atom_edges = torch.zeros(N, N).int()
#         for edge in edges:
#             atom_edges[edge[0], edge[1]] = 1
#             atom_edges[edge[1], edge[0]] = 1

#         atom_edges2 = atom_edges @ atom_edges
#         self.atom_edge = atom_edges
#         self.atom_edge2 = atom_edges2
#         edge_attr = []
#         # Initialize edges and edge_attributes
#         rows, cols = [], []
#         for i in range(N):
#             for j in range(N):
#                 if i != j:
#                     if atom_edges[i][j]:
#                         rows.append(i)
#                         cols.append(j)
#                         edge_attr.append([1])
#                         assert not atom_edges2[i][j]
#                     if atom_edges2[i][j]:
#                         rows.append(i)
#                         cols.append(j)
#                         edge_attr.append([2])
#                         assert not atom_edges[i][j]

#         edges = [rows, cols] 
#         edge_attr = torch.Tensor(np.array(edge_attr))  # [edge, 3]
#         edge_idx =torch.tensor(edges, dtype=torch.long)  # [2, M]   
#         return edge_idx,edge_attr     
    
    
#     def create_in_graph(self, edges,x,y):
#         pos_t, vel_t, vel_tm1 = x
#         y_dv,y_dx,y_pos_end,y_vel_end = y

#         edge_idx,edge_attr = self.create_edges(pos_t.shape[0], edges)

#         # # Get the ground node
#         # z0_t = self.find_min(pos_t)[1]
#         # z0_end = self.find_min(y_end)[1]
#         # # Center the y-positions around z0 for input and target
#         # pos_t -= np.array([0, z0_t, 0]) 
#         # y_end -= np.array([0, z0_end, 0])

#         # Get the foot node positions and indices
#         foot_nodes_positions, foot_nodes_indices = self.get_foot_nodes(pos_t)
#         #foot_nodes_reflected, foot_distances = self.reflected_nodes(foot_nodes_positions,z0=0.0)
        
#         # current_largest_node_index = pos_t.shape[0]
#         # reflected_nodes_indices = []
#         # for reflected_node in range(foot_nodes_indices.shape[0]):
#         #     reflected_node_index = current_largest_node_index
#         #     current_largest_node_index += 1
#         #     reflected_nodes_indices.append(reflected_node_index)
        
        
#         # # Set lists to torch tensors
#         # reflected_nodes_indices = torch.tensor(reflected_nodes_indices)
#         foot_nodes_indices = torch.tensor(foot_nodes_indices)
#         pos_t = torch.tensor(pos_t)
#         vel_t = torch.tensor(vel_t)
#         vel_tm1 = torch.tensor(vel_tm1)

#         y_dv = torch.tensor(y_dv)
#         y_dx = torch.tensor(y_dx)
#         y_pos_end = torch.tensor(y_pos_end)
#         y_vel_end = torch.tensor(y_vel_end)
        
        
#         # foot_nodes_reflected = torch.tensor(foot_nodes_reflected)
        
#         # Set the node type of feet to one
#         node_type = torch.ones(pos_t.shape[0],1)
#         node_type[foot_nodes_indices] = 0
#         # # Make reflected nodes of type 2
#         # new_node_type = torch.vstack((node_type,2*torch.ones_like(reflected_nodes_indices).unsqueeze(1))) 
        
#         # New bi-dir edge indexes
#         # new_edges_ref = torch.hstack((foot_nodes_indices.unsqueeze(1), reflected_nodes_indices.unsqueeze(1))) # connect foot edges to their reflections
#         # new_edges_ref = new_edges_ref.t()  # now [2, M]
#         # rev_new_edges_ref = new_edges_ref.flip(0)  # reverse the order to match edge index format
#         # new_edges_bidir_ref = torch.cat((new_edges_ref, rev_new_edges_ref), dim=1)  # add reverse edges
#         # new_edge_index = torch.cat([edge_idx, new_edges_bidir_ref], dim=1) # add new edges to the graph edge index
#         # s,r = new_edge_index

#         # we add the 1 as edge attr for these edges as they are 1 hop
#         # new_edge_attr = torch.vstack((edge_attr, torch.ones((new_edges_bidir_ref.shape[1], 1))))  # add new edge attributes
#         # for differentiating reflected edges we use another features i.e. type_sender*type_receiver
#         # new_edge_attr = torch.hstack((new_edge_attr,
#         #                               new_node_type[s]*new_node_type[r]))
#         # new_pos_t = torch.vstack((pos_t, foot_nodes_reflected))
#         # new_vel_t = torch.vstack((vel_t,torch.zeros_like(foot_nodes_reflected)))
#         # new_vel_tm1 = torch.vstack((vel_tm1,torch.zeros_like(foot_nodes_reflected)))

        
#         # in_graph = Data(x=new_pos_t,edge_index=new_edge_index,edge_attr=new_edge_attr)
#         # in_graph.node_vel_t = new_vel_t
#         # in_graph.node_vel_tm1 = new_vel_tm1
#         # in_graph.y_dv = y_dv
#         # in_graph.y_dx = y_dx
#         # in_graph.y_end = y_end
#         # in_graph.node_type = new_node_type

#         in_graph = Data(edge_index=edge_idx, edge_attr=edge_attr)
#         in_graph.pos = pos_t
#         in_graph.vel = vel_t
#         in_graph.prev_vel = vel_tm1
#         in_graph.y_dv = y_dv
#         in_graph.y_dx = y_dx
#         in_graph.end_pos = y_pos_end
#         in_graph.end_vel = y_vel_end
#         in_graph.node_type = node_type
#         # --- GLOBAL NODE AUGMENTATION ---
#         if self.augment:
#             in_graph = augment_graph(in_graph)
#         # --------------------------------        
        
#         return in_graph     
        
        
#     def __len__(self):
#         return len(self.in_graphs)
    
    
#     def __getitem__(self, index):
#         return self.in_graphs[index]
    
    

class HumanDatasetSeq(torch.utils.data.Dataset):
    def __init__(self, partition='train', max_samples=600, data_dir='', nsteps=1,augment=True):
        self.partition = partition
        self.data_dir = data_dir
        self.nsteps = nsteps
        self.augment = augment

        self.step_interval = 30
        self.time_step = 1

        # --- load raw data --------------------------------------
        with open(os.path.join(data_dir, 'motion.pkl'), 'rb') as f:
            edges, X = pkl.load(f)

        # your smoothing / central_diff code here...
        Ps, Vs, As = self.central_diff(X)

        # trial IDs must match exactly
        train_case_id = [20,1,17,13,14,9,4,2,7,5,16]
        val_case_id   = [3,8,11,12,15,18]
        test_case_id  = [6,19,21,0,22,10]

        # --- load or create competitor splits (fixed for central_diff) ----------
        split_path = os.path.join(data_dir, f'split_n{self.nsteps}.pkl')
        try:
            with open(split_path, 'rb') as f:
                train_mapping, val_mapping, test_mapping = pkl.load(f)
                print("Loaded competitor split!")
        except FileNotFoundError:
            print("Generating competitor split…")

            def make_map(case_ids):
                mapping = {}
                for i in case_ids:
                    core_len = Ps[i].shape[0]                    # <<— use length after central_diff
                    max_start_index = core_len - self.nsteps*self.step_interval - 1
                    min_start_index = self.step_interval
                    if max_start_index< min_start_index:
                        raise ValueError(f"Trial {i} too short for look-ahead of {self.nsteps} steps.")
                    # competitor caps at 300
                    itv = min(300, max_start_idx - min_start_idx + 1)               # +1 because j in [0..safe_max]
                    pool =np.arange(min_start_idx, min_start_idx + pool_size)                      # j ∈ [0..itv-1]
                    mapping[i] = np.random.choice(pool, size=min(100, len(pool)), replace=False)
                return mapping

            train_mapping = make_map(train_case_id)
            val_mapping   = make_map(val_case_id)
            test_mapping  = make_map(test_case_id)

            with open(split_path, 'wb') as f:
                pkl.dump((train_mapping, val_mapping, test_mapping), f)
            print("Saved competitor split!")

        # pick the mapping you need
        if   partition == 'train': mapping = train_mapping
        elif partition == 'val'  : mapping = val_mapping
        elif partition == 'test' : mapping = test_mapping
        else: raise ValueError(f"Unknown partition {partition!r}")

        # now proceed exactly as before, using `mapping` instead of your make_mapping
        each_len = max_samples // len(mapping)
        in_graphs = []
        for i, pool in mapping.items():
            for j in pool[:each_len]:
                # note: they use delta_frame; you have nsteps*30, so this is identical
                cur_x_t   = Ps[i][j]
                cur_v_t   = Vs[i][j]
                cur_v_tm1 = Vs[i][j-self.step_interval]
                y_pos_end = Ps[i][j + self.nsteps*self.step_interval]
                y_vel_end = Vs[i][j + self.nsteps*self.step_interval]

                y_dv      = y_vel_end - cur_v_t
                y_dx      = y_pos_end - cur_x_t
                gt_seq = [ Ps[i][j + k*self.step_interval] for k in range(self.nsteps+1) ]   # list of (31,3) arrays


                in_graphs.append(self.create_in_graph(
                    edges,
                    x=(cur_x_t, cur_v_t, cur_v_tm1),
                    y=(y_dv, y_dx, y_pos_end, y_vel_end),
                    gt_seq = gt_seq
                ))

        self.in_graphs = in_graphs
        print(f"[HumanDataset:{partition}] built {len(in_graphs)} samples")

    def central_diff(self, Xs, dt: float = 1.0, window_length: int = 41):
        Ps, Vs, As = [], [], []
        for x in Xs:
            v      = (x[2:] - x[:-2]) / (2*dt)
            a      = (x[2:] - 2*x[1:-1] + x[:-2]) / (dt**2)
            p      = x[1:-1]                      # align to v,a
            Ps.append(p)
            Vs.append(v)
            As.append(a)
        return Ps, Vs, As

        
    def get_foot_nodes(self, nodes):
        foot_indices = np.argsort(nodes[:,1])[:6]
        foot_pos = nodes[foot_indices]
        return foot_pos, foot_indices
    
    def reflected_nodes(self, nodes, z0=0, epsilon=1e-3):
        reflected = nodes.copy()
        reflected[:,1] = 2*z0 - nodes[:,1] - epsilon
        distances = reflected[:,1] - nodes[:,1]
        return reflected, distances
    
    def find_min(self, nodes):
        return np.min(nodes, axis=0)
    

    def create_edges(self, N, edges):
        atom_edges = torch.zeros(N, N).int()
        for edge in edges:
            atom_edges[edge[0], edge[1]] = 1
            atom_edges[edge[1], edge[0]] = 1

        atom_edges2 = atom_edges @ atom_edges
        self.atom_edge = atom_edges
        #self.atom_edge2 = atom_edges2
        edge_attr = []
        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(N):
            for j in range(N):
                if i != j:
                    if atom_edges[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([1])
                        #assert not atom_edges2[i][j]
                    # if atom_edges2[i][j]:
                    #     rows.append(i)
                    #     cols.append(j)
                    #     edge_attr.append([2])
                    #     assert not atom_edges[i][j]

        edges = [rows, cols] 
        edge_attr = torch.Tensor(np.array(edge_attr))  # [edge, 3]
        edge_idx =torch.tensor(edges, dtype=torch.long)  # [2, M]   
        return edge_idx,edge_attr     
    
    
    def create_in_graph(self, edges,x,y,gt_seq):
        pos_t, vel_t, vel_tm1 = x
        y_dv,y_dx,y_pos_end,y_vel_end = y

        edge_idx,edge_attr = self.create_edges(pos_t.shape[0], edges)

        # Get the ground node
        #z0_t = self.find_min(pos_t)[1]
        #z0_end = self.find_min(y_end)[1]
        # # Center the y-positions around z0 for input and target
        # pos_t -= np.array([0, z0_t, 0]) 
        # y_end -= np.array([0, z0_end, 0])

        # Get the foot node positions and indices
        foot_nodes_positions, foot_nodes_indices = self.get_foot_nodes(pos_t)
        foot_nodes_reflected, foot_distances = self.reflected_nodes(foot_nodes_positions,z0=0.0)
        
        # current_largest_node_index = pos_t.shape[0]
        # reflected_nodes_indices = []
        # for reflected_node in range(foot_nodes_indices.shape[0]):
        #     reflected_node_index = current_largest_node_index
        #     current_largest_node_index += 1
        #     reflected_nodes_indices.append(reflected_node_index)
        
        
        # # Set lists to torch tensors
        # reflected_nodes_indices = torch.tensor(reflected_nodes_indices)
        # foot_nodes_indices = torch.tensor(foot_nodes_indices)
        pos_t = torch.tensor(pos_t)
        vel_t = torch.tensor(vel_t)
        vel_tm1 = torch.tensor(vel_tm1)

        y_dv = torch.tensor(y_dv)
        y_dx = torch.tensor(y_dx)
        y_pos_end = torch.tensor(y_pos_end)
        y_vel_end = torch.tensor(y_vel_end)
        
        
        # foot_nodes_reflected = torch.tensor(foot_nodes_reflected)
        
        # Set the node type of feet to one
        node_type =  torch.ones(pos_t.shape[0],1)
        #node_type[foot_nodes_indices] = 0
        # # Make reflected nodes of type 2
        # new_node_type = torch.vstack((node_type,2*torch.ones_like(reflected_nodes_indices).unsqueeze(1))) 
        
        # New bi-dir edge indexes
        # new_edges_ref = torch.hstack((foot_nodes_indices.unsqueeze(1), reflected_nodes_indices.unsqueeze(1))) # connect foot edges to their reflections
        # new_edges_ref = new_edges_ref.t()  # now [2, M]
        # rev_new_edges_ref = new_edges_ref.flip(0)  # reverse the order to match edge index format
        # new_edges_bidir_ref = torch.cat((new_edges_ref, rev_new_edges_ref), dim=1)  # add reverse edges
        # new_edge_index = torch.cat([edge_idx, new_edges_bidir_ref], dim=1) # add new edges to the graph edge index
        # s,r = new_edge_index

        # we add the 1 as edge attr for these edges as they are 1 hop
        # new_edge_attr = torch.vstack((edge_attr, torch.ones((new_edges_bidir_ref.shape[1], 1))))  # add new edge attributes
        # for differentiating reflected edges we use another features i.e. type_sender*type_receiver
        # new_edge_attr = torch.hstack((new_edge_attr,
        #                               new_node_type[s]*new_node_type[r]))
        # new_pos_t = torch.vstack((pos_t, foot_nodes_reflected))
        # new_vel_t = torch.vstack((vel_t,torch.zeros_like(foot_nodes_reflected)))
        # new_vel_tm1 = torch.vstack((vel_tm1,torch.zeros_like(foot_nodes_reflected)))

        
        # in_graph = Data(x=new_pos_t,edge_index=new_edge_index,edge_attr=new_edge_attr)
        # in_graph.node_vel_t = new_vel_t
        # in_graph.node_vel_tm1 = new_vel_tm1
        # in_graph.y_dv = y_dv
        # in_graph.y_dx = y_dx
        # in_graph.y_end = y_end
        # in_graph.node_type = new_node_type

        in_graph = Data(edge_index=edge_idx, edge_attr=edge_attr)
        in_graph.pos = pos_t
        in_graph.vel = vel_t
        in_graph.prev_vel = vel_tm1
        in_graph.y_dv = y_dv
        in_graph.y_dx = y_dx
        in_graph.end_pos = y_pos_end
        in_graph.end_vel = y_vel_end
        in_graph.node_type = node_type
        in_graph.gt_seq = gt_seq

        if self.augment:
            in_graph = augment_graph(in_graph)
        # --------------------------------             
        
        return in_graph     
        
        
    def __len__(self):
        return len(self.in_graphs)
    
    
    def __getitem__(self, index):
        return self.in_graphs[index]

# --- Dataloader Helpers ---
class GraphFromRawDataset(Dataset):
    def __init__(self, raw_dataset):
        self.raw_dataset = raw_dataset
    def __len__(self):
        return len(self.raw_dataset)
    def __getitem__(self, idx):
        return self.raw_dataset[idx]

def create_dataloaders(dataset, M, shuffle=True):
    dataset_wrapped = GraphFromRawDataset(dataset)
    return DataLoader(dataset_wrapped, batch_size=M, shuffle=shuffle, collate_fn=Batch.from_data_list)

def create_dataloaders_from_raw(dataset, M, shuffle=True):
    return create_dataloaders(dataset, M, shuffle)

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

def move_train_stats_to_device(train_stats, device):
    def move(stat):
        if len(stat) == 2:
            return stat[0].to(device), stat[1].to(device)
        return stat.to(device)
    return tuple(move(s) for s in train_stats)

