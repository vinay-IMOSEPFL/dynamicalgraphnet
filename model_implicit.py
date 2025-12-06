import torch
import torch.nn as nn
from utils import build_mlp_d
from config import MODEL_SETTINGS

class RefFrameCalc(nn.Module):
    def __init__(self):
        super(RefFrameCalc, self).__init__()
        self.epsilon = 1e-6

    def forward(self, edge_index, senders_pos, receivers_pos, senders_vel, receivers_vel):
        # Calculate relative position (Edge vector)
        rel_pos = receivers_pos - senders_pos
        dist = rel_pos.norm(dim=1, keepdim=True).clamp(min=self.epsilon)
        vector_a = rel_pos / dist

        # Preliminary vectors
        # 1. Relative velocity cross edge vector
        # 2. Sum of velocities
        diff_vel = receivers_vel - senders_vel
        sum_vel = senders_vel + receivers_vel
        

        # Helper for normalizing
        def normalize(tensor):
            return tensor / tensor.norm(dim=1, keepdim=True).clamp(min=self.epsilon)

        b_a = normalize(torch.cross(diff_vel, vector_a, dim=1))
        b_c = normalize(sum_vel)

        # Combine to form b
        b = b_a + b_c 

        # Gram-Schmidt like orthogonalization
        # Project b onto a
        b_prl_dot = (b * vector_a).sum(dim=1, keepdim=True)
        b_prl = b_prl_dot * vector_a
        b_prp = b - b_prl

        # vector_b is perpendicular to a
        vector_b = normalize(torch.cross(b_prp, vector_a, dim=1))
        
        # vector_c is perpendicular to a and b
        vector_c = normalize(torch.cross(b_prl, vector_b, dim=1))
   
        return vector_a, vector_b, vector_c

class NodeEncoder(nn.Module):
    def __init__(self, latent_size):
        super(NodeEncoder, self).__init__()
        self.node_encoder = build_mlp_d(2, latent_size, latent_size, num_layers=MODEL_SETTINGS['n_layers'], lay_norm=True)
    
    def forward(self, node_scalar_feat):
        return self.node_encoder(node_scalar_feat)

class InteractionEncoder(nn.Module):
    """Message passing with optimized projections."""

    def __init__(self, latent_size):
        super(InteractionEncoder, self).__init__()
        self.edge_feat_encoder = build_mlp_d(6, latent_size, latent_size, num_layers=MODEL_SETTINGS['n_layers'], lay_norm=True)
        self.edge_encoder = build_mlp_d(2, latent_size, latent_size, num_layers=MODEL_SETTINGS['n_layers'], lay_norm=True)
        self.interaction_encoder = build_mlp_d(3*latent_size, latent_size, latent_size, num_layers=MODEL_SETTINGS['n_layers'], lay_norm=True)

    def forward(self, edge_index, edge_dx_, edge_attr, vector_a, vector_b, vector_c,
                senders_v_t_, senders_v_tm1_, 
                receivers_v_t_, receivers_v_tm1_,
                node_latent):
        
        senders, receivers = edge_index

        # --- Vectorized Projection ---
        # Stack basis vectors into a Rotation Matrix R = [a, b, c] of shape (E, 3, 3)
        # We want to project vectors v onto this basis: v_local = [v.a, v.b, v.c]
        # This is equivalent to v_local = v @ R or R^T @ v depending on layout.
        # Since our vectors are (E, 3), we can do: bmm(Basis_transpose, v.unsqueeze(-1))
        
        # Basis shape: (E, 3, 3). rows are a, b, c. 
        # Actually, stack dim=1 gives [a, b, c] as columns if we view it as matrix, 
        # but here we want rows for dot product.
        # Let's stack them such that index 1 is the basis index:
        basis = torch.stack([vector_a, vector_b, vector_c], dim=1) # (E, 3, 3)

        def project(v):
            # v: (E, 3) -> (E, 3, 1)
            # basis: (E, 3, 3)
            # result: (E, 3, 1) -> (E, 3)
            # This computes [row1.v, row2.v, row3.v] -> [v.a, v.b, v.c]
            return torch.bmm(basis, v.unsqueeze(-1)).squeeze(-1)
        
        # Project senders
        s_vt_proj   = project(senders_v_t_)
        s_vtm1_proj = project(senders_v_tm1_)

        # Project receivers (negated as per original code logic: -vector_a, etc.)
        # Note: Original code did v . (-a), v . (-b). This is equivalent to -(v . a).
        r_vt_proj   = -project(receivers_v_t_)
        r_vtm1_proj = -project(receivers_v_tm1_)

        # Concatenate features (9 features per node per edge)
        senders_features = torch.cat([s_vt_proj, s_vtm1_proj], dim=1)
        receivers_features = torch.cat([r_vt_proj, r_vtm1_proj], dim=1)
        
        # Edge encodings
        edge_dx_norm = edge_dx_.norm(dim=1, keepdim=True)
        edge_latent = self.edge_encoder(torch.cat((edge_dx_norm, edge_attr), dim=1))

        # Encode features
        senders_latent = self.edge_feat_encoder(senders_features)
        receivers_latent = self.edge_feat_encoder(receivers_features)

        # Message Passing
        # Note: Summing latents before concat is efficient
        node_sum = node_latent[senders] + node_latent[receivers]
        msg_input = torch.cat((senders_latent + receivers_latent, node_sum, edge_latent), dim=1)
        
        return self.interaction_encoder(msg_input)

class InteractionDecoder(torch.nn.Module):
    def __init__(self, latent_size=128):
        super(InteractionDecoder, self).__init__()
        self.i1_decoder = build_mlp_d(latent_size, latent_size, 3, num_layers=MODEL_SETTINGS['n_layers'], lay_norm=False)

    def forward(self, edge_index, senders_pos, receivers_pos, vector_a, vector_b, vector_c, interaction_latent, node_latent):
        senders, receivers = edge_index
        
        # Decode coefficients
        coeff_f = self.i1_decoder(interaction_latent)
        
        # Reconstruct forces in global frame
        # Linear combination: c0*a + c1*b + c2*c
        fij = (coeff_f[:, 0:1] * vector_a + 
               coeff_f[:, 1:2] * vector_b + 
               coeff_f[:, 2:3] * vector_c) # Fixed slicing for clarity
        
        return fij

class NodeResidualDecoder(torch.nn.Module):
    def __init__(self, latent_size=128):
        super(NodeResidualDecoder, self).__init__()
        self.m_decoder = build_mlp_d(latent_size, latent_size, 1, num_layers=MODEL_SETTINGS['n_layers'], lay_norm=False)
        self.fext_decoder = build_mlp_d(latent_size, latent_size, 3, num_layers=MODEL_SETTINGS['n_layers'], lay_norm=False)

    def forward(self, edge_index, node_latent, fij, node_acc):
        senders, receivers = edge_index   
        num_nodes = node_latent.shape[0]
        
        # Decode physical properties
        m = self.m_decoder(node_latent)
        fext = self.fext_decoder(node_latent)
        
        # Aggregate Forces and Torques
        # Use new_zeros to ensure device compatibility automatically
        out_fij = node_latent.new_zeros((num_nodes, 3))

        
        out_fij.index_add_(0, receivers, fij)

        # Compute accelerations
        residual = m * node_acc + out_fij - fext

        return residual

class Scaler(torch.nn.Module):
    def __init__(self):
        super(Scaler, self).__init__()

    def forward(self, senders_v_t, senders_v_tm1, receivers_v_t, receivers_v_tm1, edge_dx, train_stats):
        stat_edge_dx, stat_node_v_t, _, _ = train_stats
        
        # Use detach on stats to ensure no gradients flow back to stats (redundant but safe)
        v_scale = stat_node_v_t[1].detach() + 1e-8
        
        senders_v_t_ = senders_v_t / v_scale
        senders_v_tm1_ = senders_v_tm1 / v_scale
        receivers_v_t_ = receivers_v_t / v_scale
        receivers_v_tm1_ = receivers_v_tm1 / v_scale
        
        norm_edge_dx = edge_dx.norm(dim=1, keepdim=True)
        # Avoid division by zero
        safe_norm = norm_edge_dx + 1e-8
        
        min_stat, max_stat = stat_edge_dx
        scale_denom = (max_stat - min_stat).detach() + 1e-8
        
        # Scale magnitude, preserve direction
        scaled_mag = (norm_edge_dx - min_stat.detach()) / scale_denom
        edge_dx_ = scaled_mag * (edge_dx / safe_norm)
        
        return senders_v_t_, senders_v_tm1_, receivers_v_t_, receivers_v_tm1_, edge_dx_.detach()

class Interaction_Block(torch.nn.Module):
    def __init__(self, latent_size):
        super(Interaction_Block, self).__init__()
        self.interaction_encoder = InteractionEncoder(latent_size)
        self.interaction_decoder = InteractionDecoder(latent_size)
        self.residual_decoder = NodeResidualDecoder(latent_size)
        self.layer_norm = nn.LayerNorm(latent_size)

    def forward(self, edge_index, senders_pos, receivers_pos, edge_dx_, edge_attr, vector_a, vector_b, vector_c, 
                senders_v_t_, senders_v_tm1_, 
                receivers_v_t_, receivers_v_tm1_,
                node_acc, node_latent, history=None, latent_history=False):
            
        interaction_latent = self.interaction_encoder(
            edge_index, edge_dx_, edge_attr,
            vector_a, vector_b, vector_c,
            senders_v_t_, senders_v_tm1_,
            receivers_v_t_, receivers_v_tm1_,
            node_latent
        )

        # Residual connection
        if latent_history and history is not None:
            interaction_latent = self.layer_norm(interaction_latent + history)
            #interaction_latent = self.layer_norm(interaction_latent)
        
        # Decode forces and torques
        edge_force = self.interaction_decoder(
            edge_index, senders_pos, receivers_pos, 
            vector_a, vector_b, vector_c, 
            interaction_latent, node_latent
        )
        
        # Decode node updates
        residual = self.residual_decoder(
            edge_index, node_latent, edge_force, node_acc
        )
    
        return residual, interaction_latent

class Error_Corrector(nn.Module):
    def __init__(self, latent_size):
        super(Error_Corrector, self).__init__()
        # Input: 2 scalars (Norm of Local Residual, Norm of Steered Residual)
        # Output: 2 scalars (Alpha, Beta coefficients)
        # We use a deeper or wider MLP if needed to capture non-linear scaling laws
        self.corrector = build_mlp_d(
            3, 
            latent_size, 
            2,  # [alpha, beta]
            num_layers=MODEL_SETTINGS['n_layers'], 
            lay_norm=False
        )
    
    def forward(self, solver_input):
        """
        Args:
            solver_input: [N, 2] Tensor containing [ ||Ri||, ||RGi|| ]
        Returns:
            coeffs: [N, 2] Tensor containing [alpha, beta]
        """
        return self.corrector(solver_input)

class DynamicsSolver(torch.nn.Module):
    def __init__(self, sample_step, train_stats, num_jumps=1, num_msgs=1, latent_size=128):
        super(DynamicsSolver, self).__init__()
        self.refframecalc = RefFrameCalc()
        self.scaler = Scaler()
        self.node_encoder = NodeEncoder(latent_size)
        self.interaction_proc_layer = Interaction_Block(latent_size)
        self.interaction_init_layer = Interaction_Block(latent_size)
        self.corrector = Error_Corrector(latent_size)
        self.num_messages = num_msgs
        self.dt = sample_step
        self.sub_tstep = sample_step / num_msgs
        self.train_stats = train_stats

    def forward(self, graph):
        # ---------------------------------------------------------
        # 1. Data Preparation
        # ---------------------------------------------------------
        pos = graph.pos.float()
        vel = graph.vel.float()
        prev_vel = graph.prev_vel.float()
        edge_attr = graph.edge_attr.float()
        node_type = graph.node_type.float()
        edge_index = graph.edge_index.long()
        senders, receivers = edge_index

        # Mask for movable nodes (Body). Wall/Boundary nodes (type == 2) are fixed.
        mask_body = (graph.node_type != 2).squeeze()

        # ---------------------------------------------------------
        # 2. Newmark-Beta Predictor (Initial Guess)
        # ---------------------------------------------------------
        # Constants for unconditionally stable Average Acceleration Method
        dt = self.dt # Ideally, pass this as self.dt or graph.dt
        beta = 0.25
        gamma = 0.5

        # Calculate current acceleration (a_n) using backward difference
        acc = (vel - prev_vel) / dt  

        # Predict Displacement x_{n+1}
        # x_{n+1}^0 = x_n + dt*v_n + (0.5 - beta)*dt^2*a_n
        pred_disp = (vel * dt) + ((0.5 - beta) * (dt ** 2) * acc)
        
        # Guessed Position
        node_x_t1 = pos + pred_disp

        # Consistent Implicit Relations (Slave Variables)
        # We enforce v_{n+1} and a_{n+1} to be consistent with the guessed x_{n+1}
        
        # a_{n+1} = (1 / beta*dt^2) * (x_{n+1} - x_n - dt*v_n) - (1/2beta - 1)*a_n
        node_a_t1 = (1.0 / (beta * dt**2)) * (node_x_t1 - pos - dt * vel) - ((1.0 / (2 * beta)) - 1.0) * acc

        # v_{n+1} = v_n + dt * ((1 - gamma)*a_n + gamma*a_{n+1})
        node_v_t1 = vel + dt * ((1 - gamma) * acc + gamma * node_a_t1)
        
        # Pre-compute Node Latent (static per step)
        node_latent = self.node_encoder(torch.hstack((node_type,vel.norm(dim=1,keepdim=True))))
        
        # Residue state for RNN-like message passing (Memory)
        history = None
        total_residual = torch.zeros_like(vel.norm(dim=1,keepdim=True))

        # ---------------------------------------------------------
        # 3. Implicit Solver Loop (Message Passing)
        # ---------------------------------------------------------
        for i in range(self.num_messages):
            # A. Update Edge Geometry based on CURRENT guess
            # Optimization: Doing this inside loop is necessary as positions update
            edge_dx_t1 = node_x_t1[receivers] - node_x_t1[senders]
            
            # Prepare scaled inputs for the Physics/Interaction Block
            # (Assuming s_vt_ etc are scaled versions of velocities)
            s_vt = node_v_t1[senders]
            r_vt = node_v_t1[receivers]
            s_vtm1 = vel[senders] # v_n becomes "prev" inside the step context
            r_vtm1 = vel[receivers]

            # Scaling
            s_vt_, s_vtm1_, r_vt_, r_vtm1_, edge_dx_ = self.scaler(
                s_vt, s_vtm1, r_vt, r_vtm1, edge_dx_t1, self.train_stats
            )

            # B. Reference Frame Calculation
            # Calculates local frames for equivariant message passing
            vec_a, vec_b, vec_c = self.refframecalc(
                edge_index, node_x_t1[senders], node_x_t1[receivers],
                s_vt_, r_vt_
            )

            # C. Interaction Block (Physics Residual Calculation)
            # This predicts the force/momentum imbalance
            layer = self.interaction_init_layer if i == 0 else self.interaction_proc_layer
            history_flag = (i > 0)
            
            # Note: Assuming 'residual' output here is the Force/Momentum Imbalance vector (Ri)
            # Also assuming 'current_pos' in your snippet refers to 'node_x_t1'
            Ri,history = layer(
                edge_index, node_x_t1[senders], node_x_t1[receivers],
                edge_dx_, edge_attr, vec_a, vec_b, vec_c,
                s_vt_, s_vtm1_,  # s_wt (angular) omitted based on context, add if needed
                r_vt_, r_vtm1_,  # r_wt (angular) omitted based on context, add if needed
                node_a_t1,node_latent, history=history, latent_history=history_flag
            )

            # --- ACCUMULATE RESIDUAL ---
            # We calculate the mean squared norm of the residual force vector
            # This represents the "Physics Loss" for this iteration step
            step_residual = Ri.norm(dim=1,keepdim=True)
            
            # Optional: Weight later steps more heavily (Deep Supervision)
            gamma_decay = 0.8 ** (self.num_messages - i - 1)
            step_residual = gamma_decay * step_residual
            total_residual = total_residual + step_residual


            # D. Global Context Calculation (The Elliptic Bridge)
            # Calculate Gram Matrix G = sum(Ri * Ri.T)
            # Ri is [N, 3]. We sum over N to get [3, 3] Global Matrix.
            # Using torch.matmul(Ri.T, Ri)
            num_nodes = Ri.shape[0]
            G = torch.matmul(Ri.T, Ri)/ (num_nodes + 1) # Shape [3, 3]
            
            # Calculate Steered Residual (Tensor contraction)
            # RGi = G * Ri (Matrix-Vector product per node)
            # (N, 3) @ (3, 3).T -> (N, 3)
            RGi = torch.matmul(Ri, G.T)

            # 3. Safe Inputs for the Corrector Network
            # The inputs [||Ri||, ||RGi||] have wildly different scales.
            # Log-space transformation is safer for the network to learn.
            norm_Ri = torch.norm(Ri, dim=1, keepdim=True)
            norm_RGi = torch.norm(RGi, dim=1, keepdim=True)
            alignment = (Ri * RGi).sum(dim=1, keepdim=True)     # [N, 1]


            # E. Solver Block (Correction Prediction)
            # Predict scalar coefficients alpha, beta based on invariant features
            # Input: Norm of Local Residual and Norm of Steered Residual
            solver_input = torch.cat([
                norm_Ri,      # Local Energy
                norm_RGi,     # Steered Energy
                alignment,
            ], dim=1)
            
            # Assuming self.corrector outputs shape [N, 2] -> (alpha, beta)
            coeffs = self.corrector(solver_input) 
            c1 = coeffs[:, 0:1]
            c2 = coeffs[:, 1:2]
            #c3 = coeffs[:, 2:3]*self.dt

            # Calculate Displacement Correction
            delta_u = c1 * Ri + c2 * RGi
            
            # Apply Correction (Only to movable nodes)
            # We zero out updates for boundary nodes to enforce Dirichlet BCs implicitly
            delta_u[~mask_body] = 0.0*delta_u[~mask_body]
            
            pred_disp = pred_disp + delta_u
            node_x_t1 = pos + pred_disp

            # F. Update Kinematics (Slave Variables)
            # Re-enforce Newmark consistency for the next iteration's residual calculation
            
            # Update Acceleration
            # a_{n+1} = (1 / beta*dt^2) * (x_{n+1} - x_n - dt*v_n) - ...
            node_a_t1 = (1.0 / (beta * dt**2)) * (node_x_t1 - pos - dt * vel) - ((1.0 / (2 * beta)) - 1.0) * acc

            # Update Velocity
            # v_{n+1} = v_n + dt * ((1 - gamma)*a_n + gamma*a_{n+1})
            node_v_t1 = vel + dt * ((1 - gamma) * acc + gamma * node_a_t1)

        # Return final state
        # vel (current input v_n) becomes the "prev_vel" for the next time step
        return node_v_t1-vel, node_x_t1-pos, total_residual