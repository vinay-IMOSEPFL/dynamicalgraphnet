import torch
import torch.nn as nn
from utils.utils import build_mlp_d

class RefFrameCalc(nn.Module):
    def __init__(self):
        super(RefFrameCalc, self).__init__()
        self.epsilon = 1e-7

    def forward(self, edge_index, senders_pos, receivers_pos, senders_vel, receivers_vel, senders_omega, receivers_omega):
        # Calculate relative position (Edge vector)
        rel_pos = receivers_pos - senders_pos
        dist = rel_pos.norm(dim=1, keepdim=True).clamp(min=self.epsilon)
        vector_a = rel_pos / dist

        # Preliminary vectors
        # 1. Relative velocity cross edge vector
        # 2. Sum of velocities
        diff_vel = receivers_vel - senders_vel
        sum_vel = senders_vel + receivers_vel
        
        diff_omega = receivers_omega - senders_omega
        sum_omega = senders_omega + receivers_omega

        # Helper for normalizing
        def normalize(tensor):
            return tensor / tensor.norm(dim=1, keepdim=True).clamp(min=self.epsilon)

        b_a = normalize(torch.cross(diff_vel, vector_a, dim=1))
        b_c = normalize(sum_vel)
        b_a_ = normalize(torch.cross(diff_omega, vector_a, dim=1))
        b_c_ = normalize(sum_omega)

        # Combine to form b
        b = b_a + b_c + b_a_ + b_c_

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
    def __init__(self, node_in_f, latent_size, mlp_layers):
        super(NodeEncoder, self).__init__()
        self.node_encoder = build_mlp_d(node_in_f, latent_size, latent_size, num_layers=mlp_layers, lay_norm=True)
    
    def forward(self, node_scalar_feat):
        return self.node_encoder(node_scalar_feat)

class InteractionEncoder(nn.Module):
    """Message passing with optimized projections."""

    def __init__(self, edge_in_f, latent_size,mlp_layers):
        super(InteractionEncoder, self).__init__()
        self.edge_feat_encoder = build_mlp_d(18, latent_size, latent_size, num_layers=mlp_layers, lay_norm=True)
        self.edge_encoder = build_mlp_d(1+edge_in_f, latent_size, latent_size, num_layers=mlp_layers, lay_norm=True)
        self.interaction_encoder = build_mlp_d(3*latent_size, latent_size, latent_size, num_layers=mlp_layers, lay_norm=True)

    def forward(self, edge_index, edge_dx_, edge_attr, vector_a, vector_b, vector_c,
                senders_v_t_, senders_v_tm1_, senders_w_t_, senders_w_tm1_, senders_a_t_, senders_alpha_t_,
                receivers_v_t_, receivers_v_tm1_, receivers_w_t_, receivers_w_tm1_, receivers_a_t_, receivers_alpha_t_,
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
        s_wt_proj   = project(senders_w_t_)
        s_wtm1_proj   = project(senders_w_tm1_)
        s_at_proj   = project(senders_a_t_)
        s_alphat_proj   = project(senders_alpha_t_)

        # Project receivers (negated as per original code logic: -vector_a, etc.)
        # Note: Original code did v . (-a), v . (-b). This is equivalent to -(v . a).
        r_vt_proj   = -project(receivers_v_t_)
        r_vtm1_proj = -project(receivers_v_tm1_)
        r_wt_proj   = -project(receivers_w_t_)
        r_wtm1_proj   = project(receivers_w_tm1_)
        r_at_proj   = -project(receivers_a_t_)
        r_alphat_proj   = project(receivers_alpha_t_)

        # Concatenate features (9 features per node per edge)
        senders_features = torch.cat([s_vt_proj, s_vtm1_proj, s_wt_proj, s_wtm1_proj, s_at_proj, s_alphat_proj], dim=1)
        receivers_features = torch.cat([r_vt_proj, r_vtm1_proj, r_wt_proj, r_wtm1_proj, r_at_proj, r_alphat_proj], dim=1)
        
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
    def __init__(self, latent_size=128,mlp_layers=2):
        super(InteractionDecoder, self).__init__()
        self.i1_decoder = build_mlp_d(latent_size, latent_size, 3, num_layers=mlp_layers, lay_norm=False)
        self.i2_decoder = build_mlp_d(latent_size, latent_size, 3, num_layers=mlp_layers, lay_norm=False)
        self.f_scaler = build_mlp_d(latent_size, latent_size, 1, num_layers=mlp_layers, lay_norm=False)
        self.node_weight_decoder = build_mlp_d(latent_size, latent_size, 1, num_layers=mlp_layers, lay_norm=False, use_sigmoid=True)

    def forward(self, edge_index, senders_pos, receivers_pos, vector_a, vector_b, vector_c, interaction_latent, node_latent):
        senders, receivers = edge_index
        
        # Decode coefficients
        coeff_f = self.i1_decoder(interaction_latent)
        coeff_a = self.i2_decoder(interaction_latent)
        lambda_ij = self.f_scaler(interaction_latent)
        
        # Reconstruct forces in global frame
        # Linear combination: c0*a + c1*b + c2*c
        fij = (coeff_f[:, 0:1] * vector_a + 
               coeff_f[:, 1:2] * vector_b + 
               coeff_f[:, 2:3] * vector_c) # Fixed slicing for clarity

        aij = (coeff_a[:, 0:1] * vector_a + 
               coeff_a[:, 1:2] * vector_b + 
               coeff_a[:, 2:3] * vector_c)

        # Node weights for torque center
        w_s = self.node_weight_decoder(node_latent[senders])
        w_r = self.node_weight_decoder(node_latent[receivers])
        
        # Weighted center r0ij
        denom = w_s + w_r + 1e-8 # Add epsilon for safety
        r0ij = (w_s * senders_pos + w_r * receivers_pos) / denom
        
        # Compute torque
        # tau = aij - (r_j - r0_ij) x (lambda * fij)
        lever_arm = receivers_pos - r0ij
        torque_contribution = torch.cross(lever_arm, fij * lambda_ij, dim=1)
        tauij = aij - torque_contribution
        
        return fij, tauij

class Node_Internal_Dv_Decoder(torch.nn.Module):
    def __init__(self, latent_size=128,mlp_layers=2):
        super(Node_Internal_Dv_Decoder, self).__init__()
        self.m_inv_decoder = build_mlp_d(latent_size, latent_size, 1, num_layers=mlp_layers, lay_norm=False)
        self.i_inv_decoder = build_mlp_d(latent_size, latent_size, 1, num_layers=mlp_layers, lay_norm=False)
        self.dv_ext_decoder = build_mlp_d(latent_size, latent_size, 3, num_layers=mlp_layers, lay_norm=False)

    def forward(self, edge_index, node_latent, fij, tij):
        senders, receivers = edge_index   
        num_nodes = node_latent.shape[0]
        
        # Decode physical properties
        m_inv = self.m_inv_decoder(node_latent)
        i_inv = self.i_inv_decoder(node_latent)
        
        # Aggregate Forces and Torques
        # Use new_zeros to ensure device compatibility automatically
        out_fij = node_latent.new_zeros((num_nodes, 3))
        out_tij = node_latent.new_zeros((num_nodes, 3))
        
        out_fij.index_add_(0, receivers, fij)
        out_tij.index_add_(0, receivers, tij)

        # Compute accelerations
        # F = ma => a = F * (1/m)
        node_dv_int = m_inv * out_fij + self.dv_ext_decoder(node_latent)
        node_dw_int = i_inv * out_tij

        return node_dv_int, node_dw_int

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
    def __init__(self, edge_in_f, latent_size,mlp_layers):
        super(Interaction_Block, self).__init__()
        self.interaction_encoder = InteractionEncoder(edge_in_f, latent_size,mlp_layers)
        self.interaction_decoder = InteractionDecoder(latent_size,mlp_layers)
        self.internal_dv_decoder = Node_Internal_Dv_Decoder(latent_size,mlp_layers)
        self.layer_norm = nn.LayerNorm(latent_size)

    def forward(self, edge_index, senders_pos, receivers_pos, edge_dx_, edge_attr, vector_a, vector_b, vector_c, 
                senders_v_t_, senders_v_tm1_, senders_w_t_, senders_w_tm1_, senders_a_t_,senders_alpha_t_,
                receivers_v_t_, receivers_v_tm1_, receivers_w_t_, receivers_w_tm1_, receivers_a_t_, receivers_alpha_t_,
                node_latent, residue=None, latent_history=False):
            
        interaction_latent = self.interaction_encoder(
            edge_index, edge_dx_, edge_attr,
            vector_a, vector_b, vector_c,
            senders_v_t_, senders_v_tm1_, senders_w_t_, senders_w_tm1_, senders_a_t_,senders_alpha_t_,
            receivers_v_t_, receivers_v_tm1_, receivers_w_t_, receivers_w_tm1_, receivers_a_t_, receivers_alpha_t_,
            node_latent
        )

        # Residual connection
        if latent_history and residue is not None:
            interaction_latent = self.layer_norm(interaction_latent + residue)
            #interaction_latent = self.layer_norm(interaction_latent)
        
        # Decode forces and torques
        edge_force, edge_tau = self.interaction_decoder(
            edge_index, senders_pos, receivers_pos, 
            vector_a, vector_b, vector_c, 
            interaction_latent, node_latent
        )
        
        # Decode node updates
        node_dv, node_dw = self.internal_dv_decoder(
            edge_index, node_latent, edge_force, edge_tau
        )
    
        return node_dv, node_dw, interaction_latent

class DynamicsSolver(torch.nn.Module):
    def __init__(self, node_in_f, edge_in_f, time_step, train_stats, num_msgs=1, latent_size=128, mlp_layers=2):
        super(DynamicsSolver, self).__init__()
        self.refframecalc = RefFrameCalc()
        self.scaler = Scaler()
        self.node_encoder = NodeEncoder(node_in_f,latent_size,mlp_layers)
        self.interaction_proc_layer = Interaction_Block(edge_in_f,latent_size,mlp_layers)
        self.interaction_init_layer = Interaction_Block(edge_in_f,latent_size,mlp_layers)
        self.num_messages = num_msgs
        self.sub_tstep = time_step / num_msgs
        self.train_stats = train_stats

    def forward(self, graph):
        # Data preparation
        pos = graph.pos.float()
        vel = graph.vel.float()
        prev_vel = graph.prev_vel.float()
        edge_attr = graph.edge_attr.float()
        node_type = graph.node_type.float()
        edge_index = graph.edge_index.long()
        senders, receivers = edge_index

        # Mask for nodes that are NOT reflected (type != 2)
        # Using a boolean mask is faster for indexing
        mask_body = (graph.node_type[:,-1:] != 2).squeeze()

        # Initial State
        node_v_t = vel
        node_v_tm1 = prev_vel
        
        # Get or initialize angular quantities (optimization: use safe defaults directly)
        node_w_t = getattr(graph, 'node_w_t', torch.zeros_like(vel))
        node_w_tm1 = getattr(graph, 'node_w_tm1', torch.zeros_like(vel))
        
        # Initialize Accumulators
        sum_node_dv = torch.zeros_like(vel)
        sum_node_dx = torch.zeros_like(vel)
        
        # Pre-compute Node Latent (static per step)
        node_latent = self.node_encoder(node_type)

        # Initialize iteration vars
        current_pos = pos
        current_edge_dx = current_pos[receivers] - current_pos[senders]
        
        # Residue state for RNN-like message passing
        residue = None

        # initial predictions are set to zero.
        node_dv = torch.zeros_like(vel)
        node_dw = torch.zeros_like(vel)

        for i in range(self.num_messages):
            # Gather node attributes for edges
            # Optimization: Doing this inside loop is necessary as values update


            s_vt = node_v_t[senders]
            r_vt = node_v_t[receivers]
            s_vtm1 = node_v_tm1[senders]
            r_vtm1 = node_v_tm1[receivers]
            s_wt = node_w_t[senders]
            r_wt = node_w_t[receivers]
            s_wtm1 = node_w_tm1[senders]
            r_wtm1 = node_w_tm1[receivers]

            s_at = node_dv[senders]
            r_at = node_dv[receivers]
            s_alphat = node_dw[senders]
            r_alphat = node_dw[receivers]

            # Scaling
            s_vt_, s_vtm1_, r_vt_, r_vtm1_, edge_dx_ = self.scaler(
                s_vt, s_vtm1, r_vt, r_vtm1, current_edge_dx, self.train_stats
            )

            # Reference Frame
            vec_a, vec_b, vec_c = self.refframecalc(
                edge_index, current_pos[senders], current_pos[receivers],
                s_vt_, r_vt_, s_wt, r_wt
            )

            # Interaction Block
            layer = self.interaction_init_layer if i == 0 else self.interaction_proc_layer
            history_flag = (i > 0)
            
            node_dv, node_dw, residue = layer(
                edge_index, current_pos[senders], current_pos[receivers],
                edge_dx_, edge_attr, vec_a, vec_b, vec_c,
                s_vt_, s_vtm1_, s_wt, s_wtm1, s_at, s_alphat,
                r_vt_, r_vtm1_, r_wt, r_wtm1, r_at, r_alphat,
                node_latent, residue=residue, latent_history=history_flag
            )

            # Integration (Symplectic Euler)
            # Update accumulators and states for body nodes only
            
            # Apply updates
            
            # Update Accumulators (Total Change)
            sum_node_dv[mask_body] += node_dv[mask_body]
            
            # Update Velocity
            node_vf = node_v_t.clone()
            node_vf[mask_body] += node_dv[mask_body]
            
            # Update Angular Velocity
            node_wf = node_w_t.clone()
            node_wf[mask_body] += node_dw[mask_body]

            # Calculate Displacement for this sub-step
            # dx = (v_new + v_old) * 0.5 * dt
            step_disp = (node_v_t + node_vf) * (0.5 * self.sub_tstep)
            sum_node_dx[mask_body] += step_disp[mask_body]

            # Update Position
            current_pos = current_pos + step_disp 

            # Prepare for next iteration
            node_v_tm1 = node_v_t 
            node_v_t = node_vf
            node_w_tm1 = node_w_t
            node_w_t = node_wf
            
            # Update edge vector for next step
            current_edge_dx = current_pos[receivers] - current_pos[senders]

        return sum_node_dv, sum_node_dx