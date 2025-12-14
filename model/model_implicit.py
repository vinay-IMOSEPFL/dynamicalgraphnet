import torch
import torch.nn as nn
from utils.utils_implicit import build_mlp_d

class RefFrameCalc(nn.Module):
    def __init__(self):
        super(RefFrameCalc, self).__init__()
        self.epsilon = 1e-6

    def forward(
        self, 
        edge_index, 
        senders_pos, receivers_pos, 
        senders_vel, receivers_vel,
        senders_prev_vel, receivers_prev_vel, 
        senders_omega, receivers_omega
        ):
        # Calculate relative position (Edge vector)
        rel_pos = receivers_pos - senders_pos
        dist = rel_pos.norm(dim=1, keepdim=True)
        vector_a = rel_pos / dist

        # Preliminary vectors
        # 1. Relative velocity cross edge vector
        # 2. Sum of velocities
        diff_vel = receivers_vel - senders_vel
        sum_vel = senders_vel + receivers_vel

        diff_prev_vel = receivers_prev_vel - senders_prev_vel
        sum_prev_vel = senders_prev_vel + receivers_prev_vel        
        
        diff_omega = receivers_omega - senders_omega
        sum_omega = senders_omega + receivers_omega

        def normalize(tensor):
            return tensor / tensor.norm(dim=1, keepdim=True).clamp(min=self.epsilon)


        b_i = normalize(torch.cross(diff_vel, vector_a, dim=1))
        b_ii = normalize(sum_vel)
        b_iii = normalize(torch.cross(diff_omega, vector_a, dim=1))
        b_iv = normalize(sum_omega)
        b_v = normalize(torch.cross(diff_prev_vel, vector_a, dim=1))
        b_vi = normalize(sum_prev_vel)

        # Combine to form b
        b = b_i + b_ii + b_iii + b_iv + b_v + b_vi+torch.ones_like(b_i)

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
        self.edge_feat_encoder = build_mlp_d(1+edge_in_f, latent_size, latent_size, num_layers=mlp_layers, lay_norm=True)
        self.node_feat_encoder = build_mlp_d(6, latent_size, latent_size, num_layers=mlp_layers, lay_norm=True)
        self.interaction_encoder = build_mlp_d(3*latent_size, latent_size, latent_size, num_layers=mlp_layers, lay_norm=True)

    def forward(self, edge_index, edge_dx_, edge_attr, vector_a, vector_b, vector_c,
                senders_v_t_, senders_v_tm1_, senders_w_t_,
                receivers_v_t_, receivers_v_tm1_, receivers_w_t_,
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
        #s_vtm1_proj = project(senders_v_tm1_)
        s_wt_proj   = project(senders_w_t_)

        # Project receivers (negated as per original code logic: -vector_a, etc.)
        # Note: Original code did v . (-a), v . (-b). This is equivalent to -(v . a).
        r_vt_proj   = -project(receivers_v_t_)
        #r_vtm1_proj = -project(receivers_v_tm1_)
        r_wt_proj   = -project(receivers_w_t_)

        # Concatenate features (9 features per node per edge)
        senders_features = torch.cat([s_vt_proj, s_wt_proj], dim=1)
        receivers_features = torch.cat([r_vt_proj, r_wt_proj], dim=1)
        edge_features = torch.cat((edge_dx_.norm(dim=1, keepdim=True), edge_attr), dim=1)


        senders_proj_latent = self.node_feat_encoder(senders_features)
        receivers_proj_latent = self.node_feat_encoder(receivers_features)
        edge_latent = self.edge_feat_encoder(edge_features)

        
        # Edge encodings
        interaction_latent = self.interaction_encoder(
            torch.cat(
                (senders_proj_latent  + receivers_proj_latent, 
                 node_latent[senders] + node_latent[receivers],
                 edge_latent
                ), dim=1))
        
        return interaction_latent

class InteractionDecoder(torch.nn.Module):
    def __init__(self, latent_size=128,mlp_layers=2):
        super(InteractionDecoder, self).__init__()
        self.i1_decoder = build_mlp_d(latent_size, latent_size, 3, num_layers=mlp_layers, lay_norm=False)
        self.i2_decoder = build_mlp_d(latent_size, latent_size, 3, num_layers=mlp_layers, lay_norm=False)
        self.f_scaler = build_mlp_d(latent_size, latent_size, 1, num_layers=mlp_layers, lay_norm=False)
        self.node_weight_decoder = build_mlp_d(latent_size, latent_size, 1, num_layers=mlp_layers, lay_norm=False)

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
        denom = w_s + w_r# Add epsilon for safety
        r0ij = (w_s * senders_pos + w_r * receivers_pos) / denom
        
        # Compute torque
        # tau = aij - (r_j - r0_ij) x (lambda * fij)
        lever_arm = receivers_pos - r0ij
        torque_contribution = torch.cross(lever_arm, fij * lambda_ij, dim=1)
        tauij = aij - torque_contribution
        
        return fij, tauij

class Node_Internal_Dv_Decoder(torch.nn.Module):
    def __init__(self, latent_size=128, mlp_layers=2):
        super(Node_Internal_Dv_Decoder, self).__init__()
        self.m_decoder = build_mlp_d(latent_size, latent_size, 1, num_layers=mlp_layers, lay_norm=False)
        self.i_decoder = build_mlp_d(latent_size, latent_size, 1, num_layers=mlp_layers, lay_norm=False)
        self.f_ext_decoder = build_mlp_d(latent_size, latent_size, 3, num_layers=mlp_layers, lay_norm=False)
        self.t_ext_decoder = build_mlp_d(latent_size, latent_size, 3, num_layers=mlp_layers, lay_norm=False) 

    def forward(self, edge_index, node_latent, fij, tij, acc, alpha):
        senders, receivers = edge_index   
        num_nodes = node_latent.shape[0]
        
        m = self.m_decoder(node_latent)
        i = self.i_decoder(node_latent)
        
        out_fij = node_latent.new_zeros((num_nodes, 3))
        out_tij = node_latent.new_zeros((num_nodes, 3))
        
        out_fij.index_add_(0, receivers, fij)
        out_tij.index_add_(0, receivers, tij)

        # SEPARATE RESIDUALS (Physics Correction)
        # Linear Residual: ma - F_int - F_ext
        Ri_linear = m * acc - out_fij - self.f_ext_decoder(node_latent)
        
        # Angular Residual: I*alpha - Tau_int - Tau_ext
        # (Assuming I is scalar approximation or diagonal, otherwise matmul needed)
        # We add a decoder for External Torque (t_ext) for completeness
        Ri_angular = i * alpha - out_tij - self.t_ext_decoder(node_latent)
        
        return Ri_linear, Ri_angular

class Scaler(torch.nn.Module):
    def __init__(self):
        super(Scaler, self).__init__()

    def forward(self, senders_v_t, senders_v_tm1, receivers_v_t, receivers_v_tm1, edge_dx, train_stats):
        stat_edge_dx, stat_node_v_t, _, _ = train_stats
        
        # Use detach on stats to ensure no gradients flow back to stats (redundant but safe)
        v_scale = stat_node_v_t[1].detach()
        
        senders_v_t_ = senders_v_t / v_scale
        senders_v_tm1_ = senders_v_tm1 / v_scale
        receivers_v_t_ = receivers_v_t / v_scale
        receivers_v_tm1_ = receivers_v_tm1 / v_scale
        
        norm_edge_dx = edge_dx.norm(dim=1, keepdim=True)
        # Avoid division by zero
        safe_norm = norm_edge_dx
        
        min_stat, max_stat = stat_edge_dx
        scale_denom = (max_stat - min_stat).detach() 
        
        # Scale magnitude, preserve direction
        scaled_mag = (norm_edge_dx - min_stat.detach()) / scale_denom
        edge_dx_ = scaled_mag * (edge_dx / safe_norm)
        
        return senders_v_t_, senders_v_tm1_, receivers_v_t_, receivers_v_tm1_, edge_dx_.detach()

class Interaction_Block(torch.nn.Module):
    def __init__(self, edge_in_f, latent_size, mlp_layers):
        super(Interaction_Block, self).__init__()
        self.interaction_encoder = InteractionEncoder(edge_in_f, latent_size, mlp_layers)
        self.interaction_decoder = InteractionDecoder(latent_size, mlp_layers)
        self.internal_dv_decoder = Node_Internal_Dv_Decoder(latent_size, mlp_layers)
        self.layer_norm = nn.LayerNorm(latent_size)

    def forward(self, edge_index, senders_pos, receivers_pos, edge_dx_, edge_attr, 
                vector_a, vector_b, vector_c, 
                senders_v_t_, senders_v_tm1_, senders_w_t_,
                receivers_v_t_, receivers_v_tm1_, receivers_w_t_,
                node_a_t1, node_alpha_t1,
                node_latent, residue=None, latent_history=False):
            
        interaction_latent = self.interaction_encoder(
            edge_index, edge_dx_, edge_attr, vector_a, vector_b, vector_c,
            senders_v_t_, senders_v_tm1_, senders_w_t_, 
            receivers_v_t_, receivers_v_tm1_, receivers_w_t_, 
            node_latent
        )

        if latent_history and residue is not None:
            interaction_latent = self.layer_norm(interaction_latent + residue)
        
        edge_force, edge_tau = self.interaction_decoder(
            edge_index, senders_pos, receivers_pos, 
            vector_a, vector_b, vector_c, 
            interaction_latent, node_latent
        )
        
        # Returns TUPLE of residuals
        Ri_linear, Ri_angular = self.internal_dv_decoder(
            edge_index, node_latent, edge_force, edge_tau, node_a_t1, node_alpha_t1
        )
    
        return Ri_linear, Ri_angular, interaction_latent

class Error_Corrector(nn.Module):
    def __init__(self, latent_size,mlp_layers):
        super(Error_Corrector, self).__init__()
        # Input: 2 scalars (Norm of Local Residual, Norm of Steered Residual)
        # Output: 2 scalars (Alpha, Beta coefficients)
        # We use a deeper or wider MLP if needed to capture non-linear scaling laws
        self.corrector = build_mlp_d(
            3, 
            latent_size, 
            2,  # [alpha, beta]
            num_layers=mlp_layers, 
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
    def __init__(self, node_in_f, edge_in_f, time_step, train_stats, num_msgs=1, latent_size=128, mlp_layers=2):
        super(DynamicsSolver, self).__init__()
        self.refframecalc = RefFrameCalc()
        self.scaler = Scaler()
        self.node_encoder = NodeEncoder(node_in_f, latent_size, mlp_layers)
        self.interaction_proc_layer = Interaction_Block(edge_in_f, latent_size, mlp_layers)
        self.interaction_init_layer = Interaction_Block(edge_in_f, latent_size, mlp_layers)
        
        # Separate correctors for Linear and Angular domains
        self.corrector_v = Error_Corrector(latent_size,mlp_layers)
        self.corrector_w = Error_Corrector(latent_size,mlp_layers)
        
        self.num_messages = num_msgs
        self.dt = time_step
        self.train_stats = train_stats

    def forward(self, graph):
        # 1. Data Preparation
        pos = graph.pos.float()
        vel = graph.vel.float()
        # prev_vel = graph.prev_vel.float() # Not used in Newmark currently
        theta = getattr(graph, 'theta', torch.zeros_like(vel))
        omega = getattr(graph, 'node_w_t', torch.zeros_like(vel))
        
        edge_attr = graph.edge_attr.float()
        node_type = graph.node_type.float()
        edge_index = graph.edge_index.long()
        senders, receivers = edge_index

        mask_body = (graph.node_type != 2).squeeze()

        # 2. Newmark-Beta Predictor (Initial Guess)
        dt = self.dt
        beta = 0.25
        gamma = 0.5

        # Predictors
        pred_disp = (vel * dt)
        pred_disp_ang = (omega * dt)
        
        node_x_t1 = pos + pred_disp
        node_th_t1 = theta + pred_disp_ang

        # Consistent Accelerations (assuming a_n = 0)
        node_a_t1 = (1.0 / (beta * dt**2)) * (node_x_t1 - pos - dt * vel)
        node_alpha_t1 = (1.0 / (beta * dt**2)) * (node_th_t1 - theta - dt * omega)

        node_v_t1 = vel + dt * (gamma * node_a_t1)
        node_w_t1 = omega + dt * (gamma * node_alpha_t1)
        
        node_latent = self.node_encoder(node_type)
        
        residue = None
        total_residual = torch.zeros_like(vel.norm(dim=1, keepdim=True))

        # 3. Implicit Solver Loop
        for i in range(self.num_messages):
            # A. Update Edge Geometry
            edge_dx_t1 = node_x_t1[receivers] - node_x_t1[senders]
            
            s_vt, r_vt = node_v_t1[senders], node_v_t1[receivers]
            s_vtm1, r_vtm1 = vel[senders], vel[receivers]
            s_wt, r_wt = node_w_t1[senders], node_w_t1[receivers]        

            # Scaling
            s_vt_, s_vtm1_, r_vt_, r_vtm1_, edge_dx_ = self.scaler(
                s_vt, s_vtm1, r_vt, r_vtm1, edge_dx_t1, self.train_stats
            )

            vec_a, vec_b, vec_c = self.refframecalc(
                edge_index, node_x_t1[senders], node_x_t1[receivers],
                s_vt, r_vt, s_vtm1, r_vtm1, s_wt, r_wt
            )

            # B. Interaction Block (Returns Split Residuals)
            layer = self.interaction_init_layer if i == 0 else self.interaction_proc_layer
            history_flag = (i > 0)
            
            Ri_lin, Ri_ang, history = layer(
                edge_index, node_x_t1[senders], node_x_t1[receivers],
                edge_dx_, edge_attr, vec_a, vec_b, vec_c,
                s_vt_, s_vtm1_, s_wt,  
                r_vt_, r_vtm1_, r_wt,  
                node_a_t1, node_alpha_t1, 
                node_latent, residue=residue, latent_history=history_flag
            )
            
            # BUG FIX: Update residue for next iteration
            residue = history 

            # Calculate Step Residual (Physics Loss)
            # Combine linear and angular residuals safely (magnitude sum)
            step_res_mag = Ri_lin.norm(dim=1, keepdim=True) + Ri_ang.norm(dim=1, keepdim=True)
            gamma_decay = 0.8 ** (self.num_messages - i - 1)
            total_residual = total_residual + (gamma_decay * step_res_mag)

            # C. Global Context & Corrections (Done separately for Linear and Angular)
            
            # --- Linear Correction ---
            G_lin = torch.matmul(Ri_lin.T, Ri_lin) / (Ri_lin.shape[0] + 1e-6)
            RGi_lin = torch.matmul(Ri_lin, G_lin.T) / torch.trace(G_lin)
            
            solver_in_lin = torch.cat([
                Ri_lin.norm(dim=1, keepdim=True), 
                RGi_lin.norm(dim=1, keepdim=True),
                (Ri_lin * RGi_lin).sum(dim=1, keepdim=True)
            ], dim=1)
            
            coeffs_v = self.corrector_v(solver_in_lin)
            delta_u = coeffs_v[:, 0:1] * Ri_lin + coeffs_v[:, 1:2] * RGi_lin
            delta_u[~mask_body] = 0.0

            # --- Angular Correction ---
            G_ang = torch.matmul(Ri_ang.T, Ri_ang) / (Ri_ang.shape[0] + 1e-6)
            RGi_ang = torch.matmul(Ri_ang, G_ang.T)/ torch.trace(G_ang)

            solver_in_ang = torch.cat([
                Ri_ang.norm(dim=1, keepdim=True), 
                RGi_ang.norm(dim=1, keepdim=True),
                (Ri_ang * RGi_ang).sum(dim=1, keepdim=True)
            ], dim=1)

            coeffs_w = self.corrector_w(solver_in_ang)
            delta_th = coeffs_w[:, 0:1] * Ri_ang + coeffs_w[:, 1:2] * RGi_ang
            delta_th[~mask_body] = 0.0

            # D. Update State
            pred_disp = pred_disp + delta_u
            pred_disp_ang = pred_disp_ang + delta_th
            
            node_x_t1 = pos + pred_disp
            node_th_t1 = theta + pred_disp_ang

            # Update derivatives consistent with Newmark
            node_a_t1 = (1.0 / (beta * dt**2)) * (node_x_t1 - pos - dt * vel)
            node_v_t1 = vel + dt * (gamma * node_a_t1)
            
            node_alpha_t1 = (1.0 / (beta * dt**2)) * (node_th_t1 - theta - dt * omega)
            node_w_t1 = omega + dt * (gamma * node_alpha_t1)

        return node_v_t1 - vel, node_x_t1 - pos, total_residual