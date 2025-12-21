import torch
import torch.nn as nn
from utils.utils import build_mlp_d
from torch_scatter import scatter_mean,scatter_sum

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
        b = b_i + b_ii + b_iii + b_iv + b_v + b_vi

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


class GlobalKinematicsUpdater(nn.Module):
    def __init__(self, latent_size=128, mlp_layers=2):
        super().__init__()
        # Decodes mass weights for Center of Mass calculation
        # We only need 1 layer for a simple scalar weight
        self.weight_m_decoder = build_mlp_d(latent_size, latent_size, 1, num_layers=1, lay_norm=False)

    def forward(self, pos, prev_vel, vel, node_latent, edge_index, edge_attr, node_type):
        """
        Updates Global Node state (pos, vel) to match the Center of Mass.
        Returns updated states AND the mass weights (w_m) for consistency.
        """
        senders, receivers = edge_index
        num_nodes = pos.size(0)

        # 1. Identify Edges (Body -> Global)
        is_global = (node_type[:, -1] == -1)
        is_virtual_edge = (edge_attr[:, 0] == -1).squeeze()
        mask_rg = is_virtual_edge & is_global[receivers] & (~is_global[senders])

        # 2. Decode Mass Weights
        # Softplus ensures positive mass, +epsilon prevents division by zero
        w_m = torch.nn.functional.softplus(self.weight_m_decoder(node_latent)) + 1e-6

        # 3. Update Global State to Center of Mass
        if mask_rg.any():
            src, dst = senders[mask_rg], receivers[mask_rg]
            w_m_src = w_m[src]
            w_m_sum = scatter_sum(w_m_src, dst, dim=0, dim_size=num_nodes) + 1e-6
            
            # Weighted sums for Pos, Vel, Prev_Vel
            w_pos_sum = scatter_sum(pos[src] * w_m_src, dst, dim=0, dim_size=num_nodes)
            w_prev_vel_sum = scatter_sum(prev_vel[src] * w_m_src, dst, dim=0, dim_size=num_nodes)
            w_vel_sum = scatter_sum(vel[src] * w_m_src, dst, dim=0, dim_size=num_nodes)
            
            # Sum of weights (Total Mass)
            

            # Assign CoM values to Global Nodes
            # We use clone() to avoid in-place modification errors during backprop
            pos = pos.clone();      pos[is_global] = (w_pos_sum / w_m_sum)[is_global]
            prev_vel = prev_vel.clone(); prev_vel[is_global] = (w_prev_vel_sum / w_m_sum)[is_global]
            vel = vel.clone();      vel[is_global] = (w_vel_sum / w_m_sum)[is_global]

        return pos, prev_vel, vel, w_m

class Node_External_Dv_Decoder(nn.Module):
    def __init__(self, latent_size=128, mlp_layers=2):
        super().__init__()
        # Decodes external acceleration (dv) for ALL nodes
        self.external_dv_decoder = build_mlp_d(latent_size, latent_size, 3, num_layers=mlp_layers, lay_norm=False)

    def forward(self, node_latent, node_type, node_weights, edge_index, edge_attr):
        """
        Calculates consistent external forces.
        Requires 'w_m' from KinematicsUpdater to ensure Mass consistency.
        """
        senders, receivers = edge_index
        num_nodes = node_latent.size(0)

        # 1. Identify Edges (Body -> Global)
        is_global = (node_type[:, -1] == -1)
        is_virtual_edge = (edge_attr[:, 0] == -1).squeeze()
        mask_rg = is_virtual_edge & is_global[receivers] & (~is_global[senders])

        # 2. Decode Raw External Forces
        dv_ext_raw = self.external_dv_decoder(node_latent) # [N, 3]
        dv_ext = dv_ext_raw.clone()
        w_m = node_weights

        # 3. Enforce "Global Guides, Local Refines"
        if mask_rg.any():
            src, dst = senders[mask_rg], receivers[mask_rg]
            w_m_src = w_m[src]
            
            # Calculate Total Mass for normalization
            w_m_sum = scatter_sum(w_m_src, dst, dim=0, dim_size=num_nodes) + 1e-6

            # A. Calculate Weighted Mean of Body Predictions
            # "What is the average acceleration the body nodes want?"
            weighted_sum_body = scatter_sum(dv_ext_raw[src] * w_m_src, dst, dim=0, dim_size=num_nodes)
            mean_body = weighted_sum_body / w_m_sum
            
            # B. Calculate Correction Vector
            # "How much do we need to shift the body to match the Global Node's plan?"
            delta_g = dv_ext_raw[dst] - mean_body[dst]
            
            # C. Apply Correction to Body Nodes
            # This ensures momentum is conserved and matches the global trajectory
            dv_ext[src] = dv_ext_raw[src] + delta_g

        return dv_ext

class InteractionEncoder(nn.Module):
    """Message passing with optimized projections."""

    def __init__(self, edge_in_f, latent_size,mlp_layers):
        super(InteractionEncoder, self).__init__()
        self.edge_feat_encoder = build_mlp_d(1+edge_in_f, latent_size, latent_size, num_layers=mlp_layers, lay_norm=True)
        self.node_feat_encoder = build_mlp_d(6, latent_size, latent_size, num_layers=mlp_layers, lay_norm=True)
        self.interaction_encoder = build_mlp_d(3*latent_size, latent_size, latent_size, num_layers=mlp_layers, lay_norm=True)

    def forward(self, edge_index, edge_dx_, edge_attr, vector_a, vector_b, vector_c,
                senders_v_t_, senders_w_t_,
                receivers_v_t_, receivers_w_t_,
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
    def __init__(self, latent_size=128, mlp_layers=2):
        super(InteractionDecoder, self).__init__()
        # Standard MLP Decoders
        self.i1_decoder = build_mlp_d(latent_size, latent_size, 3, num_layers=mlp_layers, lay_norm=False)
        self.i2_decoder = build_mlp_d(latent_size, latent_size, 3, num_layers=mlp_layers, lay_norm=False)
        self.f_scaler = build_mlp_d(latent_size, latent_size, 1, num_layers=mlp_layers, lay_norm=False)
        self.dx_decoder = build_mlp_d(latent_size, latent_size, 3, num_layers=mlp_layers, lay_norm=False)
        self.dv_decoder = build_mlp_d(latent_size, latent_size, 3, num_layers=mlp_layers, lay_norm=False)


    def forward(self, edge_index, edge_attr, senders_pos, receivers_pos, 
                vector_a, vector_b, vector_c, 
                interaction_latent, w_nodes, node_type):
        
        senders, receivers = edge_index
        num_nodes = w_nodes.size(0)

        # 1. Decode Raw Vectors
        coeff_f = self.i1_decoder(interaction_latent)
        coeff_a = self.i2_decoder(interaction_latent)
        lambda_ij = self.f_scaler(interaction_latent)
        coeff_dx = self.compliance_corr_decoder(interaction_latent)

        fij = (coeff_f[:, 0:1] * vector_a + coeff_f[:, 1:2] * vector_b + coeff_f[:, 2:3] * vector_c)
        aij = (coeff_a[:, 0:1] * vector_a + coeff_a[:, 1:2] * vector_b + coeff_a[:, 2:3] * vector_c)
        dxij = (coeff_dx[:, 0:1] * vector_a + coeff_dx[:, 1:2] * vector_b + coeff_dx[:, 2:3] * vector_c)

        # ---------------------------------------------------------------------
        # 2. Self-Equilibrating Constraint (Flattened Logic)
        # ---------------------------------------------------------------------
        is_global = (node_type[:, -1] == -1)
        is_virtual = (edge_attr == -1).view(-1)

        # Helper: Calculate mean of a group and subtract it from the edges
        def remove_mean(tensor, mask, group_indices):
            if mask.any():
                # 1. Compute Mean per group (e.g. per Global Node)
                mean_val = scatter_mean(tensor[mask], group_indices[mask], dim=0, dim_size=num_nodes)
                # 2. Subtract Mean (Broadcast back to edges)
                tensor[mask] -= mean_val[group_indices[mask]]
            return tensor

        # A. Incoming Correction (Real -> Global)
        # We ensure the Global Node RECEIVES zero net force.
        mask_in = is_virtual & is_global[receivers]
        fij  = remove_mean(fij,  mask_in, receivers)
        aij  = remove_mean(aij,  mask_in, receivers)
        dxij = remove_mean(dxij, mask_in, receivers)

        # B. Outgoing Correction (Global -> Real)
        # We ensure the Global Node EXERTS zero net force.
        mask_out = is_virtual & is_global[senders]
        fij  = remove_mean(fij,  mask_out, senders)
        aij  = remove_mean(aij,  mask_out, senders)
        dxij = remove_mean(dxij, mask_out, senders)

        # ---------------------------------------------------------------------
        # 3. Torque Calculation
        # ---------------------------------------------------------------------
        w_s, w_r = w_nodes[senders], w_nodes[receivers]
        
        # Weighted Center r0ij
        r0ij = (w_s * senders_pos + w_r * receivers_pos) / (w_s + w_r)
        
        lever_arm = receivers_pos - r0ij
        torque_contribution = torch.cross(lever_arm, fij * lambda_ij, dim=1)
        tauij = aij - torque_contribution

        return fij, tauij, dxij


class Node_Internal_Dv_Decoder(torch.nn.Module):
    def __init__(self, latent_size=128, mlp_layers=2):
        super(Node_Internal_Dv_Decoder, self).__init__()
        
        self.inv_mass_decoder = build_mlp_d(latent_size, latent_size, 1, num_layers=mlp_layers, lay_norm=False)
        self.inv_inertia_decoder = build_mlp_d(latent_size, latent_size, 1, num_layers=mlp_layers, lay_norm=False)

    def forward(self, edge_index, node_latent, edge_forces, edge_torques):
        """
        Assumes Bidirectional Graph:
        Every physical connection exists twice in edge_index: (i,j) and (j,i).
        Therefore, we only aggregate on 'receivers'.
        """
        _, receivers = edge_index   
        num_nodes = node_latent.shape[0]
        
        # 1. Decode Node Properties
        inverse_mass_dt = self.inv_mass_decoder(node_latent)
        inverse_inertia_dt = self.inv_inertia_decoder(node_latent)
        #blend_weights = self.blend_weight_decoder(node_latent)
        
        # 2. Aggregate Forces (Soft Newton's 3rd Law)
        # Since graph is bidirectional, the reaction force comes from the reverse edge.
        # We simply sum all incoming forces.
        net_force = node_latent.new_zeros((num_nodes, 3))
        net_force.index_add_(0, receivers, edge_forces)
        
        net_torque = node_latent.new_zeros((num_nodes, 3))
        net_torque.index_add_(0, receivers, edge_torques)

        # 3. Aggregate Constraints (PBD Consensus)
        # Since every neighbor sends a constraint via an incoming edge, 
        # averaging 'receivers' covers all physical connections.

        # 4. Compute Dynamic Updates
        delta_velocity_int = inverse_mass_dt * net_force 
        delta_angular_velocity_int = inverse_inertia_dt * net_torque

        return delta_velocity_int, delta_angular_velocity_int

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
    def __init__(self, edge_in_f, latent_size,mlp_layers):
        super(Interaction_Block, self).__init__()
        self.interaction_encoder = InteractionEncoder(edge_in_f, latent_size,mlp_layers)
        self.interaction_decoder = InteractionDecoder(latent_size,mlp_layers)
        self.internal_dv_decoder = Node_Internal_Dv_Decoder(latent_size,mlp_layers)
        self.external_dv_decoder = Node_External_Dv_Decoder(latent_size,mlp_layers)
        #self.velocity_scaler = build_mlp_d(latent_size,latent_size,1,lay_norm=False)
        self.layer_norm = nn.LayerNorm(latent_size)

    def forward(
        self, 
        edge_index, 
        senders_pos, 
        receivers_pos, 
        edge_dx_, 
        edge_attr, 
        vector_a, vector_b, vector_c, 
        senders_v_t_, senders_w_t_,
        receivers_v_t_, receivers_w_t_,
        node_latent, 
        node_type, 
        node_weights,
        node_vel,   
        residue=None, latent_history=False
        ):


        interaction_latent = self.interaction_encoder(
            edge_index, edge_dx_, edge_attr,
            vector_a, vector_b, vector_c,
            senders_v_t_, senders_w_t_, 
            receivers_v_t_, receivers_w_t_, 
            node_latent
        )

        # Residual connection
        if latent_history and residue is not None:
            interaction_latent = self.layer_norm(interaction_latent + residue)


        node_dv_ext = self.external_dv_decoder(node_latent, node_type, node_weights, edge_index, edge_attr)
        
        # Decode forces and torques
        edge_force, edge_tau, edge_corr = self.interaction_decoder(
            edge_index, edge_attr, senders_pos, receivers_pos, 
            vector_a, vector_b, vector_c, 
            interaction_latent, node_weights, node_type
        )
        # Decode node updates
        node_dv_int, node_dw_int = self.internal_dv_decoder(
            edge_index, node_latent, edge_force, edge_tau
        )

        node_dv = node_dv_int + node_dv_ext

        updated_velocity = node_vel + node_dv
        senders,receivers =edge_index

        static_disp = scatter_mean(edge_corr, receivers, dim=0, dim_size=node_dv.shape[0])

        dynamic_disp = updated_velocity
        
        node_dx = dynamic_disp #+ static_disp

        return node_dv, node_dw_int, node_dx, interaction_latent

class DynamicsSolver(torch.nn.Module):
    def __init__(self, node_in_f, edge_in_f, time_step, train_stats, num_msgs=1, latent_size=128, mlp_layers=2):
        super(DynamicsSolver, self).__init__()
        self.refframecalc = RefFrameCalc()
        self.scaler = Scaler()
        self.node_encoder = NodeEncoder(node_in_f,latent_size,mlp_layers)
        self.global_updater = GlobalKinematicsUpdater(latent_size)
        interaction_layers= []

        for _ in range(num_msgs):
            interaction_layers.append(Interaction_Block(edge_in_f,latent_size,mlp_layers))

        self.interaction_layers = nn.ModuleList(interaction_layers)

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
        mask_body = (graph.node_type[:,-1:] != -100).squeeze()

        # Initialize Accumulators
        sum_node_dv = torch.zeros_like(vel)
        sum_node_dx = torch.zeros_like(vel)
        
        # Pre-compute Node Latent (static per step)
        node_latent = self.node_encoder(node_type)

        pos, prev_vel, vel, node_weights = self.global_updater(
            pos=pos,
            prev_vel = prev_vel, 
            vel=vel,
            node_latent=node_latent, 
            edge_index=edge_index, 
            edge_attr=edge_attr,
            node_type=node_type
        )
        # Initial State
        node_v_t = vel
        node_v_tm1 = prev_vel
        # Get or initialize angular quantities (optimization: use safe defaults directly)
        node_w_t = getattr(graph, 'node_w_t', torch.zeros_like(vel))        

        # Initialize iteration vars
        current_pos = pos        
        # Residue state for RNN-like message passing
        past_edge_latent = None

        for (i,layer) in enumerate(self.interaction_layers):
            # Gather node attributes for edges
            # Optimization: Doing this inside loop is necessary as values update
            s_vt = node_v_t[senders]
            r_vt = node_v_t[receivers]
            s_vtm1 = node_v_tm1[senders]
            r_vtm1 = node_v_tm1[receivers]
            s_wt = node_w_t[senders]
            r_wt = node_w_t[receivers]
            current_edge_dx = current_pos[receivers] - current_pos[senders]

            # Scaling
            s_vt_, s_vtm1_, r_vt_, r_vtm1_, edge_dx_ = self.scaler(
                s_vt, 
                s_vtm1, 
                r_vt, 
                r_vtm1, 
                current_edge_dx, 
                self.train_stats
            )

            # Reference Frame
            vec_a, vec_b, vec_c = self.refframecalc(
                edge_index, 
                current_pos[senders], 
                current_pos[receivers],
                s_vt, 
                r_vt, 
                s_vtm1, 
                r_vtm1, 
                s_wt, 
                r_wt
            )

            # Interaction Block
            history_flag = (i > 0)
            
            node_dv, node_dw, node_dx, past_edge_latent = layer(
                edge_index, 
                current_pos[senders], 
                current_pos[receivers],
                edge_dx_, 
                edge_attr, 
                vec_a, vec_b, vec_c,
                s_vt_, s_wt,
                r_vt_, r_wt,
                node_latent, 
                node_type, 
                node_weights,
                node_v_t,
                residue=past_edge_latent, 
                latent_history=history_flag
            )

            sum_node_dv += node_dv[mask_body]
            
            # Update Velocity
            node_vf = node_v_t.clone()
            node_vf[mask_body] += node_dv[mask_body]
            
            # Update Angular Velocity
            node_wf = node_w_t.clone()
            node_wf[mask_body] += node_dw[mask_body]

            # Calculate Displacement for this sub-step
            step_disp = node_dx
            sum_node_dx[mask_body] += step_disp[mask_body]

            # Update Position
            current_pos = current_pos + step_disp # Out-of-place to preserve graph.pos if needed elsewhere, but safe here.

            # Prepare for next iteration
            node_v_tm1 = node_v_t 
            node_v_t = node_vf
            node_w_t = node_wf
            
            # Update edge vector for next step
        return sum_node_dv, sum_node_dx