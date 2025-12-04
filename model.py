import torch
import torch.nn as nn
from utils import build_mlp_d

class RefFrameCalc(nn.Module):
    def __init__(self):
        super(RefFrameCalc, self).__init__()

    def forward(self, edge_index,senders_pos,receivers_pos, senders_vel,receivers_vel, senders_omega, receivers_omega):
        
        senders, receivers = edge_index

        epsilon = 1e-6

        vector_a = (receivers_pos - senders_pos)/ torch.clamp((receivers_pos - senders_pos).norm(dim=1, keepdim=True), min=epsilon)

        #prelimnary vectors
        b_a = torch.cross(receivers_vel-senders_vel,vector_a,dim=1)
        b_a = b_a / torch.clamp(b_a.norm(dim=1, keepdim=True), min=epsilon)
        b_c = (senders_vel + receivers_vel)
        b_c = b_c / torch.clamp(b_c.norm(dim=1, keepdim=True), min=epsilon)
        
        b_a_ = torch.cross(receivers_omega-senders_omega,vector_a,dim=1)
        b_a_ = b_a_ / torch.clamp(b_a_.norm(dim=1, keepdim=True), min=epsilon)
        b_c_ = (senders_omega + receivers_omega)
        b_c_ = b_c_ / torch.clamp(b_c_.norm(dim=1, keepdim=True), min=epsilon)

        b = b_a + b_c + b_a_ + b_c_ 

        # Compute the parallel component of b
        b_prl_dot = torch.einsum('ij,ij->i', b, vector_a).unsqueeze(1)
        b_prl = b_prl_dot * vector_a
        

        # Compute the perpendicular component of b
        b_prp = b - b_prl

        vector_b = torch.cross(b_prp, vector_a,dim=1) #perp to a and a new vector b_prp
        vector_c = torch.cross(b_prl, vector_b,dim=1) #perp to a and b
        
        vector_b = vector_b / torch.clamp(vector_b.norm(dim=1, keepdim=True), min=epsilon)
        vector_c = vector_c / torch.clamp(vector_c.norm(dim=1, keepdim=True), min=epsilon)
   
        return vector_a, vector_b, vector_c
    
    

class NodeEncoder(nn.Module):
    def __init__(self, latent_size):
        super(NodeEncoder, self).__init__()
        self.node_encoder = build_mlp_d(2, latent_size, latent_size, num_layers=2, lay_norm=True)
    def forward(self,node_scalar_feat):
        node_latent = self.node_encoder(node_scalar_feat)  
        return node_latent

class InteractionEncoder(nn.Module):
    """Message passing."""

    def __init__(self, latent_size):
        super(InteractionEncoder, self).__init__()
        self.edge_feat_encoder = build_mlp_d(9, latent_size, latent_size, num_layers=2, lay_norm=True)
        self.edge_encoder = build_mlp_d(3, latent_size, latent_size, num_layers=2, lay_norm=True)
        self.interaction_encoder = build_mlp_d(3*latent_size,latent_size, latent_size, num_layers=2, lay_norm=True)
    def forward(self, edge_index, edge_dx_, edge_dt_, edge_attr, vector_a, vector_b, vector_c,
                senders_v_t_, senders_v_tm1_, senders_w_t_,
                receivers_v_t_, receivers_v_tm1_, receivers_w_t_,
                node_latent):

        senders, receivers = edge_index

        node_v_t_senders_a = torch.einsum('ij,ij->i', senders_v_t_, vector_a).unsqueeze(1)
        node_v_t_senders_b = torch.einsum('ij,ij->i', senders_v_t_, vector_b).unsqueeze(1)
        node_v_t_senders_c = torch.einsum('ij,ij->i', senders_v_t_, vector_c).unsqueeze(1)

        node_v_tm1_senders_a = torch.einsum('ij,ij->i', senders_v_tm1_, vector_a).unsqueeze(1)
        node_v_tm1_senders_b = torch.einsum('ij,ij->i', senders_v_tm1_, vector_b).unsqueeze(1)
        node_v_tm1_senders_c = torch.einsum('ij,ij->i', senders_v_tm1_, vector_c).unsqueeze(1)    

        node_w_t_senders_a = torch.einsum('ij,ij->i', senders_w_t_, vector_a).unsqueeze(1)
        node_w_t_senders_b = torch.einsum('ij,ij->i', senders_w_t_, vector_b).unsqueeze(1)
        node_w_t_senders_c = torch.einsum('ij,ij->i', senders_w_t_, vector_c).unsqueeze(1)  
        
        node_v_t_receivers_a = torch.einsum('ij,ij->i', receivers_v_t_, -vector_a).unsqueeze(1)
        node_v_t_receivers_b = torch.einsum('ij,ij->i', receivers_v_t_, -vector_b).unsqueeze(1)
        node_v_t_receivers_c = torch.einsum('ij,ij->i', receivers_v_t_, -vector_c).unsqueeze(1)

        node_v_tm1_receivers_a = torch.einsum('ij,ij->i',receivers_v_tm1_, -vector_a).unsqueeze(1)
        node_v_tm1_receivers_b = torch.einsum('ij,ij->i',receivers_v_tm1_, -vector_b).unsqueeze(1)
        node_v_tm1_receivers_c = torch.einsum('ij,ij->i',receivers_v_tm1_, -vector_c).unsqueeze(1)       
        
        node_w_t_receivers_a = torch.einsum('ij,ij->i', receivers_w_t_, -vector_a).unsqueeze(1)
        node_w_t_receivers_b = torch.einsum('ij,ij->i', receivers_w_t_, -vector_b).unsqueeze(1)
        node_w_t_receivers_c = torch.einsum('ij,ij->i', receivers_w_t_, -vector_c).unsqueeze(1)
        
        edge_dx_a_s = edge_dx_.norm(dim=1,keepdim=True)
        edge_dt_a_s = edge_dt_.norm(dim=1,keepdim=True)

        senders_features = torch.hstack((
            node_v_t_senders_a, node_v_t_senders_b, node_v_t_senders_c,
            node_v_tm1_senders_a, node_v_tm1_senders_b, node_v_tm1_senders_c,
            node_w_t_senders_a, node_w_t_senders_b, node_w_t_senders_c
        ))

        receivers_features = torch.hstack((
            node_v_t_receivers_a, node_v_t_receivers_b, node_v_t_receivers_c,
            node_v_tm1_receivers_a, node_v_tm1_receivers_b, node_v_tm1_receivers_c,
            node_w_t_receivers_a, node_w_t_receivers_b, node_w_t_receivers_c
        ))
        
        edge_latent = self.edge_encoder(torch.hstack((edge_dx_a_s, edge_dt_a_s, edge_attr)))

        senders_latent = self.edge_feat_encoder(senders_features)
        receivers_latent = self.edge_feat_encoder(receivers_features)

        interaction_latent = self.interaction_encoder(torch.hstack((senders_latent + receivers_latent,
                                                                    node_latent[senders]+node_latent[receivers],
                                                                    edge_latent)))

        return interaction_latent

class InteractionDecoder(torch.nn.Module):

    def __init__(self, latent_size=128):
        super(InteractionDecoder, self).__init__()
        self.i1_decoder = build_mlp_d(latent_size, latent_size, 3, num_layers=2, lay_norm=False)
        self.i2_decoder = build_mlp_d(latent_size, latent_size, 3, num_layers=2, lay_norm=False)
        self.f_scaler = build_mlp_d(latent_size, latent_size, 1, num_layers=2, lay_norm=False)
        self.node_weight_decoder = build_mlp_d(latent_size, latent_size, 1, num_layers=2, lay_norm=False)

    def forward(self, edge_index, senders_pos,receivers_pos, vector_a, vector_b, vector_c, interaction_latent, node_latent):
        senders, receivers = edge_index
        coeff_f = self.i1_decoder(interaction_latent)
        coeff_a = self.i2_decoder(interaction_latent)
        node_weights = self.node_weight_decoder(node_latent)

        lambda_ij = self.f_scaler(interaction_latent) # lambda_ij in paper for stabilization

                
        fij = (coeff_f[:, 0:1] * vector_a + 
              coeff_f[:, 1:2] * vector_b + 
              coeff_f[:, 2:] * vector_c)

        aij = (coeff_a[:, 0:1] * vector_a + 
              coeff_a[:, 1:2] * vector_b + 
              coeff_a[:, 2:] * vector_c)

        r0ij = ((node_weights[senders])*senders_pos + (node_weights[receivers])*receivers_pos)/(node_weights[senders] + node_weights[receivers])
        
        tauij = aij-torch.cross(receivers_pos-r0ij,fij,dim=1)*lambda_ij
        
        return fij, tauij

class Node_Internal_Dv_Decoder(torch.nn.Module):
    def __init__(self, latent_size=128):
        super(Node_Internal_Dv_Decoder, self).__init__()
        self.m_inv_decoder = build_mlp_d(latent_size, latent_size, 1, num_layers=2, lay_norm=False)
        self.i_inv_decoder = build_mlp_d(latent_size, latent_size, 1, num_layers=2, lay_norm=False)
        self.dv_ext_decoder = build_mlp_d(latent_size, latent_size, 1, num_layers=2, lay_norm=False)
    def forward(self,edge_index,node_latent,fij,tij):
        m_inv = self.m_inv_decoder(node_latent) # decode inverse of mass
        i_inv = self.i_inv_decoder(node_latent) # decode inverse of inertia
        senders,receivers = edge_index   
        device = node_latent.device
        out_fij = torch.zeros((node_latent.shape[0], fij.shape[1])).to(device)
        out_fij = out_fij.scatter_add(0, receivers.unsqueeze(1).expand(-1, fij.shape[1]).to(device), fij.to(device))
        node_dv_int = m_inv * (out_fij) + self.dv_ext_decoder(node_latent)
        
        out_tij = torch.zeros((node_latent.shape[0], fij.shape[1])).to(device)
        out_tij = out_tij.scatter_add(0, receivers.unsqueeze(1).expand(-1, fij.shape[1]).to(device), tij.to(device))
        node_dw_int = i_inv * out_tij

        return node_dv_int, node_dw_int




class Scaler(torch.nn.Module):
    def __init__(self):
        super(Scaler, self).__init__()
        '''
        Scales the velocity and angular velocity features by maximum magnitude of respective field in training data
        Scales the magnitude of the edge_vector_dx using min-max scaling. (keeping the direction of edge_vector_dx same)
        '''

    def forward(self, senders_v_t, senders_v_tm1,receivers_v_t, receivers_v_tm1,edge_dx,train_stats):
        stat_edge_dx, stat_node_v_t, _,_= train_stats
        
        senders_v_t_ = senders_v_t/stat_node_v_t[1].detach()
        senders_v_tm1_ = senders_v_tm1/stat_node_v_t[1].detach()
        receivers_v_t_ = receivers_v_t/stat_node_v_t[1].detach()
        receivers_v_tm1_ = receivers_v_tm1/stat_node_v_t[1].detach()
        norm_edge_dx = edge_dx.norm(dim=1, keepdim=True)
        edge_dx_ = (((norm_edge_dx-stat_edge_dx[0])/(stat_edge_dx[1]-stat_edge_dx[0]))*(edge_dx/norm_edge_dx)).detach()
        return senders_v_t_, senders_v_tm1_,receivers_v_t_, receivers_v_tm1_,edge_dx_


class Interaction_Block(torch.nn.Module):
    def __init__(self, latent_size):
        super(Interaction_Block, self).__init__()
        self.interaction_encoder = InteractionEncoder(latent_size)
        self.interaction_decoder = InteractionDecoder(latent_size)
        self.internal_dv_decoder = Node_Internal_Dv_Decoder(latent_size)
        self.layer_norm = nn.LayerNorm(latent_size)

    def forward(self, edge_index, senders_pos, receivers_pos, edge_dx_, edge_dt_, edge_attr,vector_a, vector_b, vector_c, 
                senders_v_t_, senders_v_tm1_, senders_w_t_,
                receivers_v_t_, receivers_v_tm1_, receivers_w_t_,
                node_latent, residue=None, latent_history=False):
            interaction_latent = self.interaction_encoder(edge_index, edge_dx_,edge_dt_,edge_attr,
                                                          vector_a, vector_b, vector_c,
                                                          senders_v_t_, senders_v_tm1_, senders_w_t_,
                                                          receivers_v_t_, receivers_v_tm1_, receivers_w_t_,
                                                          node_latent)

            if latent_history:
                interaction_latent = interaction_latent + residue
                interaction_latent = self.layer_norm(interaction_latent)
            
            edge_interaction_force, edge_interaction_tau= self.interaction_decoder(
                edge_index, senders_pos, receivers_pos, vector_a, vector_b, vector_c, interaction_latent, node_latent
            )
            node_dv_int_decoded, node_dw_int_decoded = self.internal_dv_decoder(
                edge_index, node_latent, edge_interaction_force, edge_interaction_tau
            )
        
            return node_dv_int_decoded, node_dw_int_decoded, interaction_latent



class DynamicsSolver(torch.nn.Module):
    def __init__(self, sample_step, train_stats, num_jumps=1, num_msgs=1, latent_size=128):
        super(DynamicsSolver, self).__init__()
        self.refframecalc = RefFrameCalc()
        self.scaler = Scaler()
        self.node_encoder = NodeEncoder(latent_size)
        self.interaction_proc_layer = Interaction_Block(latent_size)
        self.interaction_init_layer = Interaction_Block(latent_size)
        self.num_messages = num_msgs
        self.sub_tstep = sample_step / num_msgs
        self.train_stats = train_stats

    def forward(self, graph):
        # Initialize graph data for processing
        device = graph.pos.device
        graph = graph.to(device)
        #node_type = torch.ones_like(graph.disp[:,0:1]).float()# get_node_type(graph).float()
        node_type = graph.node_type.float()
        pos = graph.pos.float()
        vel = graph.vel.float()
        prev_vel = graph.prev_vel.float()
        
        edge_index = graph.edge_index.long()
        senders, receivers = edge_index
        senders_pos = pos[senders]
        receivers_pos = pos[receivers]
        edge_dx = receivers_pos - senders_pos
        edge_attr = graph.edge_attr.float()
        
        mask_reflected_node = (graph.node_type!=2).squeeze()

        node_v_t = vel
        node_w_t = getattr(graph, 'node_w_t',torch.zeros_like(node_v_t))
        node_th_t = getattr(graph, 'node_th_t', torch.zeros_like(node_v_t))

        senders_v_t = node_v_t[senders].float()
        receivers_v_t = node_v_t[receivers].float()

        senders_v_tm1 = prev_vel[senders].float()
        receivers_v_tm1 = prev_vel[receivers].float()

        senders_w_t = node_w_t[senders]
        receivers_w_t = node_w_t[receivers]
        
        senders_th_t = node_th_t[senders]
        receivers_th_t = node_th_t[receivers]        

        node_disp = torch.zeros_like(node_v_t)
        node_vf = torch.zeros_like(node_v_t)
        node_wf = torch.zeros_like(node_v_t)

        sum_node_dv = torch.zeros_like(node_v_t)
        sum_node_dx = torch.zeros_like(node_v_t)
        
        node_latent = self.node_encoder(torch.hstack((node_type,vel.norm(dim=1,keepdim=True))))
        edge_dt_ = receivers_th_t - senders_th_t

        for i in range(self.num_messages):
            (
                senders_v_t_,
                senders_v_tm1_,
                receivers_v_t_,
                receivers_v_tm1_,
                edge_dx_,
            ) = self.scaler(
                senders_v_t,
                senders_v_tm1,
                receivers_v_t,
                receivers_v_tm1,
                edge_dx,
                self.train_stats,
            )


            vector_a, vector_b, vector_c = self.refframecalc(
                    edge_index,
                    senders_pos,
                    receivers_pos,
                    senders_v_t_,
                    receivers_v_t_,
                    senders_w_t,
                    receivers_w_t
                    
                )

            if i == 0:
                node_dv_int_decoded, node_dw_int_decoded,residue= self.interaction_init_layer(
                    edge_index,
                    senders_pos,
                    receivers_pos,
                    edge_dx_,
                    edge_dt_,
                    edge_attr,
                    vector_a,
                    vector_b,
                    vector_c,
                    senders_v_t_,
                    senders_v_tm1_,
                    senders_w_t,
                    receivers_v_t_,
                    receivers_v_tm1_,
                    receivers_w_t,
                    node_latent,
                    latent_history=False,
                )
            else:
                node_dv_int_decoded, node_dw_int_decoded, residue = self.interaction_proc_layer(
                    edge_index,
                    senders_pos,
                    receivers_pos,
                    edge_dx_,
                    edge_dt_,
                    edge_attr,
                    vector_a,
                    vector_b,
                    vector_c,
                    senders_v_t_,
                    senders_v_tm1_,
                    senders_w_t,
                    receivers_v_t_,
                    receivers_v_tm1_,
                    receivers_w_t,
                    node_latent,
                    residue=residue,
                    latent_history=True,
                )

            sum_node_dv [mask_reflected_node]= sum_node_dv [mask_reflected_node] + node_dv_int_decoded[mask_reflected_node]
                
            node_vf[mask_reflected_node]= node_v_t[mask_reflected_node] + node_dv_int_decoded[mask_reflected_node]
            node_wf[mask_reflected_node]= node_w_t[mask_reflected_node] + node_dw_int_decoded[mask_reflected_node]

            node_disp= (
                (node_v_t + node_vf) * 0.5 * self.sub_tstep
            )

            node_th_t = node_th_t + (node_wf + node_w_t)* 0.5 * self.sub_tstep

            sum_node_dx [mask_reflected_node]=sum_node_dx [mask_reflected_node]+ ((node_v_t + node_vf) * 0.5 * self.sub_tstep)[mask_reflected_node]

            senders_disp = node_disp[senders]
            receivers_disp = node_disp[receivers]

            senders_pos = senders_disp + senders_pos
            receivers_pos = receivers_disp + receivers_pos


            node_v_tm1 = node_v_t.clone()

            node_v_t = node_vf.clone()
            node_w_t = node_wf.clone()

            

            senders_v_tm1 = senders_v_t.clone()
            senders_v_t = node_v_t[senders].clone()
            senders_w_t = node_w_t[senders].clone()
            senders_th_t = node_th_t[senders].clone()

            receivers_v_tm1 = receivers_v_t.clone()
            receivers_v_t = node_v_t[receivers].clone()
            receivers_w_t = node_w_t[receivers].clone()
            receivers_th_t = node_th_t[receivers].clone()

            edge_dx = receivers_pos - senders_pos
            edge_dt_ = receivers_th_t - senders_th_t
        return sum_node_dv,sum_node_dx, node_v_tm1