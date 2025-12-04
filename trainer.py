import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import evaluate

class Trainer:
    def __init__(self, model, optimizer, device, train_stats, save_model_dir):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_history = []
        self.extr_test_history = []
        self.gen_test_history = []
        self.model_dir = save_model_dir
        self.train_stats = train_stats
        self.mean_node_dv = self.train_stats[2][0]
        self.mean_node_disp = self.train_stats[3][0]
        self.std_node_dv = self.train_stats[2][1]
        self.std_node_disp = self.train_stats[3][1]
        # Initialize best test loss and best epoch.
        self.best_val_loss = float('inf')
        self.best_model_path = None
        self.best_epoch = None        

    def train(self, train_graph):
        self.model.train()
        pred_node_dvel,pred_node_disp,_= self.model(train_graph.to(self.device))

        actual_node_dvel = (train_graph.y_dv).float().to(self.device)
        actual_node_disp = (train_graph.y_dx).float().to(self.device)

        pred_node_dvel_ = (pred_node_dvel-self.mean_node_dv.detach())/self.std_node_dv.detach()
        pred_node_disp_ = (pred_node_disp-self.mean_node_disp.detach())/self.std_node_disp.detach()
        actual_node_dvel_ = (actual_node_dvel - self.mean_node_dv.detach())/self.std_node_dv.detach()
        actual_node_disp_ = (actual_node_disp-self.mean_node_disp.detach())/self.std_node_disp.detach()

        loss = F.mse_loss(pred_node_disp_,actual_node_disp_) + F.mse_loss(pred_node_dvel_,actual_node_dvel_)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.loss = loss.cpu().detach().numpy()
        self.train_history.append(self.loss)
    def test(self, test_loader, mode='val', epoch=None):
        self.model.eval()
        mean_pos_error = evaluate(test_loader,
                            self.model,
                            self.device)
        
        
        if mode == 'val':
            self.val_loss_pos = mean_pos_error
            self.gen_test_history.append(self.val_loss_pos)
            # Update best test loss and best epoch if the current loss is lower
            if epoch is not None and self.val_loss_pos < self.best_val_loss:
                self.best_val_loss = self.val_loss_pos
                self.best_epoch = epoch
                self.save_model(epoch)
        if mode == 'test':
            self.test_loss_pos = mean_pos_error

    def save_model(self,iteration):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if self.best_model_path and os.path.exists(self.best_model_path):
            os.remove(self.best_model_path)
        self.best_model_path = os.path.join(self.model_dir, 
                                 f'GenLoss_{self.val_loss_pos:.5f}mm_iter{iteration}.pth')
        torch.save(self.model.state_dict(), self.best_model_path)