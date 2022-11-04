import torch.nn as nn
from torch import FloatTensor
import torch
import torch.nn.functional as F


# class FullyLoss(nn.Module):
#     def __init__(self):
#         super(FullyLoss, self).__init__()


class SemiLoss(nn.Module):
    def __init__(self, threshold=0.5, temperature=4.0)->None:
        """Calculate the loss, including BCE loss, Consistency loss and Pseudo loss

        Args:
            threshold (float, optional): Defaults to 0.5.
            temperature (float, optional): Defaults to 4.0.
        """ 
        super(SemiLoss, self).__init__()
        self.criterion = nn.BCELoss().cuda()
        self.temperature = temperature
        self.threshold = threshold
        self.out_sharp = [] 

    def forward(self, output:FloatTensor, output_consistency:FloatTensor, label:FloatTensor, source:list)->FloatTensor:
        """caculate loss function

        Args:
            output (FloatTensor): model output
            label (FloatTensor): ground truth
            source (list): data source

        Returns:
            FloatTensor: BCE loss 
        """        
        batch_size, num_class = output.shape
        loss = torch.tensor(0.).cuda()
        consistency_loss = torch.tensor(0.).cuda()
        pseudo_loss = torch.tensor(0.).cuda()   
        for b in range(batch_size):
            for c in range(num_class):
                if source[b] == 0:
                    loss += self.criterion(output[b][c], label[b][c])
                elif source[b] == 1:
                    loss += self.criterion(output[b][c], label[b][c])
                if output[b][c] > self.threshold:
                    out_sharp = output[b][c] + (1 - output[b][c]) / self.temperature
                else:
                    out_sharp = output[b][c] - output[b][c] / self.temperature
                pseudo_loss += F.mse_loss(output[b][c], out_sharp).cuda()
                consistency_loss += F.mse_loss(output_consistency[b][c], out_sharp).cuda()

        loss += consistency_loss
        loss += pseudo_loss
        loss /= batch_size
        return loss






class Loss:
    def __init__(self, threshold=0.5, temperature=4.0, loss_choices:list=['bce'])->None:
        """Calculate the loss, including BCE loss, Consistency loss and Pseudo loss

        Args:
            threshold (float, optional): Defaults to 0.5.
            temperature (float, optional): Defaults to 4.0.
            loss_choices (list, optional): Defaults to ['bce'].
        """        
        self.loss_choices = loss_choices
        self.criterion = nn.BCELoss().cuda()
        self.temperature = temperature
        self.threshold = threshold
        self.out_sharp = []
    

        

    def cal_loss(self, output:FloatTensor, output_consistency:FloatTensor, label:FloatTensor, source:list)->FloatTensor:
        """caculate loss function

        Args:
            output (FloatTensor): model output
            label (FloatTensor): ground truth
            source (list): data source

        Returns:
            FloatTensor: BCE loss 
        """        
        batch, num_class = output.shape
        loss = torch.tensor(0.).cuda()
        consistency_loss = torch.tensor(0.).cuda()
        pseudo_loss = torch.tensor(0.).cuda()   
        for b in range(batch):
            for c in range(num_class):
                if source[b] == 0:
                    loss += self.criterion(output[b][c], label[b][c])
                elif source[b] == 1:
                    loss += self.criterion(output[b][c], label[b][c])
                if output[b][c] > self.threshold:
                    out_sharp = output[b][c] + (1 - output[b][c]) / self.temperature
                else:
                    out_sharp = output[b][c] - output[b][c] / self.temperature
                pseudo_loss += F.mse_loss(output[b][c], out_sharp).cuda()
                consistency_loss += F.mse_loss(output_consistency[b][c], out_sharp).cuda()

        # if 'consistency_loss' in loss_choices:
        #     loss += consistency_loss
        # if 'pseudo_loss' in loss_choices:
        #     loss += pseudo_loss
        return loss