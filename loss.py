from torch.nn import BCELoss
import torch
class Loss:
    def __init__(self, loss_choices:list=['bce'])->None:
        self.loss_choices = loss_choices
        self.criterion = BCELoss().cuda()

                loss_consistency=torch.tensor(0.).cuda()
                loss_pseudo = torch.tensor(0.).cuda()    

    def bce_loss(self, output, label, source):
        batch, num_class = output.shape
        loss = torch.tensor(0.).cuda()
        for b in range(batch):
            for c in range(num_class):
                if source[b] == 0
                    loss += self.criterion(output[b][c], label[b][c])
                elif source[b] == 1:
                    loss += self.criterion(output[b][c], label[b][c])
                
                if output[b][c] > 0.5:
                    tmp = output[b][c]+(1-output[b][c])/4.0
                else:
                    tmp = output[b][c] - output[b][c] / 4.0

    def consistency_loss(self):
        pass

    def pseudo_loss(self):
        pass

    def cal_loss(self, output, label):
        loss = self.BCELoss()
        if 'consistency_loss' in loss_choices:
            loss += self.consistency_loss()
        if 'pseudo_loss' in loss_choices:
            loss += self.pseudo_loss()
        return loss