import torch
from torch import nn
from .memory import ContrastMemory

eps = 1e-7


class CRDLoss(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side

    Args:
        opt[0]: the dimension of student's feature
        opt[1]: the dimension of teacher's feature
        opt[3]: the dimension of the projection space
        opt[4]: number of negatives paired with each positive
        opt[5]: the temperature
        opt[6]: the momentum for updating the memory buffer
        opt[2]: the number of samples in the training set, therefor the memory buffer is: opt[2] x opt[3]
    """
    def __init__(self, opt):
        super(CRDLoss, self).__init__()
        self.embed_s = Embed(opt[0], opt[3])
        self.embed_t = Embed(opt[1], opt[3])
        self.contrast = ContrastMemory(opt[3], opt[2], opt[4], opt[5], opt[6])
        self.criterion_t = ContrastLoss(opt[2])
        self.criterion_s = ContrastLoss(opt[2])

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]

        Returns:
            The contrastive loss
        """
        #print(f_s)
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        print('idx before momery: ', idx.__class__)
        print('contract_idx: ', contrast_idx.__class__)
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        loss = s_loss + t_loss
        return loss


class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self[2])

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        #print('\n\n\n\nx class: ', x.__class__)
        #x.cuda()
        #x.to('cuda:0')
        #print(x)
        x.detach()
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
