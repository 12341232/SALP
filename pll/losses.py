
import torch
import torch.nn.functional as F

def soft_ce_loss(logits, targets_soft):
    logp = F.log_softmax(logits, dim=1)
    loss = -(targets_soft * logp).sum(dim=1).mean()
    return loss

def kl_divergence(p_logits, q_logits, T=1.0):
    p = F.log_softmax(p_logits / T, dim=1)
    q = F.softmax(q_logits / T, dim=1)
    return F.kl_div(p, q, reduction="batchmean") * (T*T)

def mse_loss(p_logits, q_logits):
    p = F.softmax(p_logits, dim=1)
    q = F.softmax(q_logits, dim=1)
    return F.mse_loss(p, q)
