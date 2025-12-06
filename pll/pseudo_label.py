
import torch
import torch.nn.functional as F

@torch.no_grad()
def class_temperature_scaling(probs, class_freq, beta=1.0, gamma=0.5, eps=1e-8):
    freq = class_freq.clamp_min(1.0)
    freq = freq / freq.sum()
    T = beta * (freq ** gamma)
    T = T / (T.mean() + eps)
    adj = probs.clamp_min(1e-8).pow(1.0 / (T.unsqueeze(0)))
    adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-8)
    return adj

@torch.no_grad()
def tri_consistency_select(s1_logits, s2_logits, t1_logits, t2_logits, quantile=0.5):
    logits_list = [s1_logits, s2_logits, t1_logits, t2_logits]
    probs = [F.softmax(z, dim=1) for z in logits_list]
    preds = [p.argmax(dim=1) for p in probs]
    agree = (preds[0] == preds[1]).float() + (preds[0] == preds[2]).float() + (preds[1] == preds[2]).float() \
            + (preds[0] == preds[3]).float() + (preds[1] == preds[3]).float() + (preds[2] == preds[3]).float()
    agree = agree / 6.0
    maxp = torch.stack([p.max(dim=1).values for p in probs], dim=1).mean(dim=1)
    score = 0.5 * agree + 0.5 * maxp
    thr = torch.quantile(score, quantile)
    mask = score >= thr
    return mask, score
