
import torch

def evaluate_f1_precision_recall(preds, targets, eps=1e-11):
    # 需要先sigmoid才能用
    tp_map = torch.sign(torch.Tensor(preds) * torch.Tensor(targets))
    tp = tp_map.sum()
    pred_p = torch.sign(torch.Tensor(preds)).sum()
    true_p = targets.sum()
    fp = pred_p - tp
    fn = true_p - tp
    recall = (tp + eps) / (tp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    f1_score = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return precision, recall, f1_score


def rna_evaluation(preds, targets, eps=1e-11):   # B L L     B L L   set_max_len
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    tp = torch.sum(preds * targets)
    tn = torch.sum((1 - preds) * (1 - targets))
    fp = torch.sum(preds * (1 - targets))
    fn = torch.sum((1 - preds) * targets)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps) # accuracy
    prec = (tp + eps) / (tp + fp + eps)  # precision
    recall = (tp + eps) / (tp + fn + eps)  # recall
    sens = (tp + eps) / (tp + fn + eps)  # senstivity
    spec = (tn + eps) / (tn + fp + eps)  # spec

    F1 = 2 * ((prec * sens) / (prec + sens))
    MCC = (tp * tn - fp * fn) / torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return accuracy, prec, recall, sens, spec, F1, MCC.cpu().item()





preds= torch.randint(0,2,(100,100))
targets= torch.randint(0,2,(100,100))
r1=evaluate_f1_precision_recall(preds, targets)
r2=rna_evaluation(preds, targets)
print('r1:',r1)
print('r2:',r2)