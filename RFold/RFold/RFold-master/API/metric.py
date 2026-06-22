import torch


def evaluate_result(pred_a, true_a, eps=1e-11):
    tp_map = torch.sign(torch.Tensor(pred_a) * torch.Tensor(true_a))
    tp = tp_map.sum()
    pred_p = torch.sign(torch.Tensor(pred_a)).sum()
    true_p = true_a.sum()
    fp = pred_p - tp
    fn = true_p - tp
    tn = (torch.Tensor(true_a) == 0).sum() - fp  # TN计算方法：真正为负的样本数

    # recall = (tp + eps)/(tp+fn+eps)
    # precision = (tp + eps)/(tp+fp+eps)
    # f1_score = (2*tp + eps)/(2*tp + fp + fn + eps)
    # # 计算MCC
    # mcc = (tp * tn - fp * fn) / torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
    # 计算Recall、Precision和F1-score
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    f1_score = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0

    # 计算MCC
    mcc = (tp * tn - fp * fn) / torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (
                tn + fp) * (tn + fn) != 0 else 0
    return precision, recall, f1_score, mcc