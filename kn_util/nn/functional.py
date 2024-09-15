import torch


def calc_iou_matrix(pred_bds, targets, type="iou"):
    """
    pred_bds:   (N, 2)
    gt:         (M, 2)

    Return:
        (N, M)
    """
    min_ed = torch.minimum(pred_bds[:, 1].unsqueeze(1), targets[:, 1])
    max_ed = torch.maximum(pred_bds[:, 1].unsqueeze(1), targets[:, 1])
    min_st = torch.minimum(pred_bds[:, 0].unsqueeze(1), targets[:, 0])
    max_st = torch.maximum(pred_bds[:, 0].unsqueeze(1), targets[:, 0])

    I = torch.maximum(min_ed - max_st, torch.zeros_like(min_ed, dtype=torch.float, device=pred_bds.device))
    area_pred = pred_bds[:, 1] - pred_bds[:, 0]
    area_gt = targets[:, 1] - targets[:, 0]
    U = area_pred[:, None] + area_gt[None, :] - I
    Ac = max_ed - min_st

    iou = I / U

    if type == "iou":
        return iou
    elif type == "giou":
        return 0.5 * (iou + U / Ac)
    else:
        raise NotImplementedError()


@torch.no_grad()
def calc_iou_1d(pred_bds, gt, type="iou"):
    """make sure the range between [0, 1) to make loss function happy (giou)
    pred_bds:   (N, 2)
    gt:         (N, 2)

    Return:
        (N, ) where N is the number of boxes
    """
    min_ed = torch.minimum(pred_bds[:, 1], gt[:, 1])
    max_ed = torch.maximum(pred_bds[:, 1], gt[:, 1])
    min_st = torch.minimum(pred_bds[:, 0], gt[:, 0])
    max_st = torch.maximum(pred_bds[:, 0], gt[:, 0])

    I = torch.maximum(min_ed - max_st, torch.zeros_like(min_ed, dtype=torch.float, device=pred_bds.device))
    area_pred = pred_bds[:, 1] - pred_bds[:, 0]
    area_gt = gt[:, 1] - gt[:, 0]
    U = area_pred + area_gt - I
    Ac = max_ed - min_st

    iou = I / U

    if type == "iou":
        return iou
    elif type == "giou":
        return 0.5 * (iou + U / Ac)
    else:
        raise NotImplementedError()

def broadcast_all(tensors):
    shape_tensor = torch.concat([t.shape for t in tensors], dim=0)
    to_shape = shape_tensor.max(dim=0).values.tolist()
    new_tensors = []

    for t in tensors:
        new_tensors.append(t.expand(*to_shape))
    
    return new_tensors
    

def broadcast_concat(tensors, dim):
    """
    Concatenate tensors with broadcasting
    """
    new_tensors = broadcast_all(tensors)

    return torch.cat(new_tensors, dim=dim)

def broadcast_stack(tensors, dim):
    """
    Stack tensors with broadcasting
    """
    new_tensors = broadcast_all(tensors)

    return torch.stack(new_tensors, dim=dim)
