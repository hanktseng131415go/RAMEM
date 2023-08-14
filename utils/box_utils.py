# -*- coding: utf-8 -*-
import torch
from itertools import product
from math import sqrt
import numpy as np

INF = 100000000

# =============================================================================
# 1. Support Component
# =============================================================================

def make_anchors(cfg, conv_h, conv_w, scale):
    prior_data = []
    # Iteration order is important (it has to sync up with the convout)
    for j, i in product(range(conv_h), range(conv_w)):
        # + 0.5 because priors are in center
        x = (i + 0.5) / conv_w
        y = (j + 0.5) / conv_h

        for ar in cfg.aspect_ratios:
            ar = sqrt(ar)
            w = scale * ar / cfg.img_size
            h = scale / ar / cfg.img_size

            prior_data += [x, y, w, h]

    return prior_data

def encode(matched, priors):
    variances = [0.1, 0.2]

    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]  # 10 * (Xg - Xa) / Wa
    g_cxcy /= (variances[0] * priors[:, 2:])  # 10 * (Yg - Ya) / Ha
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]  # 5 * log(Wg / Wa)
    g_wh = torch.log(g_wh) / variances[1]  # 5 * log(Hg / Ha)
    # return target for smooth_l1_loss
    offsets = torch.cat([g_cxcy, g_wh], 1)  # [num_priors, 4]

    return offsets

def sanitize_coordinates(_x1, _x2, img_size, padding=0):
    """
    Sanitizes the input coordinates so that x1 < x2, x1 != x2, x1 >= 0, and x2 <= image_size.
    Also converts from relative to absolute coordinates and casts the results to long tensors.

    Warning: this does things in-place behind the scenes so copy if necessary.
    """
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size

    x1 = torch.min(_x1, _x2)
    x2 = torch.max(_x1, _x2)
    x1 = torch.clamp(x1 - padding, min=0)
    x2 = torch.clamp(x2 + padding, max=img_size)

    return x1, x2

def sanitize_coordinates_numpy(_x1, _x2, img_size, padding=0):
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size

    x1 = np.minimum(_x1, _x2)
    x2 = np.maximum(_x1, _x2)
    x1 = np.clip(x1 - padding, a_min=0, a_max=1000000)
    x2 = np.clip(x2 + padding, a_min=0, a_max=img_size)

    return x1, x2


def crop(masks, boxes, padding=1):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """
    h, w, n = masks.size()
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding)

    rows = torch.arange(w, device=masks.device, dtype=x1.dtype).view(1, -1, 1).expand(h, w, n)
    cols = torch.arange(h, device=masks.device, dtype=x1.dtype).view(-1, 1, 1).expand(h, w, n)

    masks_left = rows >= x1.view(1, 1, -1)
    masks_right = rows < x2.view(1, 1, -1)
    masks_up = cols >= y1.view(1, 1, -1)
    masks_down = cols < y2.view(1, 1, -1)

    crop_mask = masks_left * masks_right * masks_up * masks_down

    return masks * crop_mask.float()

def crop_numpy(masks, boxes, padding=1):
    h, w, n = masks.shape
    x1, x2 = sanitize_coordinates_numpy(boxes[:, 0], boxes[:, 2], w, padding)
    y1, y2 = sanitize_coordinates_numpy(boxes[:, 1], boxes[:, 3], h, padding)

    rows = np.tile(np.arange(w)[None, :, None], (h, 1, n))
    cols = np.tile(np.arange(h)[:, None, None], (1, w, n))

    masks_left = rows >= (x1.reshape(1, 1, -1))
    masks_right = rows < (x2.reshape(1, 1, -1))
    masks_up = cols >= (y1.reshape(1, 1, -1))
    masks_down = cols < (y2.reshape(1, 1, -1))

    crop_mask = masks_left * masks_right * masks_up * masks_down

    return masks * crop_mask

def box_iou(box_a, box_b, detail=False):
    """
    Compute the IoU of two sets of boxes.
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    (n, A), B = box_a.shape[:2], box_b.shape[1]
    # add a dimension
    box_a = box_a[:, :, None, :].expand(n, A, B, 4)
    box_b = box_b[:, None, :, :].expand(n, A, B, 4)

    max_xy = torch.min(box_a[..., 2:], box_b[..., 2:])
    min_xy = torch.max(box_a[..., :2], box_b[..., :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    inter_area = inter[..., 0] * inter[..., 1]

    area_a = (box_a[..., 2] - box_a[..., 0]) * (box_a[..., 3] - box_a[..., 1])
    area_b = (box_b[..., 2] - box_b[..., 0]) * (box_b[..., 3] - box_b[..., 1])
    
    if detail == True:
        return area_a.squeeze(0), area_b.squeeze(0), inter_area.squeeze(0)
    else:
        out = inter_area / (area_a + area_b - inter_area)
        return out if use_batch else out.squeeze(0)

def box_iou_numpy(box_a, box_b):
    (_, A), B = box_a.shape[:2], box_b.shape[1]
    # add a dimension
    box_a = np.tile(box_a[:, :, None, :], (1, 1, B, 1))
    box_b = np.tile(box_b[:, None, :, :], (1, A, 1, 1))

    max_xy = np.minimum(box_a[..., 2:], box_b[..., 2:])
    min_xy = np.maximum(box_a[..., :2], box_b[..., :2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=100000)
    inter_area = inter[..., 0] * inter[..., 1]

    area_a = (box_a[..., 2] - box_a[..., 0]) * (box_a[..., 3] - box_a[..., 1])
    area_b = (box_b[..., 2] - box_b[..., 0]) * (box_b[..., 3] - box_b[..., 1])

    return inter_area / (area_a + area_b - inter_area)

def mask_iou(mask1, mask2):
    """
    Inputs inputs are matricies of size _ x N. Output is size _1 x _2.
    Note: if iscrowd is True, then mask2 should be the crowd.
    """
    intersection = torch.matmul(mask1, mask2.t())
    area1 = torch.sum(mask1, dim=1).reshape(1, -1)
    area2 = torch.sum(mask2, dim=1).reshape(1, -1)
    union = (area1.t() + area2) - intersection
    ret = intersection /  torch.max(union, torch.tensor(1e-12))

    return ret.cpu()

def mask_iou_cuda(mask1, mask2, detail=False):
    """
    Inputs inputs are matricies of size _ x N. Output is size _1 x _2.
    Note: if iscrowd is True, then mask2 should be the crowd.
    """
    intersection = torch.matmul(mask1, mask2.t())
    area1 = torch.sum(mask1, dim=1).reshape(1, -1)
    area2 = torch.sum(mask2, dim=1).reshape(1, -1)
    union = (area1.t() + area2) - intersection
    ret = intersection / torch.max(union, torch.tensor(1e-12))
    
    if detail == True:
        return area1, area2, union
    else:
        return ret
    
def ma_iou(cfg, bboxes1, bboxes2, gt_masks, eps=1e-6):
    """Calculate maIoU between two set of bboxes and one set of mask.
    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4)
        gt_masks (Tensor): shape (m, 4) 

    Returns:
        maious(Tensor): shape (m, n) 
        ious(Tensor): shape (m, n) 
    """

    # Compute IoUs
    rows = bboxes1.size(0)
    cols = bboxes2.size(0)

    if rows * cols == 0:
        return bboxes1.new_zeros(rows, cols), bboxes1.new_zeros(rows, cols), None

    lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
    rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

    wh = (rb - lt).clamp(min=0)  # [rows, cols, 2]
    intersection = wh[:, :, 0] * wh[:, :, 1]
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])

    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    union = area1[:, None] + area2 - intersection

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = intersection / union

    with torch.no_grad():
        # For efficiency only consider IoU>0 for maIoU computation
        larger_ind = ious > 0.
        overlap = bboxes1.new_zeros(rows, cols).cuda()
        if not torch.is_tensor(gt_masks):
            # all_gt_masks = gt_masks.to_tensor(torch.bool, bboxes1.get_device())
            all_gt_masks = gt_masks
        else:
            if cfg.cuda:
                # all_gt_masks = gt_masks.type(torch.cuda.BoolTensor)
                all_gt_masks = gt_masks
            else:
                # all_gt_masks = gt_masks.type(torch.BoolTensor)
                all_gt_masks = gt_masks
                
        gt_number, image_h, image_w = all_gt_masks.size()

        # Compute integral image for all ground truth masks (Line 2 of Alg.1 in the paper)
        if cfg.cuda:
            integral_images = integral_image_compute(cfg, all_gt_masks, gt_number, image_h, image_w).type(
                torch.cuda.FloatTensor)
        else:
            integral_images = integral_image_compute(cfg, all_gt_masks, gt_number, image_h, image_w).type(
                            torch.FloatTensor)
        # MOB Ratio
        MOB_ratio = integral_images[:, -1, -1] / (area1 + eps)

        # For each ground truth compute maIoU
        for i in range(gt_number):
            if cfg.cuda:
                all_boxes = torch.round(bboxes2[larger_ind[i]].clone()).type(torch.cuda.IntTensor)
            else:
                all_boxes = torch.round(bboxes2[larger_ind[i]].clone()).type(torch.IntTensor)
            all_boxes = torch.clamp(all_boxes, min=0)
            all_boxes[:, 2] = torch.clamp(all_boxes[:, 2], max=image_w)
            all_boxes[:, 3] = torch.clamp(all_boxes[:, 3], max=image_h)
            # Compute mask-aware intersection (Eq. 3 in the paper)
            overlap[i, larger_ind[i]] = integral_image_fetch(integral_images[i], all_boxes)/MOB_ratio[i]
        # Normaling mask-aware intersection by union yields maIoU (Eq. 5)
        maious = overlap / union

    return maious, ious, MOB_ratio

def integral_image_compute(cfg, masks, gt_number, h, w):
    integral_images = [None] * gt_number
    if cfg.cuda:
        pad_row = torch.zeros([gt_number, 1, w]).type(torch.cuda.BoolTensor)
        pad_col = torch.zeros([gt_number, h + 1, 1]).type(torch.cuda.BoolTensor)
    else:
        pad_row = torch.zeros([gt_number, 1, w]).type(torch.BoolTensor)
        pad_col = torch.zeros([gt_number, h + 1, 1]).type(torch.BoolTensor)
            
    integral_images = torch.cumsum(
        torch.cumsum(torch.cat([pad_col, torch.cat([pad_row, masks], dim=1)], dim=2), dim=1), dim=2)
    return integral_images

def integral_image_fetch(mask, bboxes):
    TLx = bboxes[:, 0].long()
    TLy = bboxes[:, 1].long()
    BRx = bboxes[:, 2].long()
    BRy = bboxes[:, 3].long()
    area = mask[BRy, BRx] + mask[TLy, TLx] - mask[TLy, BRx] - mask[BRy, TLx]
    return area

def k_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
    
    with torch.no_grad():
        targets = torch.empty(size=(targets.size(0), n_classes),
                              device=targets.device) \
                              .fill_(smoothing /(n_classes-1)) \
                              .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
    return targets

# =============================================================================
# 2. Label Assignment
# =============================================================================

def simota(cfg, box_gt, anchors, class_gt, gt_masks, num_level_bboxes, box_p, class_p):
    
    topk = 13
    iou_weight = 3.
    cls_weight = 1.

    pix_p = torch.softmax(class_p, -1)
    
    
    
    decoded_box = torch.cat((anchors[:, :2] + box_p[:, :2] * 0.1 * anchors[:, 2:],
                          anchors[:, 2:] * torch.exp(box_p[:, 2:] * 0.2)), 1)
    decoded_box[:, :2] -= decoded_box[:, 2:] / 2
    decoded_box[:, 2:] += decoded_box[:, :2]
        
    decoded_priors = torch.cat((anchors[:, :2] - anchors[:, 2:] / 2, anchors[:, :2] + anchors[:, 2:] / 2), 1)
    bboxes_priors = decoded_priors# * gt_masks.shape[1]
    bboxes_priors_cx = (bboxes_priors[:, 0] + bboxes_priors[:, 2]) / 2.0
    bboxes_priors_cy = (bboxes_priors[:, 1] + bboxes_priors[:, 3]) / 2.0
    bboxes_priors_points = torch.stack((bboxes_priors_cx, bboxes_priors_cy), dim=1)

    gt_bboxes = box_gt
    gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
    gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
    gt_points = torch.stack((gt_cx, gt_cy), dim=1)

    distance = torch.cdist(bboxes_priors_points, gt_points, p=2)  
    soft_center_prior = torch.pow(10, distance)

    overlaps = box_iou(box_gt, decoded_priors, detail=False) 

    bboxes = decoded_priors
    gt_bboxes = box_gt
    gt_labels = class_gt
    
    bboxes = bboxes * gt_masks.shape[1]
    gt_bboxes = gt_bboxes * gt_masks.shape[1]
    
    overlaps = overlaps.transpose(0, 1)
    num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

    assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                         0,
                                         dtype=torch.long)

    bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0

    ep_bboxes_cx = bboxes_cx.view(1, -1).expand(
        num_gt, num_bboxes).contiguous()#.view(-1)
    ep_bboxes_cy = bboxes_cy.view(1, -1).expand(
        num_gt, num_bboxes).contiguous()#.view(-1)
    l_ = ep_bboxes_cx.permute(1,0) - gt_bboxes[:, 0]
    t_ = ep_bboxes_cy.permute(1,0) - gt_bboxes[:, 1]
    r_ = gt_bboxes[:, 2] - ep_bboxes_cx.permute(1,0)
    b_ = gt_bboxes[:, 3] - ep_bboxes_cy.permute(1,0)
    box_min = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0]
    
    is_in_box = box_min > 0.01

    c_overlaps = overlaps * is_in_box
    
    iou_cost = -torch.log(torch.clamp(c_overlaps, min=torch.tensor(1e-8, device=c_overlaps.device)))

    gt_onehot_label = k_one_hot(gt_labels, class_p.size(-1), 0.)
    
    gt_onehot_label = gt_onehot_label * is_in_box[:, :, None]

    c_pix_p = pix_p.unsqueeze(1).repeat(1, num_gt, 1)

    soft_label = gt_onehot_label * c_overlaps[:, :, None]
    scale_factor = (soft_label - c_pix_p).abs().pow(2.0)

    cls_cost = - (torch.log(torch.clamp(c_pix_p, min=torch.tensor(1e-8, device=c_overlaps.device))) * soft_label)
    
    cls_cost = cls_cost * scale_factor
    cls_cost = cls_cost.sum(-1).to(dtype=pix_p.dtype)

    cost_matrix = soft_center_prior + iou_weight * iou_cost + cls_weight * cls_cost
    
    topk = torch.clamp(torch.round((c_overlaps).sum(0)), min=0.).int()
    
    _, candidate_idxs = (cost_matrix).topk(
                overlaps.shape[0], dim=0, largest=False)
    c_overlaps = (c_overlaps)[candidate_idxs, torch.arange(num_gt).cuda()]  
    
    for i, g in enumerate(topk):
        c_overlaps[g:, i] = 0.        
    
    is_pos = (c_overlaps > 0.)
    for gt_idx in range(num_gt):
        candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
           
    overlaps_inf = torch.full_like(overlaps, -float('inf')).t().contiguous().view(-1)   
    index = candidate_idxs.view(-1)[is_pos.view(-1)]
    overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]

    overlaps_inf = overlaps_inf.reshape(num_gt, -1).t()

    max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)

    assigned_gt_inds[max_overlaps != -float('inf')] = argmax_overlaps[max_overlaps != -float('inf')] + 1

    if gt_labels is not None:
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), 0)#.float()
        
        pos_inds = torch.nonzero(
            assigned_gt_inds > 0, as_tuple=False).squeeze()
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[
                assigned_gt_inds[pos_inds] - 1] + 1# * max_overlaps[pos_inds]
            
            
    overlaps, conf, anchor_max_i, is_pos = overlaps, assigned_labels, argmax_overlaps, overlaps_inf
    
    anchor_max_gt = box_gt[anchor_max_i] 
    
    offsets = encode(anchor_max_gt, anchors)
    
    return offsets, conf, anchor_max_gt, anchor_max_i, gt_masks, class_gt


def atss(cfg, box_gt, anchors, class_gt, gt_masks, num_level_bboxes):

    decoded_priors = torch.cat((anchors[:, :2] - anchors[:, 2:] / 2, anchors[:, :2] + anchors[:, 2:] / 2), 1)

    bboxes = decoded_priors
    gt_bboxes = box_gt
    gt_labels = class_gt

    overlaps, _, _ = ma_iou(cfg, box_gt*gt_masks.shape[1], decoded_priors*gt_masks.shape[1], gt_masks)
    
    bboxes = bboxes * gt_masks.shape[1]
    gt_bboxes = gt_bboxes * gt_masks.shape[1]
    
    overlaps = overlaps.transpose(0, 1)
    num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

    assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                         0,
                                         dtype=torch.long)
    
    # compute center distance between all bbox and gt
    gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
    gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
    gt_points = torch.stack((gt_cx, gt_cy), dim=1)
    
    bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
    bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0    
    
    topk = 9

    bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1) 
    
    distances = (bboxes_points[:, None, :] -
                  gt_points[None, :, :]).pow(2).sum(-1).sqrt()
    

    start_idx = 0
    candidate_idxs = []
    for level, bboxes_per_level in enumerate(num_level_bboxes):
        
        # on each pyramid level, for each gt,
        # select k bbox whose center are closest to the gt center
        end_idx = start_idx + bboxes_per_level
       
        distances_per_level = distances[start_idx:end_idx, :]
        
        if end_idx - start_idx < topk:
            
            _, topk_idxs_per_level = distances_per_level.topk(
            (end_idx - start_idx) , dim=0, largest=False)
        else:
            _, topk_idxs_per_level = distances_per_level.topk(
            topk, dim=0, largest=False)
        
        candidate_idxs.append(topk_idxs_per_level + start_idx)
        start_idx = end_idx
    
    
    candidate_idxs = torch.cat(candidate_idxs, dim=0)  
    candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt).cuda()]
    
    overlaps_mean_per_gt = candidate_overlaps.mean(0)
    overlaps_std_per_gt = candidate_overlaps.std(0)
    overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt
    
    is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]
    
    for gt_idx in range(num_gt):
        candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        
    ep_bboxes_cx = bboxes_cx.view(1, -1).expand(
        num_gt, num_bboxes).contiguous().view(-1)
    ep_bboxes_cy = bboxes_cy.view(1, -1).expand(
        num_gt, num_bboxes).contiguous().view(-1)
    candidate_idxs = candidate_idxs.view(-1)    

    l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
    t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
    r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
    b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
    is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
    
    is_pos = is_pos & is_in_gts
    
    overlaps_inf = torch.full_like(overlaps, -float('inf')).t().contiguous().view(-1)   
    index = candidate_idxs.view(-1)[is_pos.view(-1)]
    overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]

    overlaps_inf = overlaps_inf.reshape(num_gt, -1).t()

    max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
    
    assigned_gt_inds[max_overlaps != -float('inf')] = argmax_overlaps[max_overlaps != -float('inf')] + 1

    if gt_labels is not None:
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), 0)#.float()
        
        pos_inds = torch.nonzero(
            assigned_gt_inds > 0, as_tuple=False).squeeze()
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[
                assigned_gt_inds[pos_inds] - 1] + 1# * max_overlaps[pos_inds]
            
    overlaps, conf, anchor_max_i, is_pos = overlaps, assigned_labels, argmax_overlaps, overlaps_inf
    
    anchor_max_gt = box_gt[anchor_max_i] 
    offsets = encode(anchor_max_gt, anchors)
    
    return offsets, conf, anchor_max_gt, anchor_max_i, gt_masks, class_gt

def iou(cfg, box_gt, anchors, class_gt, gt_masks, num_level_bboxes):

    decoded_priors = torch.cat((anchors[:, :2] - anchors[:, 2:] / 2, anchors[:, :2] + anchors[:, 2:] / 2), 1)
    overlaps = box_iou(box_gt, decoded_priors, detail=False) 

    _, gt_max_i = overlaps.max(1)  # (num_gts, ) the max IoU for each gt box
    each_anchor_max, anchor_max_i = overlaps.max(0)  # (num_achors, ) the max IoU for each anchor
    
    # For the max IoU anchor for each gt box, set its IoU to 2. This ensures that it won't be filtered
    # in the threshold step even if the IoU is under the negative threshold. This is because that we want
    # at least one anchor to match with each gt box or else we'd be wasting training data.
    each_anchor_max.index_fill_(0, gt_max_i, 2)

    # Set the index of the pair (anchor, gt) we set the overlap for above.
    for j in range(gt_max_i.size(0)):
        anchor_max_i[gt_max_i[j]] = j
    
    anchor_max_gt = box_gt[anchor_max_i] # (num_achors, 4)
    
    conf = class_gt[anchor_max_i] + 1  # the class of the max IoU gt box for each anchor
    conf[each_anchor_max < cfg.pos_iou_thre] = -1  # label as neutral
    conf[each_anchor_max < cfg.neg_iou_thre] = 0  # label as background
    
    offsets = encode(anchor_max_gt, anchors)
   
    
    return offsets, conf, anchor_max_gt, anchor_max_i, gt_masks, class_gt

def match(cfg, box_gt, anchors, class_gt, gt_masks, num_level_bboxes, box_p, class_p):
    
    if cfg.label_assignment == 'iou':
        return iou(cfg, box_gt, anchors, class_gt, gt_masks, num_level_bboxes)    
    if cfg.label_assignment == 'maiou':
        return atss(cfg, box_gt, anchors, class_gt, gt_masks, num_level_bboxes)    
    if cfg.label_assignment == 'soft-simota':
        return simota(cfg, box_gt, anchors, class_gt, gt_masks, num_level_bboxes, box_p, class_p)    
    