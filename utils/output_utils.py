import torch.nn.functional as F
import cv2
from utils.box_utils import crop, box_iou, box_iou_numpy, crop_numpy, ma_iou
import torch
import numpy as np
from config import COLORS, COLORS_cuda

def fast_nms(box_thre, coef_thre, class_thre, cfg):
    # box_thre: [18525, 4]
    # coef_thre: [18525, 32]
    # class_thre: [20, 18525]

    # [20, 18428]
    class_thre, idx = class_thre.sort(1, descending=True)  # [80, 64 (the number of kept boxes)]
    
    # top_k = 200
    # [20, 18525] -> [20, 200]
    idx = idx[:, :cfg.top_k]
    # [20, 18525] -> [20, 200]
    class_thre = class_thre[:, :cfg.top_k]
    
    # 20, 200
    num_classes, num_dets = idx.size()
    # box_thre[4000, :] -> [4000, 4] -> [20, 200, 4]
    box_thre = box_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, 4)  # [80, 64, 4]
    # coef_thre[4000, :] -> [4000, 32]-> [20, 200, 32]
    coef_thre = coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]
    # ([20, 200, 4], [20, 200, 4]) -> [20, 200, 200]
    iou = box_iou(box_thre, box_thre)
    iou.triu_(diagonal=1)
    # [20, 200], _
    iou_max, _ = iou.max(1)
    
    num_gt, num_bboxes = iou_max.shape

    # Now just filter out the ones higher than the threshold
    # nms_iou_thre = 0.5
    # [20, 200]
    keep = (iou_max <= cfg.nms_iou_thre)
    # # Assign each kept detection to its corresponding class
    # # 20 -> [20, 200]
    class_ids = torch.arange(num_classes, device=box_thre.device)[:, None].expand_as(keep)
    
    class_ids, box_nms, coef_nms, class_nms = class_ids[keep], box_thre[keep], coef_thre[keep], class_thre[keep]
        
    # Only keep the top cfg.max_num_detections highest scores across all classes
    class_nms, idx = class_nms.sort(0, descending=True)

    # max_detections = 100
    # [100]
    idx = idx[:cfg.max_detections]
    # [100]
    class_nms = class_nms[:cfg.max_detections]
    
    
    # [100]
    class_ids = class_ids[idx]
    # [100, 4]
    box_nms = box_nms[idx]
    # [100, 32]

    coef_nms = coef_nms[idx]
    
    
    return box_nms, coef_nms, class_ids, class_nms

def fast_nms_numpy(box_thre, coef_thre, class_thre, cfg):
    # descending sort
    idx = np.argsort(-class_thre, axis=1)
    class_thre = np.sort(class_thre, axis=1)[:, ::-1]

    idx = idx[:, :cfg.top_k]
    class_thre = class_thre[:, :cfg.top_k]

    num_classes, num_dets = idx.shape
    box_thre = box_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, 4)  # [80, 64, 4]
    coef_thre = coef_thre[idx.reshape(-1), :].reshape(num_classes, num_dets, -1)  # [80, 64, 32]

    iou = box_iou_numpy(box_thre, box_thre)
    iou = np.triu(iou, k=1)
    iou_max = np.max(iou, axis=1)

    # Now just filter out the ones higher than the threshold
    keep = (iou_max <= cfg.nms_iou_thre)

    # Assign each kept detection to its corresponding class
    class_ids = np.tile(np.arange(num_classes)[:, None], (1, keep.shape[1]))

    class_ids, box_nms, coef_nms, class_nms = class_ids[keep], box_thre[keep], coef_thre[keep], class_thre[keep]

    # Only keep the top cfg.max_num_detections highest scores across all classes
    idx = np.argsort(-class_nms, axis=0)
    class_nms = np.sort(class_nms, axis=0)[::-1]

    idx = idx[:cfg.max_detections]
    class_nms = class_nms[:cfg.max_detections]

    class_ids = class_ids[idx]
    box_nms = box_nms[idx]
    coef_nms = coef_nms[idx]

    return box_nms, coef_nms, class_ids, class_nms

def traditional_nms(boxes, masks, scores, cfg):
    num_classes = scores.size(0)

    idx_lst, cls_lst, scr_lst = [], [], []

    # Multiplying by max_size is necessary because of how cnms computes its area and intersections
    boxes = boxes * cfg.img_size

    for _cls in range(num_classes):
        cls_scores = scores[_cls, :]
        conf_mask = cls_scores > cfg.nms_score_thre
        idx = torch.arange(cls_scores.size(0), device=boxes.device)

        cls_scores = cls_scores[conf_mask]
        idx = idx[conf_mask]

        if cls_scores.size(0) == 0:
            continue

        preds = torch.cat([boxes[conf_mask], cls_scores[:, None]], dim=1).cpu().numpy()
        keep = cnms(preds, cfg.nms_iou_thre)
        keep = torch.tensor(keep, device=boxes.device).long()

        idx_lst.append(idx[keep])
        cls_lst.append(keep * 0 + _cls)
        scr_lst.append(cls_scores[keep])

    idx = torch.cat(idx_lst, dim=0)
    class_ids = torch.cat(cls_lst, dim=0)
    scores = torch.cat(scr_lst, dim=0)

    scores, idx2 = scores.sort(0, descending=True)
    idx2 = idx2[:cfg.max_detections]
    scores = scores[:cfg.max_detections]

    idx = idx[idx2]
    class_ids = class_ids[idx2]

    # Undo the multiplication above
    return boxes[idx] / cfg.img_size, masks[idx], class_ids, scores


def nms(class_pred, box_pred, coef_pred, proto_out, anchors, cfg, seg_pred):
    
    class_p = class_pred.squeeze()  # [18525, 21]
    box_p = box_pred.squeeze()  # [18525, 4]
    coef_p = coef_pred.squeeze()  # [18525, 32]
    proto_p = proto_out.squeeze()  # og: [136, 136, 32], [100, 136, 136]

    if isinstance(anchors, list): 
        anchors = torch.tensor(anchors, device=class_p.device).reshape(-1, 4)
    
    class_p = torch.softmax(class_p , -1)
    
    #  [18525, 21] -> [21, 18525]
    class_p = class_p.transpose(1, 0).contiguous()
    
    # keep = (class_p.max(0)[1]!=0)

    # exclude the background class
    # [21, 18525] -> [20, 18525]
    class_p = class_p[1:, :] 
    

    # get the max score class of 18525 predicted boxes
    # [18524]
    class_p_max, _ = torch.max(class_p, dim=0)  
    
    
    # filter predicted boxes according the class score
    # [18525]
    keep = (class_p_max > cfg.nms_score_thre)

    # [20, 18525] -> [20, 18525]
    class_thre = class_p[:, keep]
    # box_thre: [18525, 4], anchor_thre: [18525, 4], coef_thre: [18525, 32]
    
    box_thre, anchor_thre, coef_thre = box_p[keep, :], anchors[keep, :], coef_p[keep, :]

    # decode boxes
    # cat([18525, 2], [18525, 2]) -> [18525, 4]
    box_thre = torch.cat((anchor_thre[:, :2] + box_thre[:, :2] * 0.1 * anchor_thre[:, 2:],
                          anchor_thre[:, 2:] * torch.exp(box_thre[:, 2:] * 0.2)), 1)
    box_thre[:, :2] -= box_thre[:, 2:] / 2
    box_thre[:, 2:] += box_thre[:, :2]

    box_thre = torch.clip(box_thre, min=0., max=1.)

    if class_thre.shape[1] == 0:
        return None, None, None, None, None
    else:
        if cfg.fast_nms == True:
            box_thre, coef_thre, class_ids, class_thre = fast_nms(box_thre, coef_thre, class_thre, cfg)
        if cfg.fast_nms == False:
            box_thre, coef_thre, class_ids, class_thre = traditional_nms(box_thre, coef_thre, class_thre, cfg)

        return class_ids, class_thre, box_thre, coef_thre, proto_p
    
def nms_numpy(class_pred, box_pred, coef_pred, proto_out, anchors, cfg):
    class_p = class_pred.squeeze()  # [19248, 81]
    box_p = box_pred.squeeze()  # [19248, 4]
    coef_p = coef_pred.squeeze()  # [19248, 32]
    proto_p = proto_out.squeeze()  # [138, 138, 32]
    anchors = np.array(anchors).reshape(-1, 4)

    class_p = class_p.transpose(1, 0)
    # exclude the background class
    class_p = class_p[1:, :]
    # get the max score class of 19248 predicted boxes

    class_p_max = np.max(class_p, axis=0)  # [19248]

    # filter predicted boxes according the class score
    keep = (class_p_max > cfg.nms_score_thre)
    class_thre = class_p[:, keep]

    box_thre, anchor_thre, coef_thre = box_p[keep, :], anchors[keep, :], coef_p[keep, :]

    # decode boxes
    box_thre = np.concatenate((anchor_thre[:, :2] + box_thre[:, :2] * 0.1 * anchor_thre[:, 2:],
                               anchor_thre[:, 2:] * np.exp(box_thre[:, 2:] * 0.2)), axis=1)
    box_thre[:, :2] -= box_thre[:, 2:] / 2
    box_thre[:, 2:] += box_thre[:, :2]

    if class_thre.shape[1] == 0:
        return None, None, None, None, None
    else:
        assert not cfg.traditional_nms, 'Traditional nms is not supported with numpy.'
        box_thre, coef_thre, class_ids, class_thre = fast_nms_numpy(box_thre, coef_thre, class_thre, cfg)
        return class_ids, class_thre, box_thre, coef_thre, proto_p


def after_nms(ids_p, class_p, box_p, coef_p, proto_p, img_h, img_w, cfg=None, img_name=None, og_cfg=None):
    
    # ids_p: [100]
    # class_p: [100]
    # box_p: [100, 4]
    # coef_p: [100, 32]
    # proto_p: [136, 136, 32]
    
    if ids_p is None:
        return None, None, None, None

    if cfg and cfg.visual_thre > 0: # pass
        keep = class_p >= cfg.visual_thre
        if not keep.any():
            return None, None, None, None

        ids_p = ids_p[keep]
        class_p = class_p[keep]
        box_p = box_p[keep]
        coef_p = coef_p[keep]

    if cfg and cfg.save_lincomb: # pass
        draw_lincomb(proto_p, coef_p, img_name)
        
    # [136, 136, 32] @ [32, 100] -> [136, 136, 100]
   
    masks = torch.sigmoid(torch.matmul(proto_p, coef_p.t()))
    
    if not cfg or not cfg.no_crop: # Crop masks by box_p
        masks = crop(masks, box_p)
    
    # [136, 136, 100] -> [100, 136, 136]
    
    masks = masks.permute(2, 0, 1).contiguous()

    ori_size = max(img_h, img_w)
    # in OpenCV, cv2.resize is `align_corners=False`.
    masks = F.interpolate(masks.unsqueeze(0), (ori_size, ori_size), mode='bilinear', align_corners=False).squeeze(0)
    masks.gt_(0.5)  # Binarize the masks because of interpolation.
    masks = masks[:, 0: img_h, :] if img_h < img_w else masks[:, :, 0: img_w]

    box_p *= ori_size
    box_p = box_p.int()

    return ids_p, class_p, box_p, masks


def after_nms_numpy(ids_p, class_p, box_p, coef_p, proto_p, img_h, img_w, cfg=None):
    def np_sigmoid(x):
        return 1 / (1 + np.exp(-x))

    if ids_p is None:
        return None, None, None, None

    if cfg and cfg.visual_thre > 0:
        keep = class_p >= cfg.visual_thre
        if not keep.any():
            return None, None, None, None

        ids_p = ids_p[keep]
        class_p = class_p[keep]
        box_p = box_p[keep]
        coef_p = coef_p[keep]

    assert not cfg.save_lincomb, 'save_lincomb is not supported in onnx mode.'

    masks = np_sigmoid(np.matmul(proto_p, coef_p.T))

    if not cfg or not cfg.no_crop:  # Crop masks by box_p
        masks = crop_numpy(masks, box_p)

    ori_size = max(img_h, img_w)
    masks = cv2.resize(masks, (ori_size, ori_size), interpolation=cv2.INTER_LINEAR)

    if masks.ndim == 2:
        masks = masks[:, :, None]

    masks = np.transpose(masks, (2, 0, 1))
    masks = masks > 0.5  # Binarize the masks because of interpolation.
    masks = masks[:, 0: img_h, :] if img_h < img_w else masks[:, :, 0: img_w]

    box_p *= ori_size
    box_p = box_p.astype('int32')

    return ids_p, class_p, box_p, masks


def draw_lincomb(proto_data, masks, img_name):
    for kdx in range(1):
        jdx = kdx + 0
        coeffs = masks[jdx, :].cpu().numpy()
        idx = np.argsort(-np.abs(coeffs))

        coeffs_sort = coeffs[idx]
        arr_h, arr_w = (4, 8)
        p_h, p_w, _ = proto_data.size()
        arr_img = np.zeros([p_h * arr_h, p_w * arr_w])
        arr_run = np.zeros([p_h * arr_h, p_w * arr_w])

        for y in range(arr_h):
            for x in range(arr_w):
                i = arr_w * y + x

                if i == 0:
                    running_total = proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]
                else:
                    running_total += proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]

                running_total_nonlin = (1 / (1 + np.exp(-running_total)))

                arr_img[y * p_h:(y + 1) * p_h, x * p_w:(x + 1) * p_w] = (proto_data[:, :, idx[i]] / torch.max(
                    proto_data[:, :, idx[i]])).cpu().numpy() * coeffs_sort[i]
                arr_run[y * p_h:(y + 1) * p_h, x * p_w:(x + 1) * p_w] = (running_total_nonlin > 0.5).astype(np.float)

        arr_img = ((arr_img + 1) * 127.5).astype('uint8')
        arr_img = cv2.applyColorMap(arr_img, cv2.COLORMAP_WINTER)
        cv2.imwrite(f'results/images/lincomb_{img_name}', arr_img)


def draw_img(ids_p, class_p, box_p, mask_p, img_origin, cfg, img_name=None, fps=False, write=True, sample=False):
    if ids_p is None:
        return img_origin
            
    if cfg.measure != True:
        if isinstance(ids_p, torch.Tensor):
            ids_p = ids_p.cpu().detach().numpy()
            class_p = class_p.cpu().detach().numpy()
            box_p = box_p.cpu().detach().numpy()
            mask_p = mask_p.cpu().detach().numpy()
            

    num_detected = ids_p.shape[0]

    if not cfg.hide_mask: # do
        
        if cfg.measure == True:
            masks_semantic = mask_p * (ids_p[:, None, None] + 1)  # expand ids_p' shape for broadcasting
            masks_semantic_overlaps = masks_semantic.sum(axis=0)
            masks_semantic_cuda = (masks_semantic_overlaps % (cfg.num_classes)).cuda()
            
            color_masks_cuda = COLORS_cuda[masks_semantic_cuda.long()]
            
            img_fused = color_masks_cuda.cpu().numpy().astype('uint8')
            
            if sample == True:
                img_sample = cv2.addWeighted(img_fused, 0.5, img_origin, 0.5, gamma=0)

            if fps and sample: # don't
                fps_str = f'FPS: {fps:.2f}'
                cv2.putText(img_fused, fps_str, (0, 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
            if cfg.measure == True and sample == False:
                return img_fused, img_fused
            
            img_fused_measure = img_fused.copy()
            img_fused = img_sample.copy()

        if isinstance(box_p, torch.Tensor):
            class_p = class_p.cpu().detach().numpy()
            box_p = box_p.cpu().detach().numpy()
        
        if cfg.measure != True:
            masks_semantic = mask_p * (ids_p[:, None, None] + 1)  # expand ids_p' shape for broadcasting
            # The color of the overlap area is different because of the '%' operation.
            masks_semantic = masks_semantic.astype('int').sum(axis=0) % (cfg.num_classes - 1)
            color_masks = COLORS[masks_semantic].astype('uint8')
            img_fused = cv2.addWeighted(color_masks, 0.4, img_origin, 0.6, gamma=0)
        
        if cfg.cutout: # don't
            total_obj = (masks_semantic != 0)[:, :, None].repeat(3, 2)
            total_obj = total_obj * img_origin
            new_mask = ((masks_semantic == 0) * 255)[:, :, None].repeat(3, 2)
            img_matting = (total_obj + new_mask).astype('uint8')
            if write == True:
                cv2.imwrite(f'results/images/{img_name}_total_obj.jpg', img_matting)

            for i in range(num_detected):
                one_obj = (mask_p[i])[:, :, None].repeat(3, 2)
                one_obj = one_obj * img_origin
                new_mask = ((mask_p[i] == 0) * 255)[:, :, None].repeat(3, 2)
                x1, y1, x2, y2 = box_p[i, :]
                img_matting = (one_obj + new_mask)[y1:y2, x1:x2, :]
                if write == True:
                    cv2.imwrite(f'results/images/{img_name}_{i}.jpg', img_matting)
    scale = 0.6
    thickness = 1
    font = cv2.FONT_HERSHEY_DUPLEX

    if not cfg.hide_bbox: # don't
        for i in reversed(range(num_detected)):
            if ids_p[i] == -1:
                continue
           
            x1, y1, x2, y2 = box_p[i, :]

            color = COLORS[ids_p[i] + 1].tolist()
            cv2.rectangle(img_fused, (x1, y1), (x2, y2), color, thickness)

            class_name = cfg.class_names[ids_p[i]]
            text_str = f'{class_name}: {class_p[i]:.2f}' if not cfg.hide_score else class_name

            text_w, text_h = cv2.getTextSize(text_str, font, scale, thickness)[0]
            cv2.rectangle(img_fused, (x1, y1), (x1 + text_w, y1 + text_h + 5), color, -1)
            cv2.putText(img_fused, text_str, (x1, y1 + 15), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # if fps: # don't
    #     fps_str = f'FPS: {fps:.2f}'
    #     # text_w, text_h = cv2.getTextSize(fps_str, font, scale, thickness)[0]
    #     # Create a shadow to show the fps more clearly
    #     # img_fused = img_fused.astype(np.float32)
    #     # img_fused[0:text_h + 8, 0:text_w + 8] *= 0.6
    #     # img_fused = img_fused.astype(np.uint8)
    #     cv2.putText(img_fused, fps_str, (0, text_h + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    if cfg.measure == True and sample == True:
        return img_fused_measure, img_fused
    return img_fused, img_fused

