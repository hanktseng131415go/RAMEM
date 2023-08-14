import re
import torch
import time
import argparse
import torch.utils.data as data
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch.backends.cudnn as cudnn
import cv2
import pandas as pd
import pickle
import easyocr
import math
import matplotlib.pyplot as plt
import numpy as np

from utils.coco import COCODetection, val_collate
from modules.yolact import Yolact
from utils import timer_mini as timer
from utils.output_utils import after_nms, nms, draw_img
from utils.common_utils import ProgressBar, MakeJson, APDataObject, prep_metrics, calc_map
from config import get_config
from detect import measure_av, measure_lv, scaling

parser = argparse.ArgumentParser(description='YOLACT COCO Evaluation')
parser.add_argument('--img_size', type=int, default=544, help='The image size for validation.')
parser.add_argument('--weight', type=str, default='')
parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
parser.add_argument('--val_num', default=-1, type=int, help='The number of images for test, set to -1 for all.')
parser.add_argument('--coco_api', action='store_true', help='Whether to use cocoapi to evaluate results.')

iou_thres = [x / 100 for x in range(50, 100, 5)]
make_json = MakeJson()
    
#%%
def evaluation(net, cfg, step=None):
    
    dataset = COCODetection(cfg, mode='val')
    data_loader = data.DataLoader(dataset, 1, num_workers=0, shuffle=False, pin_memory=False, collate_fn=val_collate)
    ds = len(data_loader)
    progress_bar = ProgressBar(40, ds)
    timer.reset()

    ap_data = {'box': [[APDataObject() for _ in cfg.class_names] for _ in iou_thres],
               'mask': [[APDataObject() for _ in cfg.class_names] for _ in iou_thres]}
    
    if cfg.measure == True:
        with open(cfg.env_path+'data/df_dict', 'rb') as fp:
            # pickle.dump(data_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
            data_dict = pickle.load(fp)
        h_data = pd.read_excel(cfg.env_path+'data/manual_results.xlsx')
        h_data_name = h_data['img_name'].tolist()

    keys = ['img', 'AoRDiam(cm)', 'LADimen(cm)', 'LVIDd(cm)', 'LVPWD(cm)', 'IVSd(cm)', 'LVIDs(cm)', 'LVPWs(cm)', 'IVSs(cm)', 'mMAE', 'mMSE']
    result = dict([(key, []) for key in keys])  
    mse = dict([(key, []) for key in keys])  
    mae = dict([(key, []) for key in keys])  
   
    count = 0    
    lv_count = 0
    av_count = 0
    textinimage_reader = easyocr.Reader(['en'], gpu=cfg.cuda)

    
    for i, (img, gt, gt_masks, (img_h, img_w), (img_origin, img_name)) in enumerate(data_loader):
            
        tmp_count = 0
        tmp_lv_count = 0
        tmp_av_count = 0
        temp = time.perf_counter()
        if i == 1:
            timer.start()
        
        if cfg.cuda:
            img, gt, gt_masks = img.cuda(), gt.cuda(), gt_masks.cuda()
        
        with torch.no_grad(), timer.counter('forward'):           
            class_p, box_p, coef_p, proto_p, seg_p = net(img)

        with timer.counter('nms'):
            ids_p, class_p, box_p, coef_p, proto_p = nms(class_p, box_p, coef_p, proto_p, net.anchors, cfg, seg_p)
           
        with timer.counter('after_nms'):
            ids_p, class_p, boxes_p, masks_p = after_nms(ids_p, class_p, box_p, coef_p, proto_p, img_h, img_w, og_cfg=cfg)
            if ids_p is None:
                continue
        
        if cfg.measure == True:

            with timer.counter('read'):
                split_name = img_name.split('_')[0]
                gt_m = data_dict[split_name]
                if cfg.scaler != False:
                    scale = cfg.scaler / 385
                            
                else:
                    scale = scaling(img_origin, textinimage_reader)
                    
            with timer.counter('draw'):
                m_ids_p = ids_p.clone()
                
                m_ids_p[2:] = -1
                    
                lv_b_p = torch.zeros(ids_p.shape, dtype=torch.bool, device=ids_p.device)
                av_b_p = torch.zeros(ids_p.shape, dtype=torch.bool, device=ids_p.device)
                
                for ids_pi, ids in enumerate(m_ids_p):
                    if 0 in ids or 1 in ids:
                        lv_b_p[ids_pi] = True
                    if 2 in ids or 3 in ids:
                        av_b_p[ids_pi] = True
                            
                if 0 in m_ids_p and 1 in m_ids_p:
                            
                    lv_ids_p = m_ids_p[lv_b_p]
                    lv_masks_p = masks_p[lv_b_p]
                    
                    lv_img_numpy, lv_img_sample = draw_img(lv_ids_p, class_p, boxes_p, lv_masks_p, img_origin, cfg, write=False, sample=cfg.show_img)
                
                if 2 in m_ids_p and 3 in m_ids_p:
                    av_ids_p = m_ids_p[av_b_p]
                    av_masks_p = masks_p[av_b_p]
                    
                    av_img_numpy, av_img_sample = draw_img(av_ids_p, class_p, boxes_p, av_masks_p, img_origin, cfg, write=False, sample=cfg.show_img)
                    
                    
            with timer.counter('measure'):
                
                if 0 in m_ids_p and 1 in m_ids_p:
                    LVIDs, LVPWs, IVSs, LVIDd, LVPWd, IVSd, lv_details = measure_lv(lv_img_numpy, cfg.img_size, details=True)
                    # LVIDs, LVPWs, IVSs, LVIDd, LVPWd, IVSd, lv_details = measure_menn_lv(lv_img_numpy, cfg.img_size, details=True)
                    
                    # if img_name:
                    if img_name in h_data_name and cfg.eval_humans:
                                      
                        result['img'].append(img_name)
                        mae['img'].append(img_name)
                        mse['img'].append(img_name)
                        
                        result['AoRDiam(cm)'].append(0)
                        result['LADimen(cm)'].append(0) 
                        
                        result['LVIDd(cm)'].append(LVIDd * scale)
                        result['LVPWD(cm)'].append(LVPWd * scale)
                        result['IVSd(cm)'].append(IVSd * scale)
                        result['LVIDs(cm)'].append(LVIDs * scale)
                        result['LVPWs(cm)'].append(LVPWs * scale)
                        result['IVSs(cm)'].append(IVSs * scale)  
                                   
                        if cfg.show_index:
                            print(img_name)
                            print('-LVIDd(cm): ' + str(round(result['LVIDd(cm)'][-1],3)))
                            print('-LVPWD(cm): ' + str(round(result['LVPWD(cm)'][-1],3)))
                            print('-IVSd(cm): ' + str(round(result['IVSd(cm)'][-1],3)))
                            print('-LVIDs(cm): ' + str(round(result['LVIDs(cm)'][-1],3)))
                            print('-LVPWs(cm): ' + str(round(result['LVPWs(cm)'][-1],3)))
                            print('-IVSs(cm): ' + str(round(result['IVSs(cm)'][-1],3)))
                            
                        if cfg.show_img != True:
                            lv_img_sample = img_origin#.detach().cpu().numpy()
                        
                        for k in lv_details.keys():
                            cv2.drawMarker(lv_img_sample, lv_details[k], color=(255,255,255),markerSize=20, thickness=2, markerType=0)
                            cv2.putText(lv_img_sample, k, np.asarray(lv_details[k])+10, cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                            
                            # '''For MENN'''
                            # for j in range(len(lv_details[k])):
                            #     cv2.drawMarker(lv_img_sample, lv_details[k][j], color=(255,255,255),markerSize=20, thickness=2, markerType=0)
                            #     cv2.putText(lv_img_sample, k, np.asarray(lv_details[k][j])+10, cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                                
                           
                            postion_x = 800
                            cv2.putText(lv_img_sample, '-LVPWs(cm): ' + str(round(result['LVPWs(cm)'][-1],3)), (postion_x,125), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255), 1, cv2.LINE_AA)
                            cv2.putText(lv_img_sample, '-LVIDs(cm): ' + str(round(result['LVIDs(cm)'][-1],3)), (postion_x,150), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255), 1, cv2.LINE_AA)
                            cv2.putText(lv_img_sample, '-IVSs(cm): ' + str(round(result['IVSs(cm)'][-1],3)), (postion_x,175), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255), 1, cv2.LINE_AA)
                            cv2.putText(lv_img_sample, '-LVPWD(cm): ' + str(round(result['LVPWD(cm)'][-1],3)), (postion_x,200), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255), 1, cv2.LINE_AA)                            
                            cv2.putText(lv_img_sample, '-LVIDd(cm): ' + str(round(result['LVIDd(cm)'][-1],3)), (postion_x,225), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255), 1, cv2.LINE_AA)
                            cv2.putText(lv_img_sample, '-IVSd(cm): ' + str(round(result['IVSd(cm)'][-1],3)), (postion_x,250), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255), 1, cv2.LINE_AA)
                           
                        if cfg.show_img:
                            plt.imshow(lv_img_sample)
                            plt.show()
                        
                        if cfg.save_img:
                            cv2.imwrite(cfg.env_path+'results/images/lv_'+img_name, lv_img_sample)
        
                        
                        mse_LVIDd = abs(gt_m['LVIDd(cm)'] - result['LVIDd(cm)'][-1])**2
                        mse_LVPWD = abs(gt_m['LVPWD(cm)'] - result['LVPWD(cm)'][-1])**2
                        mse_IVSd = abs(gt_m['IVSd(cm)'] - result['IVSd(cm)'][-1])**2
                        mse_LVIDs = abs(gt_m['LVIDs(cm)'] - result['LVIDs(cm)'][-1])**2
                        mse_LVPWs = abs(gt_m['LVPWs(cm)'] - result['LVPWs(cm)'][-1])**2
                        mse_IVSs = abs(gt_m['IVSs(cm)'] - result['IVSs(cm)'][-1])**2
                        
                        mae_LVIDd = abs(gt_m['LVIDd(cm)'] - result['LVIDd(cm)'][-1])
                        mae_LVPWD = abs(gt_m['LVPWD(cm)'] - result['LVPWD(cm)'][-1])
                        mae_IVSd = abs(gt_m['IVSd(cm)'] - result['IVSd(cm)'][-1])
                        mae_LVIDs = abs(gt_m['LVIDs(cm)'] - result['LVIDs(cm)'][-1])
                        mae_LVPWs = abs(gt_m['LVPWs(cm)'] - result['LVPWs(cm)'][-1])
                        mae_IVSs = abs(gt_m['IVSs(cm)'] - result['IVSs(cm)'][-1])
                        
                        mse_LVIDd = mse_LVIDd if not math.isnan(mse_LVIDd) else 0.
                        mse_LVPWD = mse_LVPWD if not math.isnan(mse_LVPWD) else 0.
                        mse_IVSd = mse_IVSd if not math.isnan(mse_IVSd) else 0.
                        mse_LVIDs = mse_LVIDs if not math.isnan(mse_LVIDs) else 0.
                        mse_LVPWs = mse_LVPWs if not math.isnan(mse_LVPWs) else 0.
                        mse_IVSs = mse_IVSs if not math.isnan(mse_IVSs) else 0.
                        
                        mae_LVIDd = mae_LVIDd if not math.isnan(mae_LVIDd) else 0.
                        mae_LVPWD = mae_LVPWD if not math.isnan(mae_LVPWD) else 0.
                        mae_IVSd = mae_IVSd if not math.isnan(mae_IVSd) else 0.
                        mae_LVIDs = mae_LVIDs if not math.isnan(mae_LVIDs) else 0.
                        mae_LVPWs = mae_LVPWs if not math.isnan(mae_LVPWs) else 0.
                        mae_IVSs = mae_IVSs if not math.isnan(mae_IVSs) else 0.
                        
                        mse['LVIDd(cm)'].append(mse_LVIDd)
                        mse['LVPWD(cm)'].append(mse_LVPWD)
                        mse['IVSd(cm)'].append(mse_IVSd)
                        mse['LVIDs(cm)'].append(mse_LVIDs)
                        mse['LVPWs(cm)'].append(mse_LVPWs)
                        mse['IVSs(cm)'].append(mse_IVSs)
                        result['mMSE'].append((mse_LVIDd + mse_LVPWD + mse_IVSd + mse_LVIDs + mse_LVPWs + mse_IVSs)/6)
                        
                        mae['LVIDd(cm)'].append(mae_LVIDd)
                        mae['LVPWD(cm)'].append(mae_LVPWD)
                        mae['IVSd(cm)'].append(mae_IVSd)
                        mae['LVIDs(cm)'].append(mae_LVIDs)
                        mae['LVPWs(cm)'].append(mae_LVPWs)
                        mae['IVSs(cm)'].append(mae_IVSs)  
                        result['mMAE'].append((mae_LVIDd + mae_LVPWD + mae_IVSd + mae_LVIDs + mae_LVPWs + mae_IVSs)/6)
                        
                    tmp_lv_count += 1
                    lv_count += tmp_lv_count
                    tmp_count += 1
                        
                if 2 in m_ids_p and 3 in m_ids_p:
                    
                    AoR, LA, av_details = measure_av(av_img_numpy, cfg.img_size, details=True)
                    
                    # if img_name:
                    if img_name in h_data_name and cfg.eval_humans:
                    
                        result['img'].append(img_name)
                        mae['img'].append(img_name)
                        mse['img'].append(img_name)
                        
                        result['AoRDiam(cm)'].append(AoR * scale)
                        result['LADimen(cm)'].append(LA * scale)
                        
                        result['LVIDd(cm)'].append(0)
                        result['LVPWD(cm)'].append(0)
                        result['IVSd(cm)'].append(0)
                        result['LVIDs(cm)'].append(0)
                        result['LVPWs(cm)'].append(0)
                        result['IVSs(cm)'].append(0) 
                        
                        if cfg.show_index:
                            print(img_name)
                            print('-AoRDiam(cm): ' + str(round(result['AoRDiam(cm)'][-1],3)))
                            print('-LADimen(cm): ' + str(round(result['LADimen(cm)'][-1],3)))
                        
                        if cfg.show_img != True:
                            av_img_sample = img_origin#.detach().cpu().numpy()
                        
                        for k in av_details.keys():
                            cv2.drawMarker(av_img_sample, av_details[k], color=(255,255,255),markerSize=20, thickness=2, markerType=0)
                            cv2.putText(av_img_sample, k, np.asarray(av_details[k])+10, cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                            
                        postion_x = 800
                        cv2.putText(av_img_sample, '-AoRDiam(cm): ' + str(round(result['AoRDiam(cm)'][-1],3)), (postion_x,275), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(av_img_sample, '-LADimen(cm): ' + str(round(result['LADimen(cm)'][-1],3)), (postion_x,300), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255,255,255), 1, cv2.LINE_AA)
                                       
                        if cfg.show_img:    
                            plt.imshow(av_img_sample)
                            plt.show()
                            
                        if cfg.save_img:
                            cv2.imwrite(cfg.env_path+'results/images/av_'+img_name, av_img_sample)
                           
                        
                        mse_AoRDiam = abs(gt_m['AoRDiam(cm)'] - result['AoRDiam(cm)'][-1])**2
                        mse_LADimen = abs(gt_m['LADimen(cm)'] - result['LADimen(cm)'][-1])**2
                        
                        mae_AoRDiam = abs(gt_m['AoRDiam(cm)'] - result['AoRDiam(cm)'][-1])
                        mae_LADimen = abs(gt_m['LADimen(cm)'] - result['LADimen(cm)'][-1])        
                        
                        mse_AoRDiam = mse_AoRDiam if not math.isnan(mse_AoRDiam) else 0.
                        mse_LADimen = mse_LADimen if not math.isnan(mse_LADimen) else 0.
                        
                        mae_AoRDiam = mae_AoRDiam if not math.isnan(mae_AoRDiam) else 0.
                        mae_LADimen = mae_LADimen if not math.isnan(mae_LADimen) else 0.
                       
                        mse['AoRDiam(cm)'].append(mse_AoRDiam)
                        mse['LADimen(cm)'].append(mse_LADimen) 
                        result['mMSE'].append((mse_AoRDiam + mse_LADimen)/2)
                        
                        mae['AoRDiam(cm)'].append(mae_AoRDiam)
                        mae['LADimen(cm)'].append(mae_LADimen)
                        result['mMAE'].append((mae_AoRDiam + mae_LADimen)/2)
                        
                        
                    tmp_av_count += 1
                    av_count += tmp_av_count
                    tmp_count += 1
                    
                if tmp_count == 0:
                    if img_name in h_data_name:
                        result['img'].append(img_name)
                        
                        result['AoRDiam(cm)'].append(0)
                        result['LADimen(cm)'].append(0)
                        
                        result['LVIDd(cm)'].append(0)
                        result['LVPWD(cm)'].append(0)
                        result['IVSd(cm)'].append(0)
                        result['LVIDs(cm)'].append(0)
                        result['LVPWs(cm)'].append(0)
                        result['IVSs(cm)'].append(0) 
                        result['mMSE'].append(0)
                        result['mMAE'].append(0)
                        
                    count += 1    
                    
        with timer.counter('metric'):

            ids_p = list(ids_p.cpu().numpy().astype(int))
            class_p = list(class_p.cpu().numpy().astype(float))
            
            if cfg.coco_api:
                boxes_p = boxes_p.cpu().numpy()
                masks_p = masks_p.cpu().numpy()

                for j in range(masks_p.shape[0]):
                    if (boxes_p[j, 3] - boxes_p[j, 1]) * (boxes_p[j, 2] - boxes_p[j, 0]) > 0:
                        make_json.add_bbox(dataset.ids[i], ids_p[j], boxes_p[j, :], class_p[j])
                        make_json.add_mask(dataset.ids[i], ids_p[j], masks_p[j, :, :], class_p[j])
            else:
                prep_metrics(cfg, ap_data, ids_p, class_p, boxes_p, masks_p, gt, gt_masks, img_h, img_w, iou_thres)
        
        aa = time.perf_counter()
        if i > 0:
            batch_time = aa - temp
            timer.add_batch_time(batch_time)
        
            if cfg.measure == True:
                t_t, t_d, t_r, t_f, t_nms, t_an, t_me, t_da, t_m = timer.get_times(['batch', 'data', 'read', 'forward',
                                                    'nms', 'after_nms', 'metric', 'draw', 'measure'])
                time_sum = (t_d + t_f + t_nms + t_an + t_da + t_m)
            else:
                t_t, t_d, t_f, t_nms, t_an, t_me = timer.get_times(['batch', 'data', 'forward',
                                                                    'nms', 'after_nms', 'metric'])
                time_sum = (t_d + t_f + t_nms + t_an)

            fps, t_fps = 1 / time_sum, 1 / t_t

            bar_str = progress_bar.get_bar(i + 1)
            
            if cfg.measure == True:
                print(f'\rTesting: {bar_str} {i + 1}/{ds}, fps: {fps:.2f} | total fps: {t_fps:.2f} | '
                      f't_t: {t_t:.3f} | t_d: {t_d:.3f} | t_f: {t_f:.3f} | t_nms: {t_nms:.3f} | '
                      f't_after_nms: {t_an:.3f} | t_draw: {t_da:.3f} | t_measure: {t_m:.3f} | t_metric: {t_me:.3f}', end='')
                
            else:
                print(f'\rTesting: {bar_str} {i + 1}/{ds}, fps: {fps:.2f} | total fps: {t_fps:.2f} | '
                      f't_t: {t_t:.3f} | t_d: {t_d:.3f} | t_f: {t_f:.3f} | t_nms: {t_nms:.3f} | '
                      f't_after_nms: {t_an:.3f} | t_metric: {t_me:.3f}', end='')
                            
            
    if cfg.coco_api:
        make_json.dump()
        print(f'\nJson files dumped, saved in: \'results/\', start evaluating.')

        gt_annotations = COCO(cfg.val_ann)
        bbox_dets = gt_annotations.loadRes(f'results/bbox_detections.json')
        mask_dets = gt_annotations.loadRes(f'results/mask_detections.json')

        print('\nEvaluating BBoxes:')
        bbox_eval = COCOeval(gt_annotations, bbox_dets, 'bbox')
        bbox_eval.evaluate()
        bbox_eval.accumulate()
        bbox_eval.summarize()

        print('\nEvaluating Masks:')
        bbox_eval = COCOeval(gt_annotations, mask_dets, 'segm')
        bbox_eval.evaluate()
        bbox_eval.accumulate()
        bbox_eval.summarize()
        
    else:
        table, box_row, mask_row = calc_map(ap_data, iou_thres, len(cfg.class_names), step=step)
        print(table)
        
        if cfg.measure == True:
            MAEs = {}
            MSEs = {}
            if len(mae['img']) > 0 and len(mse['img']) > 0:
                for indicator in keys[1:-2]:
                
                    MAEs[indicator] = round(sum(mae[indicator]) / len(mae[indicator]), 3)
                    MSEs[indicator] = round(sum(mse[indicator]) / len(mse[indicator]), 3)
                
                print('MAE: ', MAEs)
                print('MSE: ', MSEs)
                
                mMAE = round(np.asarray([*MAEs.values()]).mean(), 3)
                sMAE = round(np.asarray([*MAEs.values()]).std(), 3)
                print('mMAE: {0}, sMAE: {1}'.format(mMAE, sMAE))
                
                mMSE = round(np.asarray([*MSEs.values()]).mean(), 3)
                sMSE = round(np.asarray([*MSEs.values()]).std(), 3)
                print('mMSE: {0}, sMSE: {1}'.format(mMSE, sMSE))
                if mMAE < 0.425 and mMSE < 0.378:
                    print('stop')
            # result_df = pd.DataFrame.from_dict(result)
            # result_df.to_csv(cfg.env_path+'/results/result_df.csv', index=False)

        return table, box_row, mask_row

#%%
if __name__ == '__main__':
    args = parser.parse_args()
    prefix = re.findall(r'best_\d+\.\d+_', args.weight)[0]
    suffix = re.findall(r'_\d+\.pth', args.weight)[0]
    args.cfg = args.weight.split(prefix)[-1].split(suffix)[0]
    cfg = get_config(args, mode='val')

    net = Yolact(cfg)
    net.load_weights(cfg.weight, cfg.cuda)
    net.eval()

    if cfg.cuda:
        cudnn.benchmark = True
        cudnn.fastest = True
        net = net.cuda()

    evaluation(net, cfg)

