import re
import torch
import time
import argparse
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import cv2
import scipy
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import os

from utils.coco import COCODetection, detect_collate
from modules.yolact import Yolact
from utils import timer_mini as timer
from utils.output_utils import after_nms, nms, draw_img
from utils.common_utils import ProgressBar, MakeJson
from utils.augmentations import val_aug
from config import get_config

def scaling(img, reader, pix_hight=385):
    #OCR to get scaler
  
    cropped = img[10:250, 10:150]#.detach().cpu().numpy()

    results = reader.readtext(cropped, detail=0)
    for result in results:
        if 'cm' in result:
            text = result.replace('i', '1').replace('T', '1').replace('cm', '').replace('O','0').replace('S', '5').replace('I','1')
    scaler = float(text) / pix_hight
    
    return scaler

def av_ocr(img, reader):
    
    cropped = img[100:200, 600:850]
    results = reader.readtext(cropped, detail=0)
    result_dict = {}
    try:
        for result in results:
            if 'AoR Diam' in result:
                text = results[results.index(result)+1].replace('cm', '').replace('O','0').replace('S', '5').replace('I','1')
                result_dict['AoRDiam(cm)'] = float(text)
            if 'LA Dimen' in result:
                text = results[results.index(result)+1].replace('cm', '').replace('O','0').replace('S', '5').replace('I','1')
                result_dict['LADimen(cm)'] = float(text)
    except:
        pass
    
    return result_dict

def lv_ocr(img, reader):
    
    cropped = img[100:300, 600:900]
    results = reader.readtext(cropped, detail=0)
    result_dict = {}
    try:
        for result in results:
            if 'LVPWs' in result:
                text = results[results.index(result)+1].replace('cm', '').replace('O','0').replace('S', '5').replace('I','1')
                result_dict['LVPWs(cm)'] = float(text)
            if 'LVIDs' in result:
                text = results[results.index(result)+1].replace('cm', '').replace('O','0').replace('S', '5').replace('I','1')
                result_dict['LVIDs(cm)'] = float(text)
            if 'IVSs' in result:
                text = results[results.index(result)+1].replace('cm', '').replace('O','0').replace('S', '5').replace('I','1')
                result_dict['IVSs(cm)'] = float(text)
            if 'LVPWd' in result:
                text = results[results.index(result)+1].replace('cm', '').replace('O','0').replace('S', '5').replace('I','1')
                result_dict['LVPWD(cm)'] = float(text)
            if 'LVIDd' in result:
                text = results[results.index(result)+1].replace('cm', '').replace('O','0').replace('S', '5').replace('I','1')
                result_dict['LVIDd(cm)'] = float(text)
            if 'IVSd' in result:
                text = results[results.index(result)+1].replace('cm', '').replace('O','0').replace('S', '5').replace('I','1')
                result_dict['IVSd(cm)'] = float(text)
    except:
        pass
    
    return result_dict
                
def measure_av(mask_img_numpy, img_size, details=False):
   
    AoR, LA = 0, 0
    AoRt, LAt = np.asarray([0,0]), np.asarray([0,0])
    try:
        img = mask_img_numpy
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thre = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        # equ (8)
        contours, hierarchy = cv2.findContours(thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    

        for i, c in enumerate(contours):
            x2, y2, w2, h2 = cv2.boundingRect(c)  
            if w2 > mask_img_numpy.shape[1]*0.33:
                break
            
        cnt = contours[i]
        
        # equ (9)
        LAt = cnt[0][0]
        # equ (10)-(13)
        LA = scan_down(thre, LAt, 50, 2, img_size/544)
        gap = scan_up(thre, LAt, 50, 1, img_size/544)
        AoRb = (LAt[0], LAt[1]-gap)
        
        # to make sure AoRb locate at mask, if not -> redo
        k=1
        while AoRb[1] < mask_img_numpy.shape[0]*0.5:
            img = mask_img_numpy
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thre = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
            thre = cv2.erode(thre, np.ones((k, k), np.uint8))
            contours, hierarchy = cv2.findContours(thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    
            k = k + 2
            for i, c in enumerate(contours):
                x2, y2, w2, h2 = cv2.boundingRect(c)  
                if w2 > mask_img_numpy.shape[1]*0.33:
                    break
            
            cnt = contours[i]
        
            LAt = cnt[0][0]
            LA = scan_down(gray, LAt, 50, 2, img_size/544)
            gap = scan_up(thre, LAt, 50, 1, img_size/544)
            AoRb = (LAt[0], LAt[1]-gap)
        
        AoR = scan_up(gray, AoRb, 50, 2, img_size/544)
        AoRt = (AoRb[0], AoRb[1]-AoR)
                
    except:
        pass
    
    if details == True:
        return AoR, LA, {'AoRt':np.asarray(AoRt), 'LAt':np.asarray(LAt), 'LAb':np.asarray([LAt[0], LAt[1]+LA])}
        
    return AoR, LA

def measure_lv(mask_img_numpy, img_size, details=False):
    
    LVIDs, LVPWs, IVSs, LVIDd, LVPWd, IVSd = 0, 0, 0, 0, 0, 0
    
    IVSs_t, IVSd_t = np.asarray([0,0]), np.asarray([0,0])
    IVSs_b, IVSd_b = np.asarray([0,0]), np.asarray([0,0])
    LVPWs_t, LVPWd_t = np.asarray([0,0]), np.asarray([0,0])
    LVPWs_b, LVPWd_b = np.asarray([0,0]), np.asarray([0,0])

    try:       
        img = mask_img_numpy
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thre = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        
        # equ (14)
        contours, hierarchy = cv2.findContours(thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, c in enumerate(contours):
            x2, y2, w2, h2 = cv2.boundingRect(c)  
            if w2 > mask_img_numpy.shape[1]*0.33:
                break
        
        cnt = contours[i]     
        # equ (15)
        LVIDs_b = cnt[0][0]
        # equ (16)-(21)
        LVIDs = scan_up(thre, cnt[0][0], 50, 1, img_size/544)
                
        LVIDs_t = (cnt[0][0][0], cnt[0][0][1] - LVIDs)
        IVSs = scan_up(thre, LVIDs_t, 50, 2, img_size/544)
        LVPWs = scan_down(thre, cnt[0][0], 50, 2, img_size/544)
        
        # equ (22)
        hull2 = cv2.convexHull(cnt, returnPoints = False)
        defects = cv2.convexityDefects(cnt, hull2)
        
        LVIDs_b = cnt[0][0]
        IVSs_t = np.asarray([LVIDs_t[0], LVIDs_t[1]-IVSs])
        LVPWs_b = np.asarray([LVIDs_b[0], LVIDs_b[1]+LVPWs])
        
        # equ (23)-(24)
        s = cnt[defects[:, 0, 0]][:, 0, 0]
        e = cnt[defects[:, 0, 1]][:, 0, 0]
        sign = ((s - e) > 0)[: ,None]
        farthest_idx = (defects[:, :, -1]*sign).argmax()
        s,e,f,d = defects[farthest_idx,0]
        LVPWd_t = tuple(cnt[f][0])
        # equ (25)-(30)
        LVIDd = scan_up(thre, LVPWd_t, 50, 1, img_size/544)
        
        LVIDd_t = (LVPWd_t[0], LVPWd_t[1] - LVIDd)
        IVSd = scan_up(thre, LVIDd_t, 50, 2, img_size/544)    
        
        LVPWd = scan_down(thre, LVPWd_t, 50, 2, img_size/544)
        
        LVIDd_t = [LVPWd_t[0], LVPWd_t[1]-LVIDd]
        
        # to make sure LVIDd_t and IVSs_t locate accurately, if not -> redo
        if LVIDd_t[1] < IVSs_t[1]:
            k=1
            while LVIDd_t[1] < IVSs_t[1]:
                img = mask_img_numpy
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, thre = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
                thre = cv2.erode(thre, np.ones((k, k), np.uint8))
                contours, hierarchy = cv2.findContours(thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    
                k = k + 2
                
                for i, c in enumerate(contours):
                    x2, y2, w2, h2 = cv2.boundingRect(c)  
                    if w2 > mask_img_numpy.shape[1]*0.33:
                        break      
                    
                cnt = contours[i]     
                LVIDs_b = cnt[0][0]
                LVIDs = scan_up(thre, cnt[0][0], 50, 1, img_size/544)
                
                LVIDs_t = (cnt[0][0][0], cnt[0][0][1] - LVIDs)
                IVSs = scan_up(thre, LVIDs_t, 50, 2, img_size/544)
                LVPWs = scan_down(thre, cnt[0][0], 50, 2, img_size/544)
                
                hull2 = cv2.convexHull(cnt, returnPoints = False)
                defects = cv2.convexityDefects(cnt, hull2)
                
                LVIDs_b = cnt[0][0]
                IVSs_t = np.asarray([LVIDs_t[0], LVIDs_t[1]-IVSs])
                LVPWs_b = np.asarray([LVIDs_b[0], LVIDs_b[1]+LVPWs])
                
                s = cnt[defects[:, 0, 0]][:, 0, 0]
                e = cnt[defects[:, 0, 1]][:, 0, 0]
                sign = ((s - e) > 0)[: ,None]
                farthest_idx = (defects[:, :, -1]*sign).argmax()
                s,e,f,d = defects[farthest_idx,0]
                LVPWd_t = tuple(cnt[f][0])
                LVIDd = scan_up(thre, LVPWd_t, 50, 1, img_size/544)
                
                LVIDd_t = (LVPWd_t[0], LVPWd_t[1] - LVIDd)
         
        IVSd = scan_up(thre, LVIDd_t, 50, 2, img_size/544)    
        LVPWd = scan_down(thre, LVPWd_t, 50, 2, img_size/544)
         
        IVSd_t = [LVIDd_t[0], LVIDd_t[1]-IVSd]
        LVPWd_b = [LVPWd_t[0], LVPWd_t[1]+LVPWd]
        
        IVSs_t, IVSd_t = np.asarray([IVSs_t[0], IVSs_t[1]]), np.asarray([IVSd_t[0], IVSd_t[1]])
        IVSs_b, IVSd_b = np.asarray([LVIDs_t[0], LVIDs_t[1]]), np.asarray([LVIDd_t[0], LVIDd_t[1]])
        LVPWs_t, LVPWd_t = np.asarray([LVIDs_b[0], LVIDs_b[1]]), np.asarray([LVPWd_t[0], LVPWd_t[1]])
        LVPWs_b, LVPWd_b = np.asarray([LVPWs_b[0], LVPWs_b[1]]), np.asarray([LVPWd_b[0], LVPWd_b[1]])
        
            
    except:
        pass
    
    if details == True:
        details_dict = {
            'IVSs_t':IVSs_t,
            'IVSd_t':IVSd_t,
            'IVSs_b':IVSs_b,
            'IVSd_b':IVSd_b,
            'LVPWs_t':LVPWs_t,
            'LVPWd_t':LVPWd_t,
            'LVPWs_b':LVPWs_b,
            'LVPWd_b':LVPWd_b           
            }
        return LVIDs, LVPWs, IVSs, LVIDd, LVPWd, IVSd, details_dict
    
    return LVIDs, LVPWs, IVSs, LVIDd, LVPWd, IVSd

def measure_menn_lv(mask_img_numpy, img_size, details=False):

    LVIDs, LVPWs, IVSs, LVIDd, LVPWd, IVSd = 0, 0, 0, 0, 0, 0
    
    try:
        img = mask_img_numpy
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
        ret, thre = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        
        for i, c in enumerate(contours):
            x2, y2, w2, h2 = cv2.boundingRect(c)  
            if w2 > 200:
                break
        
        cnt2 = contours[i]
        x2, y2, w2, h2 = cv2.boundingRect(cnt2)     
        signal_low = []
        signal_high = []
        
        for i in range (x2,x2+w2):
            tmp = y2
            while(gray[tmp][i]==0):
                tmp+=1
            signal_low.append(tmp)
            signal_high.append(-tmp)
            
        peaks_low, _ = scipy.signal.find_peaks(np.asarray(signal_low), distance = 15) # 15 = MENN defualt setting
        peaks_high, _ = scipy.signal.find_peaks(np.asarray(signal_high), distance = 15)
        LVIDd_tmp = []
        LVPWd_tmp = []
        IVSd_tmp = []
        LVIDs_tmp = []
        LVPWs_tmp = []
        IVSs_tmp = []
        
        LVPWd_t_tmp, LVPWd_b_tmp = [], []
        IVSd_t_tmp, IVSd_b_tmp = [], []
        
        for i in peaks_low:
            LVPWd_ti = np.asarray([x2 + i, signal_low[i]])
            LVPWd_t_tmp.append(LVPWd_ti)
            lvidd = scan_up(gray, LVPWd_ti, 50, 1, img_size/544)
            LVIDd_tmp.append(lvidd)
            
            IVSd_bi = np.asarray([LVPWd_ti[0], LVPWd_ti[1]-lvidd])
            IVSd_b_tmp.append(IVSd_bi)
            ivsd = scan_up(gray, IVSd_bi, 50, 2, img_size/544)
            IVSd_tmp.append(ivsd)  
            IVSd_ti = np.asarray([IVSd_bi[0], IVSd_bi[1]-ivsd])
            IVSd_t_tmp.append(IVSd_ti)
            
            lvpwd = scan_down(gray, LVPWd_ti, 0, 2, img_size/544)
            LVPWd_tmp.append(lvpwd)
            LVPWd_ti = np.asarray([LVPWd_ti[0], LVPWd_ti[1]+lvpwd])
            LVPWd_b_tmp.append(LVPWd_ti)

        LVPWs_t_tmp, LVPWs_b_tmp = [], []
        IVSs_t_tmp, IVSs_b_tmp = [], []
            
        for i in peaks_high:
            LVPWs_ti = np.asarray([x2 + i, -signal_high[i]])
            LVPWs_t_tmp.append(LVPWs_ti)
            lvids = scan_up(gray, LVPWs_ti, 50, 1, img_size/544)
            LVIDs_tmp.append(lvids)
            
            IVSs_bi = np.asarray([LVPWs_ti[0], LVPWs_ti[1]-lvids])
            IVSs_b_tmp.append(IVSs_bi)
            ivss = scan_up(gray, IVSs_bi, 50, 2, img_size/544)
            IVSs_tmp.append(ivss)
            IVSs_ti = np.asarray([IVSs_bi[0], IVSs_bi[1]-ivss])
            IVSs_t_tmp.append(IVSs_ti)
            
            lvpws = scan_down(gray, LVPWs_ti, 0, 2, img_size/544)
            LVPWs_tmp.append(lvpws)
            LVPWs_ti = np.asarray([LVPWs_ti[0], LVPWs_ti[1]+lvpws])
            LVPWs_b_tmp.append(LVPWs_ti)
            
        LVIDd = sum(LVIDd_tmp) / len(LVIDd_tmp)
        LVPWd = sum(LVPWd_tmp) / len(LVPWd_tmp)
        IVSd = sum(IVSd_tmp) / len(IVSd_tmp)
        LVIDs = sum(LVIDs_tmp) / len(LVIDs_tmp)
        LVPWs = sum(LVPWs_tmp) / len(LVPWs_tmp)
        IVSs = sum(IVSs_tmp) / len(IVSs_tmp)
    except:
        pass
    
    if details == True:
        details_dict = {
            'IVSs_t':IVSs_t_tmp,
            'IVSd_t':IVSd_t_tmp,
            'IVSs_b':IVSs_b_tmp,
            'IVSd_b':IVSd_b_tmp,
            'LVPWs_t':LVPWs_t_tmp,
            'LVPWd_t':LVPWd_t_tmp,
            'LVPWs_b':LVPWs_b_tmp,
            'LVPWd_b':LVPWd_b_tmp           
            }
        return LVIDs, LVPWs, IVSs, LVIDd, LVPWd, IVSd, details_dict
    
    return LVIDs, LVPWs, IVSs, LVIDd, LVPWd, IVSd

def scan_up(gray, start_point, thre, mode, scale):
    # scale = 1
    if mode == 1:#bg -> mask
        move = 1
        while gray[start_point[1] - move][start_point[0]] < thre:
            move += 2
            if start_point[1] - move <= (200*scale):
                break
    
        return move
    else : #mask -> bg
        move = 1
        while gray[start_point[1] - move][start_point[0]] > thre:
            move += 2
            if start_point[1] - move <= (200*scale):
                break
        return move

def scan_down(gray, start_point, thre, mode, scale):
    # scale = 1
    if mode == 1:#bg -> mask
        move = 1
        while gray[start_point[1] + move][start_point[0]] < thre:
            move += 2
            if start_point[1] + move <= (200*scale):
                break
    
        return move
    else : #mask -> bg
        move = 1
        while gray[start_point[1] + move][start_point[0]] > thre:
            move += 2
            if start_point[1] + move <= (200*scale):
                break
        return move 

parser = argparse.ArgumentParser(description='YOLACT COCO Evaluation')
parser.add_argument('--img_size', type=int, default=544, help='The image size for validation.')
parser.add_argument('--weight', type=str, default='')
parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
parser.add_argument('--val_num', default=-1, type=int, help='The number of images for test, set to -1 for all.')
parser.add_argument('--coco_api', action='store_true', help='Whether to use cocoapi to evaluate results.')

iou_thres = [x / 100 for x in range(50, 100, 5)]
make_json = MakeJson()

    
#%%
def image_detection(net, cfg, step=None):
    
    dataset = COCODetection(cfg, mode='detect')
    data_loader = data.DataLoader(dataset, 1, num_workers=0, shuffle=False, pin_memory=False, collate_fn=detect_collate)
    ds = len(data_loader)
    progress_bar = ProgressBar(40, ds)
    timer.reset()

    keys = ['img', 'AoRDiam(cm)', 'LADimen(cm)', 'LVIDd(cm)', 'LVPWD(cm)', 'IVSd(cm)', 'LVIDs(cm)', 'LVPWs(cm)', 'IVSs(cm)', 'mMAE', 'mMSE']
    result = dict([(key, []) for key in keys])  
   
    count = 0    
    lv_count = 0
    av_count = 0
    textinimage_reader = easyocr.Reader(['en'], gpu=cfg.cuda)

    
    for i, (img, img_origin, img_name) in enumerate(data_loader):
        
        tmp_count = 0
        tmp_lv_count = 0
        tmp_av_count = 0
        temp = time.perf_counter()
        if i == 1:
            timer.start()
        
        if cfg.cuda:
            img = img.cuda()
            
        img_h, img_w = img_origin.shape[0:2]
        
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
                    
                    if img_name:
                
                        result['img'].append(img_name)
                        
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
                                
                            
                        position_x = int(img_h * 0.78125)
                        position_y = int(img_w * 0.16276)
                        
                        cv2.putText(lv_img_sample, '-LVPWs(cm): ' + str(round(result['LVPWs(cm)'][-1],3)), (position_x,position_y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(lv_img_sample, '-LVIDs(cm): ' + str(round(result['LVIDs(cm)'][-1],3)), (position_x,position_y+20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(lv_img_sample, '-IVSs(cm): ' + str(round(result['IVSs(cm)'][-1],3)), (position_x,position_y+40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(lv_img_sample, '-LVPWD(cm): ' + str(round(result['LVPWD(cm)'][-1],3)), (position_x,position_y+60), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)                            
                        cv2.putText(lv_img_sample, '-LVIDd(cm): ' + str(round(result['LVIDd(cm)'][-1],3)), (position_x,position_y+80), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(lv_img_sample, '-IVSd(cm): ' + str(round(result['IVSd(cm)'][-1],3)), (position_x,position_y+100), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                                  
                        if cfg.show_img:
                            plt.imshow(lv_img_sample)
                            plt.show()
                        
                        if cfg.save_img:
                            cv2.imwrite(cfg.env_path+'results/test_images/lv_'+img_name, lv_img_sample)
                           
                            
                    tmp_lv_count += 1
                    lv_count += tmp_lv_count
                    tmp_count += 1
                        
                if 2 in m_ids_p and 3 in m_ids_p:
                    
                    AoR, LA, av_details = measure_av(av_img_numpy, cfg.img_size, details=True)
                    
                    if img_name:
                    
                        result['img'].append(img_name)
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
                            
                        position_x = int(img_h * 0.78125)
                        position_y = int(img_w * 0.26855)
                       
                        cv2.putText(av_img_sample, '-AoRDiam(cm): ' + str(round(result['AoRDiam(cm)'][-1],3)), (position_x,position_y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(av_img_sample, '-LADimen(cm): ' + str(round(result['LADimen(cm)'][-1],3)), (position_x,position_y+20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                                     
                        if cfg.show_img:    
                            plt.imshow(av_img_sample)
                            plt.show()
                            
                        if cfg.save_img:
                            cv2.imwrite(cfg.env_path+'results/test_images/av_'+img_name, av_img_sample)
                           
                    tmp_av_count += 1
                    av_count += tmp_av_count
                    tmp_count += 1
                    
                if tmp_count == 0:
                    
                    if img_name:
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

        aa = time.perf_counter()
        if i > 0:
            batch_time = aa - temp
            timer.add_batch_time(batch_time)
        
            if cfg.measure == True:
                t_t, t_d, t_r, t_f, t_nms, t_an, t_da, t_m = timer.get_times(['batch', 'data', 'read', 'forward',
                                                    'nms', 'after_nms', 'draw', 'measure'])
                time_sum = (t_d + t_f + t_nms + t_an + t_da + t_m)
            else:
                t_t, t_d, t_f, t_nms, t_an = timer.get_times(['batch', 'data', 'forward',
                                                                    'nms', 'after_nms'])
                time_sum = (t_d + t_f + t_nms + t_an)

            fps, t_fps = 1 / time_sum, 1 / t_t

            bar_str = progress_bar.get_bar(i + 1)
            
            if cfg.measure == True:
                print(f'\rTesting: {bar_str} {i + 1}/{ds}, fps: {fps:.2f} | total fps: {t_fps:.2f} | '
                      f't_t: {t_t:.3f} | t_d: {t_d:.3f} | t_f: {t_f:.3f} | t_nms: {t_nms:.3f} | '
                      f't_after_nms: {t_an:.3f} | t_draw: {t_da:.3f} | t_measure: {t_m:.3f}', end='')
                
            else:
                print(f'\rTesting: {bar_str} {i + 1}/{ds}, fps: {fps:.2f} | total fps: {t_fps:.2f} | '
                      f't_t: {t_t:.3f} | t_d: {t_d:.3f} | t_f: {t_f:.3f} | t_nms: {t_nms:.3f} | '
                      f't_after_nms: {t_an:.3f}', end='')
                            

    print('\nFinished, saved in: results/images.')
    # result = pd.DataFrame.from_dict(result)
    # result.to_csv('./results/result.csv')
    
def video_detection(net, cfg, step=None):
    
    vid = cv2.VideoCapture(cfg.video)
    
    target_fps = round(vid.get(cv2.CAP_PROP_FPS))
    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    num_frames = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (int(frame_width), int(frame_height))
    
    progress_bar = ProgressBar(40, num_frames)
    timer.reset()

    keys = ['img', 'AoRDiam(cm)', 'LADimen(cm)', 'LVIDd(cm)', 'LVPWD(cm)', 'IVSd(cm)', 'LVIDs(cm)', 'LVPWs(cm)', 'IVSs(cm)', 'mMAE', 'mMSE']
    result = dict([(key, []) for key in keys])  
   
    fps = 0
    count = 0    
    lv_count = 0
    av_count = 0
    textinimage_reader = easyocr.Reader(['en'], gpu=cfg.cuda)

    for i in range(num_frames):
        
        ret, img_origin = vid.read()
        img_h, img_w = img_origin.shape[0:2]
        img = val_aug(img_origin, cfg.img_size)
        img = torch.tensor(img).float()

        img_name = cfg.video.split('/')[-1].split('.')[0]
        
        tmp_count = 0
        tmp_lv_count = 0
        tmp_av_count = 0
        temp = time.perf_counter()
        if i == 1:
            timer.start()
        
        if cfg.cuda:
            img = img.cuda()
        
        img_h, img_w = img_origin.shape[0:2]
        
        with torch.no_grad(), timer.counter('forward'):           
            class_p, box_p, coef_p, proto_p, seg_p = net(img.unsqueeze(0))

        with timer.counter('nms'):
            ids_p, class_p, box_p, coef_p, proto_p = nms(class_p, box_p, coef_p, proto_p, net.anchors, cfg, seg_p)
           
        with timer.counter('after_nms'):
            ids_p, class_p, boxes_p, masks_p = after_nms(ids_p, class_p, box_p, coef_p, proto_p, img_h, img_w, og_cfg=cfg)
            if ids_p is None:
                continue
        
        if cfg.measure == True:

            with timer.counter('read'):
                
                if cfg.scaler != False:
                    scale = cfg.scaler/385
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
                    
                    lv_img_numpy, lv_img_sample = draw_img(lv_ids_p, class_p, boxes_p, lv_masks_p, img_origin, cfg, write=False, sample=cfg.show_img, fps=fps)
                
                if 2 in m_ids_p and 3 in m_ids_p:
                    av_ids_p = m_ids_p[av_b_p]
                    av_masks_p = masks_p[av_b_p]
                    
                    av_img_numpy, av_img_sample = draw_img(av_ids_p, class_p, boxes_p, av_masks_p, img_origin, cfg, write=False, sample=cfg.show_img, fps=fps)
                    
            with timer.counter('measure'):
                
                if 0 in m_ids_p and 1 in m_ids_p:
                    
                    LVIDs, LVPWs, IVSs, LVIDd, LVPWd, IVSd, lv_details = measure_lv(lv_img_numpy, cfg.img_size, details=True)
                    # LVIDs, LVPWs, IVSs, LVIDd, LVPWd, IVSd, lv_details = measure_menn_lv(lv_img_numpy, cfg.img_size, details=True)
                    
                    if img_name:
                
                        result['img'].append(img_name)
                        
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
                            
                        if cfg.show_img == False:
                            lv_img_sample = img_origin#.detach().cpu().numpy()
                        
                        for k in lv_details.keys():
                            cv2.drawMarker(lv_img_sample, lv_details[k], color=(255,255,255),markerSize=20, thickness=2, markerType=0)
                            cv2.putText(lv_img_sample, k, np.asarray(lv_details[k])+10, cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                            
                            # '''For MENN'''
                            # for j in range(len(lv_details[k])):
                            #     cv2.drawMarker(lv_img_sample, lv_details[k][j], color=(255,255,255),markerSize=20, thickness=2, markerType=0)
                            #     cv2.putText(lv_img_sample, k, np.asarray(lv_details[k][j])+10, cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                            
                        position_x = int(img_h * 0.78125)
                        position_y = int(img_w * 0.16276)
                        
                        cv2.putText(lv_img_sample, '-LVPWs(cm): ' + str(round(result['LVPWs(cm)'][-1],3)), (position_x,position_y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(lv_img_sample, '-LVIDs(cm): ' + str(round(result['LVIDs(cm)'][-1],3)), (position_x,position_y+20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(lv_img_sample, '-IVSs(cm): ' + str(round(result['IVSs(cm)'][-1],3)), (position_x,position_y+40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(lv_img_sample, '-LVPWD(cm): ' + str(round(result['LVPWD(cm)'][-1],3)), (position_x,position_y+60), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)                            
                        cv2.putText(lv_img_sample, '-LVIDd(cm): ' + str(round(result['LVIDd(cm)'][-1],3)), (position_x,position_y+80), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(lv_img_sample, '-IVSd(cm): ' + str(round(result['IVSd(cm)'][-1],3)), (position_x,position_y+100), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                        fps_str = f'FPS: {fps:.2f}'
                        cv2.putText(lv_img_sample, fps_str, (0, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                                       
                        if cfg.show_img and i%24 == 0:
                            plt.imshow(lv_img_sample)
                            plt.show()
                            
                        if cfg.save_img:
                            cv2.imwrite(cfg.env_path+'results/videos_images/lv_'+img_name+'_'+str(i)+'.jpg', lv_img_sample)
                        
                    tmp_lv_count += 1
                    lv_count += tmp_lv_count
                    tmp_count += 1
                        
                if 2 in m_ids_p and 3 in m_ids_p:
                    
                    AoR, LA, av_details = measure_av(av_img_numpy, cfg.img_size, details=True)
                    
                    if img_name:
                    
                        result['img'].append(img_name)
                        
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
                        
                        if cfg.show_img == False:
                            av_img_sample = img_origin#.detach().cpu().numpy()
                        
                        for k in av_details.keys():
                            cv2.drawMarker(av_img_sample, av_details[k], color=(255,255,255),markerSize=20, thickness=2, markerType=0)
                            cv2.putText(av_img_sample, k, np.asarray(av_details[k])+10, cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                       
                        position_x = int(img_h * 0.78125)
                        position_y = int(img_w * 0.26855)
                       
                        cv2.putText(av_img_sample, '-AoRDiam(cm): ' + str(round(result['AoRDiam(cm)'][-1],3)), (position_x,position_y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                        cv2.putText(av_img_sample, '-LADimen(cm): ' + str(round(result['LADimen(cm)'][-1],3)), (position_x,position_y+20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                        
                        fps_str = f'FPS: {fps:.2f}'
                        cv2.putText(av_img_sample, fps_str, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                         
                        if cfg.show_img and i%24 == 0:
                            plt.imshow(av_img_sample)
                            plt.show()
                            
                        if cfg.save_img:
                            cv2.imwrite(cfg.env_path+'results/videos_images/av_'+img_name+'_'+str(i)+'.jpg', av_img_sample)
                         
                    tmp_av_count += 1
                    av_count += tmp_av_count
                    tmp_count += 1
                    
                if tmp_count == 0:
                    if img_name:
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

        aa = time.perf_counter()
        if i > 0:
            batch_time = aa - temp
            timer.add_batch_time(batch_time)
        
            if cfg.measure == True:
                t_t, t_d, t_r, t_f, t_nms, t_an, t_da, t_m = timer.get_times(['batch', 'data', 'read', 'forward',
                                                    'nms', 'after_nms', 'draw', 'measure'])
                time_sum = (t_d + t_f + t_nms + t_an + t_da + t_m)
            else:
                t_t, t_d, t_f, t_nms, t_an = timer.get_times(['batch', 'data', 'forward',
                                                                    'nms', 'after_nms'])
                time_sum = (t_d + t_f + t_nms + t_an)

            fps, t_fps = 1 / time_sum, 1 / t_t 

            bar_str = progress_bar.get_bar(i + 1)
            
            if cfg.measure == True:
                print(f'\rTesting: {bar_str} {i + 1}/{num_frames}, fps: {fps:.2f} | total fps: {t_fps:.2f} | '
                      f't_t: {t_t:.3f} | t_d: {t_d:.3f} | t_f: {t_f:.3f} | t_nms: {t_nms:.3f} | '
                      f't_after_nms: {t_an:.3f} | t_draw: {t_da:.3f} | t_measure: {t_m:.3f}', end='')
                
            else:
                print(f'\rTesting: {bar_str} {i + 1}/{num_frames}, fps: {fps:.2f} | total fps: {t_fps:.2f} | '
                      f't_t: {t_t:.3f} | t_d: {t_d:.3f} | t_f: {t_f:.3f} | t_nms: {t_nms:.3f} | '
                      f't_after_nms: {t_an:.3f}', end='')
                            
        
    vid.release()
    
    print('\nFinished, video generating in...: results/videos.')
    image_folder = cfg.env_path+'results/videos_images/'
    video_name = cfg.env_path+'results/videos/video.mp4'
   
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    
    video = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*"mp4v"), target_fps, frame_size)
    # the reason for saving frame and then generating video is to prevent unable generating video accurately
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    
    video.release()    
    print('\nFinished, saved in: results/videos.')
    
    # result = pd.DataFrame.from_dict(result)
    # result.to_csv('./results/result.csv')

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

    image_detection(net, cfg)