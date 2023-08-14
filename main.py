#import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import time
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import argparse
import datetime
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = argparse.ArgumentParser(description='RAMEM implement Script')
parser.add_argument('--env_path', default='./RAMEM/')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--cfg', default='upanet80_meis', help='The configuration name to use.')
parser.add_argument('--train_bs', type=int, default=9, help='total training batch size')
parser.add_argument('--img_size', default=544, type=int, help='The image size for training.')
parser.add_argument('--resume', default=True, type=str, help='The path of the weight file to resume training with.')
parser.add_argument('--val_num', default=-1, type=int, help='The number of images for test, set to -1 for all.')
parser.add_argument('--traditional_nms', default=False, action='store_true', help='Whether to use traditional nms.')
parser.add_argument('--coco_api', action='store_true', help='Whether to use cocoapi to evaluate results.')
parser.add_argument('--af', default=False, help='anchor free')
parser.add_argument('--plot', default=True, help='plot train history by matplotlib')
parser.add_argument('--plot_every_iters', default=1, help='plot evaluattion every iters in training')
parser.add_argument('--show_every_iters', default=100, help='show process every iters in training')
parser.add_argument('--cuda', default=True, help='using gpu by cuda accelerate')
parser.add_argument('--train', default=False, help='traing sign')

# the args belows are for M-mode ecocardiography, the value will be False and unapplicalbe in other tasks
parser.add_argument('--measure', default=True, help='measure on M-mode ecocardiograms')
parser.add_argument('--scaler', default=False, help='giving image scale, otherwise automatic detect (it will slow the proccess)')
parser.add_argument('--eval_humans', default=True, help='compared with humans results in 38 patients, otherwise gt.')
# show_img in video detection real-time through cv2 will slow down FPS performance
parser.add_argument('--show_img', default=False, help='show deteced results')
parser.add_argument('--save_img', default=True, help='save deteced results')
parser.add_argument('--show_index', default=False, help='show deteced index')
parser.add_argument('--hide_bbox', default=False, action='store_true', help='Hide boxes in results.')
parser.add_argument('--hide_score', default=False, action='store_true', help='Hide scores in results.')
parser.add_argument('--hide_mask', default=False, action='store_true', help='Hide masks in results.')
parser.add_argument('--image', default='./data/image_folder/', type=str, help='The folder of images for detecting.')
parser.add_argument('--video', default='./data/video.mp4', type=str, help='The folder of images for detecting.')
parser.add_argument('--image_detect', default=False, type=str, help='using image detection')
parser.add_argument('--video_detect', default=True, type=str, help='using video detection')

parser.add_argument('--weight', default='weights/RAMEM_UPANet80_V2/best_47.145_upanet80_meis_21295.pth', type=str)
    
args = parser.parse_args()

import sys
sys.path.append(args.env_path)

from utils import timer_mini as timer
from modules.yolact import Yolact
from config import get_config
from utils.coco import COCODetection, train_collate
from utils.common_utils import save_best, save_latest
from eval import evaluation
from detect import image_detection, video_detection

cfg = get_config(args, mode='train')
cfg_name = cfg.__class__.__name__
cfg.name = args.cfg

net = Yolact(cfg)

if args.resume:
    net.load_weights(cfg.weight, cfg.cuda)
    start_step = int(cfg.weight.split('.pth')[0].split('_')[-1])
    start_step = 0
else:
    start_step = 1

dataset = COCODetection(cfg, mode='train')
                  
optimizer = optim.SGD(net.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)

train_sampler = None
main_gpu = False
num_gpu = 0
if cfg.cuda:
    cudnn.benchmark = True
    
    cudnn.fastest = True
    net = net.to('cuda')
    
    if args.resume:
        
        if args.image_detect:
            cfg = get_config(args, mode='detect')
            net.eval()
            image_detection(net, cfg, start_step)
        elif args.video_detect:
            cfg = get_config(args, mode='detect')
            net.eval()
            video_detection(net, cfg, start_step)
        else:
            net.eval()
            table, box_row, mask_row = evaluation(net, cfg, start_step)
        
        net.train()
    
        cfg = get_config(args, mode='train')
data_loader = data.DataLoader(dataset, cfg.train_bs, num_workers=0, shuffle=True,
                           collate_fn=train_collate, pin_memory=True)

epoch_seed = 0
map_tables = []
training = True
timer.reset()
step = start_step
val_step = start_step
epoch_iter = len(dataset) // args.train_bs
num_epochs = math.ceil(cfg.lr_steps[-1] / epoch_iter)
total_num_epochs = math.ceil(cfg.lr_steps[-1] / epoch_iter)
print('steps: ', cfg.lr_steps)
print(f'Number of all parameters: {sum([p.numel() for p in net.parameters()])}\n')

#%%
init = 0
time_last = 0
try:  # try-except can shut down all processes after Ctrl + C.
    if not args.train or (args.image_detect or args.video_detect):
        raise UnboundLocalError('no training')
    losses_c = []
    losses_b = []
    losses_s = []
    losses_m = []
    losses_total = []
    itrs = []
    epochs = []
    box_maps = []
    mask_maps = []
    
    test_losses_c = []
    test_losses_b = []
    test_losses_s = []
    test_losses_m = []
    test_losses_total = []
    test_itrs = []
    test_itr = 0
    best = 0.
    # count = 0
    
    for epoch in range(0, total_num_epochs, 1):
        count = 0
            
        for images, targets, masks in tqdm(data_loader):
            timer.start()
 
            if cfg.warmup_until > 0 and step <= cfg.warmup_until:  # warm up learning rate.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = (cfg.lr - cfg.warmup_init) * (step / cfg.warmup_until) + cfg.warmup_init
                    
            if step in cfg.lr_steps:  # learning rate decay.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cfg.lr * 0.1 ** cfg.lr_steps.index(step)
                   
            if cfg.cuda:
                
                images = images.to('cuda').detach()
                targets = [ann.to('cuda').detach() for ann in targets]
                masks = [mask.to('cuda').detach() for mask in masks]

            with timer.counter('for+loss'):
                
                loss_c, loss_b, loss_m, loss_s = net(images, targets, masks)
                
                losses_c.append(loss_c.item())
                losses_b.append(loss_b.item())
                losses_s.append(loss_s.item())
                losses_m.append(loss_m.item())
                itrs.append(step)

            with timer.counter('backward'):
                
                loss = loss_c + loss_b + loss_s + loss_m
                
                losses_total.append(loss.item())
                optimizer.zero_grad()

                if loss != 0:

                    loss.backward(retain_graph=False)
                 
            with timer.counter('update'):

                if loss != 0:
                    
                    optimizer.step()
                    
                    optimizer.zero_grad()

            time_this = time.time()
            if step > start_step:
                batch_time = time_this - time_last
                timer.add_batch_time(batch_time)
                time_last = time_this
            else:
                time_last = time_this

            if step % args.show_every_iters == 0 and step != start_step:

                time_name = ['batch', 'data', 'for+loss', 'backward', 'update']
                t_t, t_d, t_fl, t_b, t_u = timer.get_times(time_name)
                seconds = (cfg.lr_steps[-1] - step) * t_t
                eta = str(datetime.timedelta(seconds=seconds)).split('.')[0]
        
                l_c = loss_c.item()
                l_b = loss_b.item()
                l_s = loss_s.item()
                l_m = loss_m.item()
                for param_group in optimizer.param_groups:
                    cur_lr = param_group['lr']

                print(f'epoch: {epoch} | step: {step} | lr: {cur_lr:.2e} | ls: {loss:.3f} | l_class: {l_c:.3f} | l_box: {l_b:.3f} | '
                      f'l_mask: {l_m:.3f} | l_semantic: {l_s:.3f} | t_t: {t_t:.3f} | t_d: {t_d:.3f} | '
                      f't_fl: {t_fl:.3f} | t_b: {t_b:.3f} | t_u: {t_u:.3f} | ETA: {eta}')

            timer.reset()
            step += 1


        if epoch % args.plot_every_iters == 0:
            
            test_itr = test_itr +1
            val_step = step
            net.eval()       
            
            table, box_row, mask_row = evaluation(net, cfg, step)
            epochs.append(epoch)
            box_maps.append(box_row[1])
            mask_maps.append(mask_row[1])
            if args.plot == True:
                plt.plot(epochs, box_maps, c='blue', label='box')
                plt.plot(epochs, mask_maps, c='green', label='mask')
                plt.xlabel('Epochs')
                plt.ylabel('mAP')
                plt.title('Test')
                plt.legend(loc=0)  
                plt.grid()
                plt.show()
                
            tmp_best = (mask_row[1]+box_row[1])/2
            if tmp_best > best:
                best = tmp_best
                
                save_best(net if cfg.cuda else net, tmp_best, cfg_name, step)
            print('best: ', best)
            map_tables.append(table)
            net.train()
 
        save_latest(net if cfg.cuda else net, cfg_name, step)
        # if epoch > 0 and epoch % 60 == 0:
        if step >= cfg.lr_steps[-1]:
            break
    
    import glob
    weight = glob.glob('weights/best*')
    weight = [aa for aa in weight if cfg_name in aa]
    net.load_weights(weight[0], cfg.cuda)
    
    net.eval()
    table, box_row, mask_row = evaluation(net, cfg, step)
    
    map_tables.append(table)     
    
except KeyboardInterrupt:
    if (not cfg.cuda) or main_gpu:
        save_latest(net if cfg.cuda else net, cfg_name, step)

        print('\nValidation results during training:\n')
        for table in map_tables:
            print(table, '\n')
