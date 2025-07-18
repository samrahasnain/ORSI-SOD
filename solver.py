import torch
from torch.nn import functional as F
from conformer import build_model
import numpy as np
import os
import cv2
import time

import torch.nn as nn
import argparse
import os.path as osp
import os
size_coarse = (10, 10)
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from  utils import  count_model_flops,count_model_params
from PIL import Image
import json



class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every

        # Build model
        self.net = build_model(self.config.network, self.config.arch)

        # Load weights
        if config.mode == 'test':
            print(f'Loading pre-trained model for testing from {self.config.model}...')
            self.net.load_state_dict(torch.load(self.config.model, map_location=torch.device('cpu')))
        elif config.mode == 'train':
            if self.config.load == '':
                print("Loading pre-trained ImageNet weights for fine-tuning")
                self.net.JLModule.load_pretrained_model(self.config.pretrained_model)
            else:
                print('Loading pretrained model to resume training')
                self.net.load_state_dict(torch.load(self.config.load))

        # Use GPU if available
        if self.config.cuda:
            self.net = self.net.cuda()

        #self.scripted_net = torch.jit.script(self.net)

        # Optimizer and LR scheduler
        self.lr = self.config.lr
        self.wd = self.config.wd
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        # 🔑 Add ReduceLROnPlateau scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,   # reduce LR by half
            patience=5,   # wait 5 epochs without improvement
            verbose=True
        )

        self.print_network(self.net, 'Conformer based RSI-SOD Structure')

    def print_network(self, model, name):
        num_params_t = 0
        num_params = 0
        for p in model.parameters():
            if p.requires_grad:
                num_params_t += p.numel()
            else:
                num_params += p.numel()
        print(name)
        print(f"The number of trainable parameters: {num_params_t}")
        print(f"The number of parameters: {num_params}")
        print(f'Flops: {count_model_flops(model)}')
        print(f'Parameters: {count_model_params(model)}')


    def test(self):
        print('Testing...')
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size, depth = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size']), \
                                           data_batch['depth']
            with torch.no_grad():
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    images = images.to(device)
                    depth = depth.to(device)

                #input = torch.cat((images, depth), dim=0)
                if self.config.cuda:
                    torch.cuda.synchronize()
               
                start_time = time.time()  # Timing starts AFTER synchronization
                preds,coarse_sal_rgb,coarse_sal_depth,sal_edge_rgbd0,sal_edge_rgbd1,sal_edge_rgbd2= self.net(images,depth)
                if self.config.cuda:
                    torch.cuda.synchronize()

                frame_time = time.time() - start_time  # Time for one frame
                print(frame_time)           
                preds = F.interpolate(preds, tuple(im_size), mode='bilinear', align_corners=True)
                pred = np.squeeze(torch.sigmoid(preds)).cpu().data.numpy()
                #print(pred.shape)
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                multi_fuse = 255 * pred
                filename = os.path.join(self.config.test_folder, name[:-4] + '_RSI.png')
                cv2.imwrite(filename, multi_fuse)
                
        time_e = time.time()
        print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        print('Test Done!')
    
  
    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        loss_vals = []

        for epoch in range(self.config.epoch):
            r_sal_loss = 0
            r_sal_loss_item = 0

            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_depth, sal_label, sal_edge = data_batch['sal_image'], data_batch['sal_depth'], data_batch['sal_label'], data_batch['sal_edge']

                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue

                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    sal_image, sal_depth, sal_label, sal_edge = sal_image.to(device), sal_depth.to(device), sal_label.to(device), sal_edge.to(device)

                self.optimizer.zero_grad()

                sal_label_coarse = F.interpolate(sal_label, size_coarse, mode='bilinear', align_corners=True)
                sal_final, coarse_sal_rgb, coarse_sal_depth, sal_edge_rgbd0, sal_edge_rgbd1, sal_edge_rgbd2 = self.net(sal_image, sal_depth)

                sal_loss_coarse_rgb = F.binary_cross_entropy_with_logits(coarse_sal_rgb, sal_label_coarse, reduction='sum')
                sal_loss_coarse_depth = F.binary_cross_entropy_with_logits(coarse_sal_depth, sal_label_coarse, reduction='sum')
                sal_final_loss = F.binary_cross_entropy_with_logits(sal_final, sal_label, reduction='sum')
                edge_loss_rgbd0 = F.smooth_l1_loss(sal_edge_rgbd0, sal_edge)
                edge_loss_rgbd1 = F.smooth_l1_loss(sal_edge_rgbd1, sal_edge)
                edge_loss_rgbd2 = F.smooth_l1_loss(sal_edge_rgbd2, sal_edge)

                sal_loss_fuse = sal_final_loss + 512 * edge_loss_rgbd0 + 1024 * edge_loss_rgbd1 + 2048 * edge_loss_rgbd2 + sal_loss_coarse_rgb + sal_loss_coarse_depth
                sal_loss = sal_loss_fuse / (self.iter_size * self.config.batch_size)
                r_sal_loss += sal_loss.data
                r_sal_loss_item += sal_loss.item() * sal_image.size(0)

                sal_loss.backward()
                self.optimizer.step()

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/epoch_%d.pth' % (self.config.save_folder, epoch + 1))

            train_loss = r_sal_loss_item / len(self.train_loader.dataset)
            loss_vals.append(train_loss)

            print(f'Epoch:[{epoch+1:2d}/{self.config.epoch}] | Train Loss : {train_loss:.6f}')

            # 🔑 Step the scheduler with the train loss
            self.scheduler.step(train_loss)

            # Optional: show current LR
            for param_group in self.optimizer.param_groups:
                print(f"Current LR: {param_group['lr']}")

        # Final save
        torch.save(self.net.state_dict(), '%s/final.pth' % self.config.save_folder)

        
