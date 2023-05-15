import sys
import os

import warnings

from model import CSRNet

from utils import save_checkpoint, save_checkpoint_for_later, train, validate, AverageMeter, adjust_learning_rate

import torch
import torch.nn as nn
from torchvision import datasets, transforms

import numpy as np
import argparse
import json
import cv2
import dataset
import time

parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('test_json', metavar='TEST',
                    help='path to test json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')

parser.add_argument('gpu',metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('task',metavar='TASK', type=str,
                    help='task id to use.')


def main():
    
    global args, best_prec1
    
    best_prec1 = 1e6
    
    args = parser.parse_args()
    args.original_lr = 1e-7
    args.lr = 1e-7
    args.batch_size    = 1
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 200  #400
    args.steps         = [-1,1,100,150]
    args.scales        = [1,1,1,1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 30
    with open(args.train_json, 'r') as outfile:        
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:       
        val_list = json.load(outfile)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)
    
    model = CSRNet()
    
    model = model.cuda()
    
    criterion = nn.MSELoss(size_average=False).cuda()
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))
            
    for epoch in range(args.start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch, args)
        train(train_list, [], None, 0, model, criterion, optimizer, epoch, args)
        prec1 = validate(val_list, model, criterion, args)
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.task)
        if epoch + 1 in [args.epochs // 4, args.epochs * 2 // 4, args.epochs * 3 // 4, args.epochs]:
            save_checkpoint_for_later({
                'epoch': epoch + 1,
                'arch': args.pre,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, args.task)
        with open(os.path.join('.', args.task, 'train_data.txt'), 'a+') as f:
            f.write(str(epoch) + '\t' + str(prec1.cpu().numpy()) + '\n')


if __name__ == '__main__':
    main()
