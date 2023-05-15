import os

import h5py
import torch
import shutil
import numpy as np
import dataset
from torchvision import transforms
from torch.autograd import Variable
import time


def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)


# 保存检查点
def save_checkpoint(state, is_best, task_id, filename='checkpoint.pth.tar'):
    os.makedirs("./" + task_id, exist_ok=True)
    torch.save(state, os.path.join('.', task_id, filename))
    # if is_best:
    #     shutil.copyfile(os.path.join('.', task_id, filename), os.path.join('.', task_id, 'best.pth.tar'))


def save_checkpoint_for_later(state, task_id, filename='checkpoint.pth.tar'):
    epoch = str(state["epoch"])
    os.makedirs("./"+task_id, exist_ok=True)
    torch.save(state, os.path.join('.', task_id, epoch+filename))


def train(labeled_list, unlabeled_list, pseudo_label_path, task, model, criterion, optimizer, epoch, args):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(labeled_list, unlabeled_list, pseudo_label_path, task,
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]),
                            seen=model.seen,
                            batch_size=args.batch_size,
                            num_workers=args.workers),
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))

    model.train()
    end = time.time()

    for i, (img, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img = img.cuda()
        img = Variable(img)
        output = model(img)

        target = target.type(torch.FloatTensor).unsqueeze(1).cuda()  # unsqueeze(0)?
        target = Variable(target)

        loss = criterion(output, target)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))


def validate(val_list, model, criterion, args):
    print('begin test')
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset([], val_list, None, 6,
                            shuffle=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ])),
        batch_size=args.batch_size)

    model.eval()

    mae = 0

    for i, (img, target, _) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)
        output = model(img)

        mae += abs(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())

    mae = mae / len(test_loader)
    print(' * MAE {mae:.3f} '.format(mae=mae))

    return mae


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    args.lr = args.original_lr

    for i in range(len(args.steps)):

        scale = args.scales[i] if i < len(args.scales) else 1

        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

