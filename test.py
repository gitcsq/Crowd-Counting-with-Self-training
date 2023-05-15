import os.path
import json
from model import CSRNet
import argparse
import torch.nn as nn
from utils import *
from torch.utils.data import Dataset


parser = argparse.ArgumentParser(description='test anything')
parser.add_argument('test_json', metavar='TRAIN',
                    help='path to test json')
parser.add_argument('checkpoint', metavar='CHECKPOINT', type=str)
parser.add_argument('gpu', metavar='GPU', type=str,
                    help='GPU id to use.')

args = parser.parse_args()

args.batch_size = 1
args.workers = 4
args.seed = time.time()


def test_main(test_list):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)
    model = CSRNet()
    model = model.cuda()
    criterion = nn.MSELoss(size_average=False).cuda()
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    validate_save(test_list, model, criterion, args)


def validate_save(val_list, model, criterion, args):
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
    l = []
    with torch.no_grad():
        for i, (img, target, path) in enumerate(test_loader):
            img = img.cuda()
            output = model(img)
            mae_one = abs(np.sum(output.detach().cpu().numpy()) - np.sum(target.detach().cpu().numpy()))
            mae += mae_one
            l.append((path[0], float(mae_one)))
            print((path[0], mae_one))
        with open('unreliable.json', 'w') as f:
            json.dump(l, f)
        mae = mae / len(test_loader)
        print(' * MAE {mae:.3f} '.format(mae=mae))

    return mae


if __name__ == "__main__":

    with open(args.test_json, 'r') as outfile:
        test_list = json.load(outfile)
    test_main(test_list)



