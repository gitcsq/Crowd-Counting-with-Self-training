import os.path
import json
from model import CSRNet
import argparse
import torch.nn as nn
from utils import *

# generate pseudo label
# train new models iteratively using labeled and pseudo-labeled images


parser = argparse.ArgumentParser(description='ST_after_pretraining')
parser.add_argument('unlabeled_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')
parser.add_argument('gpu', metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('task', metavar='TASK', type=str,
                    help='task id to use.')
parser.add_argument('plus', type=int, help='whether to use ST++')


args = parser.parse_args()
args.original_lr = 1e-7
args.lr = 1e-7
args.batch_size = 1
args.momentum = 0.95
args.decay = 5 * 1e-4
args.start_epoch = 0
args.epochs = 200  # 200  # 400
args.steps = [-1, 1, 100, 150]
args.scales = [1, 1, 1, 1]
args.workers = 4
args.seed = time.time()
args.print_freq = 30
args.pseudo_labels = None
args.reliable_ids = None


def generate_pseudo_label(unlabeled_list, mode, model):
    print('will generate {} pseudo labels'.format(str(len(unlabeled_list))))
    unlabeled_loader = torch.utils.data.DataLoader(
        dataset.listDataset([], unlabeled_list, None, mode,
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]),
                            seen=model.seen,
                            batch_size=args.batch_size,
                            num_workers=args.workers),
        batch_size=args.batch_size)
    model.eval()
    mae = 0
    for i, (img, target, img_path) in enumerate(unlabeled_loader):
        img_name = os.path.basename(img_path[0])
        img = img.cuda()
        img = Variable(img)
        output = model(img)
        torch.save(output.squeeze().cpu(), args.pseudo_labels + img_name.replace('.jpg', '.pth'))
        mae += abs(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())

    mae = mae / len(unlabeled_loader)
    print(' * MAE {mae:.3f} '
          .format(mae=mae))


def retrain(labeled_list, unlabeled_list, test_list, mode):
    best_prec1 = 1e6
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
        train(labeled_list, unlabeled_list, args.pseudo_labels, mode, model, criterion, optimizer, epoch, args)
        prec1 = validate(test_list, model, criterion, args)
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        if args.plus:
            dir_name = 'ST++'
            if mode == 3:
                filename = 'ST++/retrain1.pth.tar'
            else:  # mode == 5
                filename = 'ST++/retrain2.pth.tar'
        else:
            dir_name = 'ST'
            filename = 'ST/retrain-ws.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.task, filename=filename)
        with open(os.path.join('.', args.task, dir_name, 'train_process-ws.txt'), 'a+') as f:
            f.write(str(epoch) + '\t' + str(prec1.cpu().numpy()) + '\n')
        # generate_pseudo_label(unlabeled_list, 2, model)


def load_model(path):
    model = CSRNet()
    model = model.cuda()
    model.eval()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def get_list():
    with open(args.unlabeled_json, 'r') as outfile:
        unlabeled_list = json.load(outfile)
    labeled_json = args.unlabeled_json.replace('unlabeled', 'train')
    with open(labeled_json, 'r') as outfile:
        labeled_list = json.load(outfile)
    test_json = './data_split/part_A_test.json' if 'part_A' in args.unlabeled_json else './data_split/part_B_test.json'
    with open(test_json, 'r') as outfile:
        test_list = json.load(outfile)
    return labeled_list, unlabeled_list, test_list


def realse_list(a):
    del a[:]
    del a


def st_plusplus_main():
    labeled_list, unlabeled_list, test_list = get_list()

    # <===================================== Select Reliable IDs =====================================>
    print('\n\n\n================> Total stage [1]: Select reliable images for the 1st stage re-training')

    models = []
    for model_name in os.listdir(os.path.join('.', args.task)):
        if model_name.endswith('.tar'):
            path = os.path.join('.', args.task, model_name)
            model = load_model(path)
            models.append(model)
    print('{} models in total'.format(len(models)))
    select_reliable(models, unlabeled_list, 1)
    realse_list(models)
    with open(os.path.join(args.reliable_ids, 'reliable_ids.json'), 'r') as f:
        reliable_list = json.load(f)

    # <================================ Pseudo label reliable images =================================>
    print('\n\n\n================> Total stage [2]: Pseudo labeling reliable images')

    checkpoint_path = os.path.join('.', args.task, '150checkpoint.pth.tar')
    model = load_model(checkpoint_path)
    generate_pseudo_label(reliable_list, 2, model)

    # <================================== The 1st stage re-training ==================================>
    print('\n\n\n'
          '================> Total stage [3]: The 1st stage re-training on labeled and reliable unlabeled images\n')
    print('labeled images: {}\n'
          'unlabeled RELIABLE images: {}\n'.format(str(len(labeled_list)), str(len(reliable_list))))
    retrain(labeled_list, reliable_list, test_list, 3)

    # <=============================== Pseudo label unreliable images ================================>
    print('\n\n\n================> Total stage [4]: Pseudo labeling unreliable images')

    with open(os.path.join(args.reliable_ids, 'unreliable_ids.json'), 'r') as f:
        unreliable_list = json.load(f)
    checkpoint_path = os.path.join('.', args.task, 'ST++', 'retrain1.pth.tar')
    model = load_model(checkpoint_path)
    generate_pseudo_label(unreliable_list, 4, model)

    # <================================== The 2nd stage re-training ==================================>
    print('\n\n\n================> Total stage [5]: The 2nd stage re-training on labeled and all unlabeled images')
    print('labeled images: {}\n'
          'unlabeled RELIABLE images: {}\n'
          'unlabeled UNRELIABLE images: {}\n'
          .format(str(len(labeled_list)), str(len(reliable_list)), str(len(unreliable_list))))

    retrain(labeled_list, reliable_list + unreliable_list, test_list, 5)


def select_reliable(models, unlabeled_list, mode=1):
    unlabeled_loader = torch.utils.data.DataLoader(
        dataset.listDataset([], unlabeled_list, None, mode,
                            shuffle=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]),
                            batch_size=args.batch_size,
                            num_workers=args.workers),
        batch_size=args.batch_size)
    for i in range(len(models)):
        models[i].eval()
    id_to_reliability = []
    with torch.no_grad():
        for i, (img, target, img_path) in enumerate(unlabeled_loader):
            img = img.cuda()
            preds = []
            for model in models:
                preds.append(model(img).cpu().numpy())
            mae = 0

            for k in range(len(preds) - 1):
                j = k + 1
                while j < len(preds):
                    mae += abs(np.sum(preds[k]) - np.sum(preds[j]))
                    j = j + 1
            reliability = mae
            id_to_reliability.append((img_path[0], reliability))
    id_to_reliability.sort(key=lambda elem: elem[1])
    reliable_list = []
    for elem in id_to_reliability[:len(id_to_reliability) // 2]:
        reliable_list.append(elem[0])
    unreliable_list = []
    for elem in id_to_reliability[len(id_to_reliability) // 2:]:
        unreliable_list.append(elem[0])

    with open(os.path.join(args.reliable_ids, 'reliable_ids.json'), 'w') as f:
        json.dump(reliable_list, f)
    with open(os.path.join(args.reliable_ids, 'unreliable_ids.json'), 'w') as f:
        json.dump(unreliable_list, f)


def st_main():
    labeled_list, unlabeled_list, test_list = get_list()

    # <================================ Pseudo label reliable images =================================>
    print('\n\n\n================> Total stage [1]: Pseudo labeling all unlabeled images')
    #
    checkpoint_path = os.path.join('.', args.task, '150checkpoint.pth.tar')
    model = load_model(checkpoint_path)
    generate_pseudo_label(unlabeled_list, 2, model)

    # <================================== Re-training ==================================>
    print('\n\n\n'
          '================> Total stage [2]: Re-training on labeled and unlabeled images\n')
    print('labeled images: {}\n'
          'unlabeled images: {}\n'.format(str(len(labeled_list)), str(len(unlabeled_list))))

    retrain(labeled_list, unlabeled_list, test_list, 3)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    plus_name = 'ST++' if args.plus else 'ST'
    args.pseudo_labels = './{}/{}/data/pseudo_labels/'.format(args.task, plus_name)
    if os.path.exists(args.pseudo_labels):
        shutil.rmtree(args.pseudo_labels)
    os.makedirs(args.pseudo_labels)

    if args.plus:
        args.reliable_ids = './{}/{}/data/reliable_ids/'.format(args.task, plus_name)
        if os.path.exists(args.reliable_ids):
            shutil.rmtree(args.reliable_ids)
        os.makedirs(args.reliable_ids)
        st_plusplus_main()
    else:
        st_main()



