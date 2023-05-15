from torch.utils.data import Dataset
from image import *


class listDataset(Dataset):
    '''

            mode:                               labeled samples needed  unlabeled samples needed
            0: pretrain                         T
            1: select reliable samples                                  T (only images)
            2: generate pseudo labels 1                                 T (only images)
            3: retrain 1                        T                       T (images + pseudo labels)
            4: generate pseudo labels 2                                 T (only images)
            5: retrain 2                        T                       T (images + pseudo labels)
            6: test                                                     T
    '''

    def __init__(self, labeled_root, unlabeled_root, pseudo_label_path, mode, shape=None, shuffle=True,
                 transform=None, seen=0, batch_size=1, num_workers=4):
        self.labeled_root = labeled_root
        self.unlabeled_root = unlabeled_root
        self.root = self.labeled_root + self.unlabeled_root
        self.mode = mode
        if mode in [0, 3, 5]:
            self.root = self.root * 4
            random.shuffle(self.root)
        self.pseudo_label_path = pseudo_label_path
        # if train:
        #     root = root * 4
        # random.shuffle(root)
        self.nSamples = len(self.root)
        # self.lines = root
        self.transform = transform
        # self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.root[index]
        # img_name = os.path.basename(img_path)

        if self.mode in [0, 1, 2, 4, 6]:
            img, target = load_data(img_path)

        if self.mode in [3, 5]:
            if img_path in self.labeled_root:
                img, target = load_data(img_path)
            else:
                img, target = load_data(img_path, unlabeled=True, need_pseudo_label=True,
                                        pseudo_label_path=self.pseudo_label_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, target, img_path
