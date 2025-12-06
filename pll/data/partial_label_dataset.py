
import random, os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
from utils.augment import CIFARAugment, MiniImageNetAugment
from .mini_imagenet import ensure_miniimagenet

def cifar_train_dataset(root, name):
    assert name in ["cifar10", "cifar100"]
    if name=="cifar10":
        return datasets.CIFAR10(root, train=True, download=True)
    else:
        return datasets.CIFAR100(root, train=True, download=True)

def cifar_test_dataset(root, name):
    assert name in ["cifar10", "cifar100"]
    if name=="cifar10":
        return datasets.CIFAR10(root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
                ]))
    else:
        return datasets.CIFAR100(root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010))
                ]))

def miniimagenet_datasets(root):
    base = ensure_miniimagenet(root)
    # Use ImageFolder for train/val merged as train, test as test (num_classes=100 typical)
    tf_test = transforms.Compose([
        transforms.Resize(92),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])
    train_dir = os.path.join(base, "train")
    val_dir   = os.path.join(base, "val")
    test_dir  = os.path.join(base, "test")
    train_set = datasets.ImageFolder(train_dir)
    val_set   = datasets.ImageFolder(val_dir)
    # Merge train+val for stronger training
    train_imgs = [img for img,_ in train_set.samples] + [img for img,_ in val_set.samples]
    train_targets = [lab for _,lab in train_set.samples] + [lab for _,lab in val_set.samples]
    # remap labels to [0..C-1]
    import numpy as np
    uniq = sorted(set(train_targets))
    remap = {c:i for i,c in enumerate(uniq)}
    train_targets = [remap[c] for c in train_targets]
    # load raw PIL on the fly when __getitem__
    return (train_imgs, train_targets), datasets.ImageFolder(test_dir, transform=tf_test)

def split_labeled_by_rate(targets, num_classes, labeled_rate, seed=1):
    rng = np.random.RandomState(seed)
    idxs = np.arange(len(targets))
    labeled_idx = []
    for c in range(num_classes):
        cls_idx = idxs[np.array(targets) == c]
        rng.shuffle(cls_idx)
        k = int(len(cls_idx) * labeled_rate)
        labeled_idx.extend(cls_idx[:k].tolist())
    labeled_idx = np.array(labeled_idx)
    mask = np.ones(len(targets), dtype=bool)
    mask[labeled_idx] = False
    unlabeled_idx = np.where(mask)[0]
    return labeled_idx, unlabeled_idx

def split_labeled_unlabeled_by_lpc(targets, num_classes, labels_per_class, unlabeled_ratio, seed=1):
    rng = np.random.RandomState(seed)
    idxs = np.arange(len(targets))
    labeled_idx, unlabeled_idx = [], []
    for c in range(num_classes):
        cls_idx = idxs[np.array(targets) == c]
        rng.shuffle(cls_idx)
        labeled = cls_idx[:labels_per_class]
        labeled_idx.extend(labeled.tolist())
    labeled_idx = np.array(labeled_idx)
    mask = np.ones(len(targets), dtype=bool)
    mask[labeled_idx] = False
    remaining = np.where(mask)[0]
    rng.shuffle(remaining)
    num_unlabeled = int(len(labeled_idx) * unlabeled_ratio)
    unlabeled_idx = remaining[:num_unlabeled]
    return labeled_idx, unlabeled_idx

def make_partial_labels(true_label, num_classes, kmin=2, kmax=4, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    k = rng.randint(kmin, kmax+1)
    cand = set([true_label])
    while len(cand) < k:
        j = rng.randint(0, num_classes)
        cand.add(int(j))
    vec = np.zeros(num_classes, dtype=np.float32)
    for j in cand: vec[j] = 1.0
    vec = vec / vec.sum()
    return vec

class PLLDataset(Dataset):
    """
    Generic dataset wrapper supporting CIFAR-10/100 and mini-ImageNet (train side only).
    Produces two views (weak/strong), partial-label vector q for labeled subset,
    and supports unlabeled subset.
    """
    def __init__(self, args, split="train", as_unlabeled=False):
        self.args = args
        self.name = args.dataset
        self.num_classes = args.num_classes
        self.as_unlabeled = as_unlabeled
        self.seed = args.seed
        self.rng = np.random.RandomState(self.seed)
        if self.name in ["cifar10","cifar100"]:
            base = cifar_train_dataset(args.data_root, self.name)
            self.images = base.data
            self.targets = np.array(base.targets)
            self.augment = CIFARAugment(img_size=32, cutout=args.cutout)
            if self.name=="cifar100":
                self.num_classes = 100
        elif self.name=="miniimagenet":
            (train_imgs, train_targets), _ = miniimagenet_datasets(args.data_root)
            self.images = train_imgs
            self.targets = np.array(train_targets)
            self.augment = MiniImageNetAugment(img_size=84, cutout=args.cutout)
            self.num_classes = len(set(train_targets))
        else:
            raise ValueError("Unknown dataset")

        # Split
        if split=="train":
            if args.labeled_rate is not None:
                labeled_idx, unlabeled_idx = split_labeled_by_rate(self.targets, self.num_classes, args.labeled_rate, seed=self.seed)
            else:
                # derive labels_per_class if not provided
                if args.labels_per_class is None:
                    # compute from rate if possible based on dataset stats
                    if self.name=="cifar10":
                        lpc = int(5000 * 0.2)  # default 20% if not provided
                    elif self.name=="cifar100":
                        lpc = int(500 * 0.2)
                    else:
                        # for miniimagenet, choose 100 per class for default
                        lpc = 100
                    args.labels_per_class = lpc
                labeled_idx, unlabeled_idx = split_labeled_unlabeled_by_lpc(
                    self.targets, self.num_classes, args.labels_per_class, args.unlabeled_ratio, seed=self.seed
                )
            self.labeled_idx = labeled_idx
            self.unlabeled_idx = unlabeled_idx
            # Partial labels for labeled set
            self.partial_labels = {}
            for idx in labeled_idx:
                y = int(self.targets[idx])
                self.partial_labels[int(idx)] = make_partial_labels(y, self.num_classes, args.pll_ambiguity_min, args.pll_ambiguity_max, self.rng)
        else:
            # For eval, we leave as full set, but evaluation will use provided test set
            self.labeled_idx = np.arange(len(self.targets))
            self.unlabeled_idx = np.array([], dtype=int)
            self.partial_labels = None

    def __len__(self):
        if self.as_unlabeled:
            return len(self.unlabeled_idx)
        return len(self.labeled_idx)

    def __getitem__(self, i):
        if self.as_unlabeled:
            idx = int(self.unlabeled_idx[i])
        else:
            idx = int(self.labeled_idx[i])
        if self.name in ["cifar10","cifar100"]:
            pil = Image.fromarray(self.images[idx])
        else:
            # mini-ImageNet stores paths; load PIL
            pil = Image.open(self.images[idx]).convert("RGB")

        xw, xs = self.augment(pil)
        y = int(self.targets[idx])
        if self.as_unlabeled:
            return xw, xs, -1, idx, y
        else:
            q = self.partial_labels[int(idx)]
            return xw, xs, torch.from_numpy(q), idx, y
