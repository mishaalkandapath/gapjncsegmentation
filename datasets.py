from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms.v2 as v2
import torch.distributed as dist

import lightning as L

import re, cv2, os, torch, numpy as np, random
from tqdm import tqdm
from utils import lprint

import time

def good_split(dataset, num_gpus):
    ls = [[] for _ in range(num_gpus)]
    ns = [[] for _ in range(num_gpus)]

    merry_go_round = np.array([i for i in range(num_gpus)])

    while dataset:
        if len(dataset) < num_gpus:
            merry_go_round = merry_go_round[:len(dataset)]
            num_gpus = len(dataset)
            
        get_maxes = dataset[-num_gpus:]
        for i in range(num_gpus):
            ls[merry_go_round[i]].append(get_maxes[i][1])
            ns[merry_go_round[i]].append(get_maxes[i][0])
        merry_go_round = np.roll(merry_go_round, 1)
        dataset = dataset[:-num_gpus]
    
    print("Label lengths: ", [len(l) for l in ls], "Num file lengths: ", [sum(n) for n in ns])

    return ls
        

class UnsupervisedDataset(Dataset):
    def __init__(self, 
                dataset_dir,
                metric=True):
        
        if metric:
            self.images = [(os.path.join(dataset_dir, f, im), f) for f in os.listdir(dataset_dir) for im in os.listdir(os.path.join(dataset_dir, f)) if "DS" not in im]
            self.images, self.labels = zip(*self.images)
            self.num_classes = len(set(self.labels))

            self.labels_to_files = {}
            temp_labels = list(set(self.labels))
            for i in range(len(temp_labels)):
                num = len(os.listdir(os.path.join(dataset_dir, temp_labels[i])))
                assert temp_labels[i] not in self.labels_to_files, f"{num}, {self.labels_to_files[temp_labels[i]]}"
                self.labels_to_files[temp_labels[i]] = num
            self.labels_to_files = sorted([(v, int(re.findall(r"neuron_\d+", k)[0][len("neuron_"):])) for k, v in self.labels_to_files.items()])
        else:
            self.images = [os.path.join(dataset_dir, im) for im in os.listdir(dataset_dir) if "DS" not in im]
            #setup augmentations
            self.aug = v2.Compose([
                v2.RandomCrop(size=(512, 512)),
                v2.ColorJitter(contrast=0.3, saturation=0.3),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomApply([v2.RandomRotation(degrees=(0, 180))], p=0.4),
                v2.RandomApply([v2.GaussianBlur(5)], p=0.5)
            ])
        self.metric = metric

    def __getitem__(self, i):
        if self.metric: return torch.from_numpy(cv2.imread(self.images[i], -1)).to(torch.float32), int(re.findall("neuron_\d+", self.labels[i])[0][len("neuron_"):])
        return torch.from_numpy(cv2.imread(self.images[i], -1)).to(torch.float32), self.aug(torch.from_numpy(cv2.imread(self.images[i], -1)).to(torch.float32))

    def __len__(self):
        return len(self.images)


# a balanced sampler - courtesy of https://discuss.pytorch.org/t/load-the-same-number-of-data-per-class/65198/3
class BalancedBatchSampler(BatchSampler):
    """
    Each 
    
    """
    def __init__(self, allowed_indices, dataset, n_classes, n_samples):
        self.labels_list = dataset.labels
        self.labels_list = [int(re.findall("neuron_\d+", label)[0][len("neuron_"):]) for label in self.labels_list]
        
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set([label for label in self.labels.numpy().tolist() if label in allowed_indices]))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])

        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            assert len(indices) <= (self.n_classes * self.n_samples), "bruh"
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
    
class DistributedBatchSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None,
                 rank= None, shuffle= True,
                 seed= 0, drop_last= False, n_samples=100, n_classes=4, actual_dataset=None):
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed,
                         drop_last=drop_last)
        self.n_samples = n_samples
        self.n_classes = n_classes//self.num_replicas

        self.actual_dataset = actual_dataset

        self.index_splits = None
    
    def __iter__(self):
        if self.index_splits is None: self.index_splits = good_split(self.actual_dataset.labels_to_files, self.num_replicas)
        indices = self.index_splits[self.rank]
        batch_sampler = BalancedBatchSampler(allowed_indices=indices, dataset=self.actual_dataset, n_classes=self.n_classes, n_samples=self.n_samples)    
        return iter(batch_sampler)
    
    # def __len__(self):
    #     return self. # WTH ?


def contrastive_metric_collate(batch):
    imgs, labels = zip(*batch)

    imgs = torch.stack(imgs, axis=0)
    labels = torch.tensor(labels)

    un_labels, counts = torch.unique(labels, return_counts=True)
    max_count = torch.max(counts)

    # lprint(max_count, len(un_labels), imgs.shape, labels.shape)

    imgs = imgs.unsqueeze(1) # channel = 1

    #setup augmentations
    aug = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomApply([v2.RandomRotation(degrees=(0, 180))], p=0.4),
        v2.RandomApply([v2.GaussianBlur(5)], p=0.5)
    ])

    #append augmentations to dataset:
    new_imgs, new_labels = [], []
    for j in range(un_labels.shape[0]):
        cls_imgs = imgs[labels == un_labels[j], :]
        random_choice = cls_imgs[random.randint(0, cls_imgs.shape[0]-1)]
        new_imgs.append(aug(random_choice))
        new_labels.append(un_labels[j])

        if counts[j] - max_count != 0:
            for _ in range(max_count - counts[j]):
                random_choice = cls_imgs[random.randint(0, cls_imgs.shape[0]-1)]
                new_imgs.append(aug(random_choice))
                new_labels.append(un_labels[j])

    imgs = torch.cat([imgs, torch.stack(new_imgs, axis=0)], axis=0)
    labels = torch.cat([labels, torch.tensor(new_labels)], axis=0)

    n_classes_expected = int(os.environ.get("N_CLASSES_CUSTOM", 4))//torch.cuda.device_count()
    n_samples_expected = int(os.environ.get("N_SAMPLES_CUSTOM", 100))
    batch_expected = n_classes_expected * (n_samples_expected+1)

    assert imgs.shape[0] == batch_expected, f"Expected {batch_expected} but got {imgs.shape[0]} we have \n{torch.unique(labels, return_counts=True)[1]}"

    # #shuffle em 
    idx = torch.randperm(imgs.size(0))
    imgs, labels = imgs[idx, :], labels[idx]

    assert imgs.shape[0] == batch_expected, f"Expected {batch_expected} but got {imgs.shape[0]}"
    return imgs, labels
 
def contrastive_collate(batch):
    imgs, aug_imgs = zip(*batch)

    imgs = torch.stack(imgs, axis=0)
    aug_imgs = torch.stack(aug_imgs, axis=0)

    labels = torch.arange(1, imgs.size(0)+1)
    #repeat it
    labels = torch.cat([labels, labels], axis=0)

    all_imgs = torch.cat([imgs, aug_imgs], axis=0)
    all_imgs = all_imgs[torch.randperm(all_imgs.size(0))]
    return all_imgs, labels