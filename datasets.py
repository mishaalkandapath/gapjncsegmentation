from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms.v2 as v2
import torch.distributed as dist

import lightning as L

import re, cv2, os, torch, numpy as np, random
from tqdm import tqdm

class UnsupervisedDataset(Dataset):
    def __init__(self, 
                dataset_dir):
        self.images = [(os.path.join(dataset_dir, f, im), f) for f in os.listdir(dataset_dir) for im in os.listdir(os.path.join(dataset_dir, f)) if "DS" not in im]
        self.images, self.labels = zip(*self.images)
        self.num_classes = len(set(self.labels))

    def __getitem__(self, i):
        return torch.from_numpy(cv2.imread(self.images[i], -1)).to(torch.float32), int(re.findall("neuron_\d+", self.labels[i])[0][len("neuron_"):])

    def __len__(self):
        return len(self.images)


# a balanced sampler - courtesy of https://discuss.pytorch.org/t/load-the-same-number-of-data-per-class/65198/3
class BalancedBatchSampler(BatchSampler):
    """
    Each 
    
    """
    def __init__(self, allowed_indices, dataset, n_classes, n_samples):
        self.labels_list = dataset.labels
        self.labels_list = [int(re.findall("neuron_\d+", label)[0][len("neuron_"):]) for label in self.labels_list if int(re.findall("neuron_\d+", label)[0][len("neuron_"):]) in allowed_indices]
        
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy().tolist()))
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
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
    
class DistributedBatchSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None,
                 rank= None, shuffle= True,
                 seed= 0, drop_last= False, n_samples=4, n_classes=100, actual_dataset=None):
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed,
                         drop_last=drop_last)
        self.n_samples = n_samples//self.num_replicas
        self.n_classes = n_classes//self.num_replicas
        self.actual_dataset = actual_dataset
    
    def __iter__(self):
        # indices = list(super().__iter__
        indices = self.dataset[self.rank::self.num_replicas]
        batch_sampler = BalancedBatchSampler(allowed_indices=indices, dataset=self.actual_dataset, n_classes=self.n_classes, n_samples=self.n_samples)    
        return iter(batch_sampler)
    
    # def __len__(self):
    #     return self. # WTH ?


def contrastive_collate(batch):
    imgs, labels = zip(*batch)

    imgs = torch.stack(imgs, axis=0)
    labels = torch.tensor(labels)

    un_labels, counts = torch.unique(labels, return_counts=True)
    max_count = torch.max(counts)

    print(max_count)

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

    imgs = torch.cat([torch.stack(new_imgs, axis=0), imgs], axis=0)
    labels = torch.cat([torch.tensor(new_labels), labels], axis=0)

    #shuffle em 
    idx = torch.randperm(imgs.size(0))
    imgs, labels = imgs[idx, :], labels[idx]

    return imgs, labels