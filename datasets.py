from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler

import re, cv2, os, torch

class UnsupervisedDataset(Dataset):
    def __init__(self, 
                dataset_dir):
        self.images = [os.path.join(dataset_dir, f, im) for f in os.listdir(dataset_dir) for im in os.listdir(os.path.join(dataset_dir, f))]
        self.labels = [f*len(os.listdir(f)) for f in os.listdir(dataset_dir)]
        self.num_classes = len(set(self.labels))

    def __getitem__(self, i):
        return torch.from_numpy(cv2.imread(self.images[i], -1)).to(dtype=torch.float32), int(re.findall("neuron_\d+", self.labels[i])[len(neuron_):])


# a balanced sampler - courtesy of https://discuss.pytorch.org/t/load-the-same-number-of-data-per-class/65198/3
class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, label in loader:
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
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

def contrastive_collate(batch):
    imgs, labels = zip(*batch)

    imgs = torch.stack(imgs, axis=0)
    labels = torch.tensor(labels)

    un_labels, counts = torch.unique(labels, return_counts=True)
    max_count = torch.max(counts)

    #append augmentations to dataset:
    new_imgs, new_labels = [], []
    for j in range(num_classes):
        cls_imgs = imgs[labels == un_labels[j]]
        random_choice = cls_imgs[torch.randint(0, cls_imgs.shape[0])]
        new_imgs.append(aug(random_choice))
        new_labels.append(un_labels[j])

        if counts[j] - max_count != 0:
            for _ in range(max_count - count[j])
            random_choice = cls_imgs[torch.randint(0, cls_imgs.shape[0])]
            new_imgs.append(aug(random_choice))
            new_labels.append(un_labels[j])

    imgs = torch.cat([torch.stack(new_imgs, axis=0), imgs], axis=0)
    labels = torch.cat([torch.tensor(new_labels), labels], axis=0)

    #shuffle em 
    idx = torch.randperm(imgs.size(0))
    imgs, labels = imgs[idx, :], labels[idx]