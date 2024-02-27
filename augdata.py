from random import shuffle
import torch
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
from torchvision import transforms

from configs import get_train_config

args = get_train_config()

image_size = 224
num_class = 27

class Dataset(Dataset):
    def __init__(self, lines, type):
        self.args = args.parse_args()
        super(Dataset, self).__init__()
        self.data_dir = self.args.data_dir
        self.annotation_lines = lines
        self.type = type
        self.train_batches = len(self.annotation_lines)

    def __len__(self):
        return self.train_batches

    def __getitem__(self, index):
        n = len(self.annotation_lines)
        index = index % n
        img, y = self.collect_image_label(self.annotation_lines[index])

        if self.type == 'train':
            tran = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.RandomPerspective(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            temp_y = int(y) - 1
        else:
            tran = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            temp_y = int(y) - 1
            temp_y=self.onehot_encode(temp_y)

        temp_img = tran(img)
        return temp_img.float(), temp_y

    def collect_image_label(self, line):
        line = line.split('*')
        image_path = line[0]
        label = line[1]
        image = Image.open(image_path).convert("RGB")

        return image, label

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def img_augment(self, image):
        h_flip = self.rand() < 0.5
        v_flip = self.rand() < 0.5

        if h_flip:
            image = transforms.RandomHorizontalFlip()(image)
        if v_flip:
            image = transforms.RandomVerticalFlip()(image)

        return image

    def onehot_encode(self, label, n_class=num_class):
        diag = torch.eye(n_class)
        oh_vector = diag[label].view(n_class)
        return oh_vector

class MixupDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.beta_dist = torch.distributions.beta.Beta(0.2,0.2)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        if self.rand() < 0.01:
            idx_a = index
            idx_b = np.random.randint(len(self))

            image_a, label_a = self.get_oneitem(idx_a)
            image_b, label_b = self.get_oneitem(idx_b)

            if label_a == label_b:
                image = image_a
                oh_label = self.onehot_encode(label_a)
            else:
                mix_rate = self.beta_dist.sample()
                if mix_rate < 0.5:
                    mix_rate = 1.-mix_rate

                image = mix_rate*image_a+(1.-mix_rate)*image_b
                oh_label = mix_rate*self.onehot_encode(label_a)+(1.-mix_rate)*self.onehot_encode(label_b)

            return image, oh_label
        else:

            return self.dataset[index][0],self.onehot_encode(self.dataset[index][1])

    def get_oneitem(self, idx):
        image = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return image, label

    def onehot_encode(self, label, n_class=num_class):
        diag = torch.eye(n_class)
        oh_vector = diag[label].view(n_class)
        return oh_vector
    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a


if __name__ == "__main__":
    Dataset()
