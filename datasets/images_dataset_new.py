import itertools
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import random

def make_dataset(root_dir):
    dataset = []
    for root, _, files in os.walk(root_dir):
        for file in sorted(files):
            if file.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                path = os.path.join(root, file)
                dataset.append(path)
    return dataset

class ImagesDataset(Dataset):
    def __init__(self, main_root, transform=None):
        # print(main_root)
        self.subfolders = sorted([d for d in os.listdir(main_root) if os.path.isdir(os.path.join(main_root, d))])
        assert len(self.subfolders) >= 3, "There must be at least 3 subfolders."
        self.subfolder_datasets = [make_dataset(os.path.join(main_root, sf)) for sf in self.subfolders]
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.combinations = self._combinations_gen()

    def _combinations_gen(self):
        iterators = [itertools.cycle(sf) for sf in self.subfolder_datasets]
        for index in itertools.product(*[range(len(sf)) for sf in self.subfolder_datasets]):
            same_folder_iter = iterators[index[0]]
            first_image = next(same_folder_iter)
            second_image = next(same_folder_iter)

            different_folder_index = (index[0] + 1) % len(self.subfolders)
            different_folder_iter = iterators[different_folder_index]
            third_image = next(different_folder_iter)

            yield first_image, second_image, third_image

    def __len__(self):
        return max(len(sf) for sf in self.subfolder_datasets) * len(self.subfolders)

    def __getitem__(self, index):
        combination = next(self.combinations)
        images = []
        for image_path in combination:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            images.append(image)
        # print(len(images))
        return images
