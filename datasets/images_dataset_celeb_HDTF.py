import os
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

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
        self.subfolders = sorted([d for d in os.listdir(main_root) if os.path.isdir(os.path.join(main_root, d))])
        assert len(self.subfolders) >= 3, "There must be at least 3 subfolders."
        self.subfolder_datasets = [make_dataset(os.path.join(main_root, sf)) for sf in self.subfolders]
        self.transform = transform
        # self.transform = transform if transform else transforms.Compose([
        #     transforms.Resize((256, 256)),
        #     transforms.ToTensor(),
        # ])

    def __len__(self):
        return max(len(sf) for sf in self.subfolder_datasets) * len(self.subfolders)

    def __getitem__(self, index):
        subfolder_indices = random.sample(range(len(self.subfolders)), 3)
        same_folder_index = subfolder_indices[0]
        different_folder_index = subfolder_indices[1]
        if len(self.subfolder_datasets[same_folder_index]) > 0:
            first_image_path = self.subfolder_datasets[same_folder_index][index % len(self.subfolder_datasets[same_folder_index])]
        else:
            # 如果列表为空，可以执行相应的处理，例如打印警告、抛出异常或跳过这个列表
            print("Warning: The subfolder dataset is empty!")
        # 如果需要抛出异常，可以使用以下代码：
        # raise ValueError("The subfolder dataset is empty!")
        # first_image_path = random.choice(self.subfolder_datasets[same_folder_index])
        second_image_path = random.choice(self.subfolder_datasets[same_folder_index])
        while first_image_path == second_image_path:
            second_image_path = random.choice(self.subfolder_datasets[same_folder_index])
        third_image_path = random.choice(self.subfolder_datasets[different_folder_index])

        images = []
        for image_path in [first_image_path, second_image_path, third_image_path]:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            images.append(image)

        return images