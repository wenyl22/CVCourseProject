import sys
import os
sys.path[0] = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random

def get_path(path, max_dataset_size=2000):
    ret = []
    for root, _, fnames in sorted(os.walk(path)):
        for fname in fnames:
            if fname.endswith('.jpg') or fname.endswith('.png'):
                ret.append(os.path.join(root, fname))
    return ret[:min(max_dataset_size, len(ret))]

def get_transform(opt, grayscale=False):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    # resize
    transform_list.append(transforms.Resize([opt.load_size, opt.load_size], transforms.InterpolationMode.BICUBIC))
    # crop
    transform_list.append(transforms.RandomCrop(opt.crop_size))
    # flip
    if 'train' in opt.phase:
        transform_list.append(transforms.RandomHorizontalFlip())
    transform_list += [transforms.ToTensor()]
    # normalize
    if grayscale:
        transform_list += [transforms.Normalize((0.5,), (0.5,))]
    else:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

class Dataset(Dataset):
    def __init__(self, opt):
        dataroot = 'dataset/' + opt.dataroot
        self.dir_A = os.path.join(dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(dataroot, opt.phase + 'B')
        self.A_paths = sorted(get_path(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(get_path(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        
        self.transform_A = get_transform(opt, grayscale=(opt.input_nc == 1))
        self.transform_B = get_transform(opt, grayscale=(opt.output_nc == 1))

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        index = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

if __name__ == '__main__':
    from options.train_options import TrainOptions
    opt = TrainOptions().parse()
    dataset = Dataset(opt)
    for i, data in enumerate(dataset):
        if i == 0:
            print(data)
            break