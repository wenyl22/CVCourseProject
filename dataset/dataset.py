import sys
import os
sys.path[0] = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

def get_path(path, max_dataset_size = 3000):
    ret = []
    for root, _, fnames in sorted(os.walk(path)):
        for fname in fnames:
            if fname.endswith('.jpg') or fname.endswith('.png'):
                ret.append(os.path.join(root, fname))
    return ret[:min(max_dataset_size, len(ret))]

def get_transform(opt, flag = 0):
    transform_list = []
    # resize
    transform_list.append(transforms.Resize([opt.load_size, opt.load_size], transforms.InterpolationMode.BICUBIC))
    # crop
    transform_list.append(transforms.RandomCrop(opt.crop_size))
    # flip
    if 'train' in opt.phase:
        transform_list.append(transforms.RandomHorizontalFlip())
    # normalize
    if flag:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))]
    else:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

class Dataset(Dataset):
    def __init__(self, opt):
        dataroot = 'dataset/' + opt.dataroot
        self.dir_A = os.path.join(dataroot, opt.phase + '_A')
        self.dir_A_blur = os.path.join(dataroot, opt.phase + '_A_blur')
        self.dir_A_canny = os.path.join(dataroot,  opt.phase + '_A_canny')

        self.dir_B = os.path.join(dataroot, opt.phase + '_B')
        self.dir_B_blur = os.path.join(dataroot, opt.phase + '_B_blur')
        self.dir_B_canny = os.path.join(dataroot, opt.phase + '_B_canny')

        self.A_paths = sorted(get_path(self.dir_A, opt.max_dataset_size))
        self.A_blur_paths = sorted(get_path(self.dir_A_blur, opt.max_dataset_size))
        self.A_canny_paths = sorted(get_path(self.dir_A_canny, opt.max_dataset_size))

        self.B_paths = sorted(get_path(self.dir_B, opt.max_dataset_size))
        self.B_blur_paths = sorted(get_path(self.dir_B_blur, opt.max_dataset_size))
        self.B_canny_paths = sorted(get_path(self.dir_B_canny, opt.max_dataset_size))

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.transform = get_transform(opt, 0)
        self.transform_with_canny = get_transform(opt, 1)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        A_blur_path = self.A_blur_paths[index % self.A_size]
        A_canny_path = self.A_canny_paths[index % self.A_size]

        B_path = self.B_paths[index % self.B_size]
        B_blur_path = self.B_blur_paths[index % self.B_size]
        B_canny_path = self.B_canny_paths[index % self.B_size]

        A_img = transforms.ToTensor()(Image.open(A_path).convert('RGB'))
        A_blur_img = transforms.ToTensor()(Image.open(A_blur_path).convert('RGB'))
        A_canny_img = transforms.ToTensor()(Image.open(A_canny_path).convert('L'))

        B_img = transforms.ToTensor()(Image.open(B_path).convert('RGB'))
        B_blur_img = transforms.ToTensor()(Image.open(B_blur_path).convert('RGB'))
        B_canny_img = transforms.ToTensor()(Image.open(B_canny_path).convert('L'))
        # concate A_img and A_canny_img
        
        CA = torch.cat((A_img, A_canny_img), dim = 0)
        CA = self.transform_with_canny(CA)
        CB = torch.cat((B_img, B_canny_img), dim = 0)
        CB = self.transform_with_canny(CB)
        A = CA[:3]
        A_canny = CA[3].unsqueeze(0)
        A_blur = self.transform(A_blur_img)

        B = CB[:3]
        B_canny = CB[3].unsqueeze(0)
        B_blur = self.transform(B_blur_img)
        #print(A.shape, B.shape, A_blur.shape, B_blur.shape, A_canny.shape, B_canny.shape)
        return {'A': A, 'B': B, 'A_blur': A_blur, 'B_blur': B_blur, 'A_canny': A_canny, 'B_canny': B_canny, 'A_paths': A_path, 'B_paths': B_path}
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