
import os
import torch
from options.test_options import TestOptions
import dataset.dataset as ds
from model.cyclegan import CycleGAN
import utils.utils as utils
from torch.utils.data import DataLoader

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 0 
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    dataset = ds.Dataset(opt)
    dataset = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)
    model = CycleGAN(opt)
    model.load_networks(40)
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    dir = f'{opt.results_dir}/{opt.name}'
    if not os.path.exists(opt.results_dir):
        os.mkdir(opt.results_dir)
    if not os.path.exists(dir):
        os.mkdir(dir)
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.input(data)        
        with torch.no_grad():
            model.forward()
        pic = model.get_current_visuals()
        for key, value in pic.items():
            if key != 'fake_A' and key != 'real_B':
                continue
            img = utils.tensor2im(value)
            utils.save_image(img, f'{dir}/{i}_{key}.png')
 