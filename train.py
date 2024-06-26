from options.train_options import TrainOptions
import dataset.dataset as ds
from torch.utils.data import DataLoader
def main():
    opt = TrainOptions().parse()
    train_dataset = ds.Dataset(opt)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    print('The number of training images = %d' % len(train_dataset))
if __name__=='__main__':
    main()