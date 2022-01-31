import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from utils.data_prepare import get_data

def get_loader(cfg):
    train_dataset = get_data(
        input_size=cfg.basic.input_size,
        output_size=cfg.basic.output_size,
        save_dir=cfg.dir.dataset_dir,
        split='train'
    )

    val_dataset = get_data(
        input_size=cfg.basic.input_size,
        output_size=cfg.basic.output_size,
        save_dir=cfg.dir.dataset_dir,
        split='val'
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.train_batch_size,
                              #num_workers=cfg.train.GPU_num*4,
                              num_workers=cfg.train.GPU_num*1,
                              pin_memory=True,
                              drop_last=False,
                              sampler=train_sampler,
                              prefetch_factor=2)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            #num_workers=cfg.train.GPU_num*4,
                            num_workers=cfg.train.GPU_num*1,
                            pin_memory=True,
                            drop_last=False)

    return train_loader, val_loader
