from torch.utils.data import DataLoader
from lensless_imaging_transformer.dataset import ReconstructionDataset


def get_loader(cfg):
    train_dataset = ReconstructionDataset(
        input_size=cfg.basic.input_size,
        output_size=cfg.basic.output_size,
        save_dir=cfg.dir.dataset_dir,
        split='train'
    )

    val_dataset = ReconstructionDataset(
        input_size=cfg.basic.input_size,
        output_size=cfg.basic.output_size,
        save_dir=cfg.dir.dataset_dir,
        split='val'
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.train_batch_size,
                              shuffle=True,
                              num_workers=cfg.train.GPU_num,
                              pin_memory=True,
                              drop_last=False,
                              prefetch_factor=2)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=cfg.train.GPU_num,
                            pin_memory=True,
                            drop_last=False)

    return train_loader, val_loader
