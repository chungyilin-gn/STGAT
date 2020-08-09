from torch.utils.data import DataLoader

from STGAT.trajectories import TrajectoryDataset, seq_collate


def data_loader(args, path):
    print('# data_loader():',path)

    print('  -Loading Dataset(TrajectoryDataset)')
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)
    print('     --Loading DataLoader(TrajectoryDataset)')
    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate,
        pin_memory=True)
    return dset, loader
