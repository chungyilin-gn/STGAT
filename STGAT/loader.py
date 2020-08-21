from torch.utils.data import DataLoader

from STGAT.trajectories import TrajectoryDataset, seq_collate


def data_loader(args, path):

    print("$loader.py","call TrajectoryDataset()")
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        min_ped = args.min_ped_in_scene,
        delim=args.delim)
    
    

    print("$loader.py","dset.obs_traj:",dset.obs_traj.size())
    print("$loader.py","dset.pred_traj:",dset.pred_traj.size())
    print("$loader.py","dset.obs_traj_rel:",dset.obs_traj_rel.size())
    print("$loader.py","dset.pred_traj_rel:",dset.pred_traj_rel.size())
    

    print("$loader.py","call DataLoader()")
    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate,
        pin_memory=True)
    
    return dset, loader
