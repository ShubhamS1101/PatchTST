from torch.utils.data import DataLoader
from .data_loader import Dataset_Custom, Dataset_Pred

def data_provider(args, flag):
    """
    Extended to support VMD decomposition and per-IMF scalers.
    """
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        Data = Dataset_Custom
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        Data = Dataset_Pred
    else:  # train / val
        shuffle_flag = True if flag == 'train' else False
        drop_last = True
        batch_size = args.batch_size
        Data = Dataset_Custom

    # 👇 pass new args to dataset
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
        use_vmd=getattr(args, "use_vmd", False),
        num_imfs=getattr(args, "num_imfs", 5)
    )
    print(f"{flag} dataset size: {len(data_set)}")

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )

    # Return scalers only during training
    if flag == 'train' and hasattr(data_set, "scalers"):
        return data_set, data_loader, data_set.scalers
    else:
        return data_set, data_loader
