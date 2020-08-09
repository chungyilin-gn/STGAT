import logging
import os
import math
from IPython import embed
import numpy as np

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def seq_collate(data):
    (
        obs_seq_list,
        pred_seq_list,
        obs_seq_rel_list,
        pred_seq_rel_list,
        non_linear_ped_list,
        loss_mask_list,
    ) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [
        [start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])
    ]
    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj,
        pred_traj,
        obs_traj_rel,
        pred_traj_rel,
        non_linear_ped,
        loss_mask,
        seq_start_end,
    ]

    return tuple(out)


def read_file(_path, delim="\t"):
    data = []
    if delim == "tab":
        delim = "\t"
    elif delim == "space":
        delim = " "
    with open(_path, "r") as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0

"""
 $GN:
 - 繼承"torch.Dataset"的屬性
"""
class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    
    """
      $GN:
      - 帶有兩個下劃線開頭的函數是聲明該屬性為專有，不能在類地外部被使用或直接訪問
    """
    def __init__(
        self,
        data_dir,
        obs_len=8,
        pred_len=12,
        skip=1,
        threshold=0.002,
        min_ped=1,
        delim="\t",
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """

        """
        $GN: 
        - 用父類的初始化方法來初始化"繼承自父類"的屬性
        - 子類繼承了父類的所有屬性和方法，父類屬性自然會用父類方法來進行初始化
        - ex: super(xxx, self).__init__()
        """
        super(TrajectoryDataset, self).__init__()  # @linchungyi.gn: 用"Dataset"的init()進行初始化

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        print("$GN","data_dir:",data_dir,"obs_len:",obs_len,"pred_len:",pred_len,"skip:",skip,"seq_len:",self.seq_len)

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            print("$GN", "\n# File_Path:",path)

            """
            $GN:
            - ex2: zara2/train/biwi_hotel_train.txt
              =>data = [[ 0.0	1.0	1.41	-5.68]
                        [ 0.0	2.0	0.51	-6.94]
                        [ 0.0	3.0	2.3	-4.59]
                        ...
                        [10.0	1.0	1.28	-6.35]
                        [10.0	2.0	0.55	-7.59]
                        [10.0	3.0	1.94	-4.12]
                        ...
                        [20.0	3.0	1.53	-3.49]
                        [20.0	4.0	2.73	-1.07]
                        [20.0	5.0	-1.59	0.93]
                        ...]
            """
            data = read_file(path, delim)
            
            """
            $GN:
            - np.unique: 除數組中的重複數字，並進行排序後輸出
            - np.unique(data[:, 0])=> 找出frame_number未重複的
            - ex: frames: [0.0, 10.0, 20.0]
            """
            frames = np.unique(data[:, 0]).tolist()
            print("$GN", "frames.length:",len(frames),"\nframes:",frames[:6])


            frame_data = []  # 記錄不重複的軌跡資料
            """
            $GN:
            - 利用未重複的list:"frames", 來比對並找出不重複的item of "data"
            - ex: frame_data:  [ array([[ 0.  ,  1.  ,  1.41, -5.68],
                                         [ 0.  ,  2.  ,  0.51, -6.94],
                                         [ 0.  ,  3.  ,  2.3 , -4.59],
                                         ...])
                                  array([[10.  ,  1.  ,  1.28, -6.35],
                                         [10.  ,  2.  ,  0.55, -7.59],
                                         [10.  ,  3.  ,  1.94, -4.12],
                                         [10.  ,  4.  ,  2.76, -1.77],
                                         ...])
                                  array([[20.  ,  3.  ,  1.53, -3.49],
                                         [20.  ,  4.  ,  2.73, -1.07],
                                         [20.  ,  5.  , -1.59,  0.93],
                                         ...])
                                  ...]
            """
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            print("$GN", "frame_data.length:",len(frame_data),"\nframe_data:",frame_data[:6])

            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))
            print("$GN", "\nnum_sequences:",num_sequences)


            for idx in range(0, num_sequences * self.skip + 1, skip):
                # curr_seq_data is a 20 length sequence
                curr_seq_data = np.concatenate(
                    frame_data[idx : idx + self.seq_len], axis=0
                )
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []

                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1

                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    #_non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    #non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(seq_list[:, :, : self.obs_len]).type(
            torch.float
        )
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len :]).type(
            torch.float
        )
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, : self.obs_len]).type(
            torch.float
        )
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len :]).type(
            torch.float
        )
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :],
            self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :],
            self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end],
            self.loss_mask[start:end, :],
        ]
        return out
