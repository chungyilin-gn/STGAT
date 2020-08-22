import logging
import os
import math
from IPython import embed
import numpy as np
import sys
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

"""
$GN:
- DataLoader會集滿"batch_size個"dataset物件後, 再傳入自定義方法(ex:seq_collate)
- 從Trajectory物件,找出index對應的軌跡, 並組成out物件
- ex: 
  self.seq_start_end: [(0, 2), (2, 4), (4, 7), (7, 10) ... (25499, 25501), (25501, 25503), (25503, 25505), (25505, 25507)]
  - self.seq_start_end.length: 2112 
    self.batch_size: 64
    => iterator times for batch: 33 (2112/64)

  #1 batch in enumerate(train_loader)
  data[]= 
  [
    [#1
      self.obs_traj[0:2, :],        
      self.pred_traj[0:2, :],
      self.obs_traj_rel[0:2, :],
      self.pred_traj_rel[0:2, :],
      self.non_linear_ped[0:2],
      self.loss_mask[0:2, :],
    ],
    [#2
      self.obs_traj[2:4, :],        
      self.pred_traj[2:4, :],
      self.obs_traj_rel[2:4, :],
      self.pred_traj_rel[2:4, :],
      self.non_linear_ped[2:4],
      self.loss_mask[2:4, :],
    ],
      ... 
    [#64
      self.obs_traj[161:164, :],        
      self.pred_traj[161:164, :],
      self.obs_traj_rel[161:164, :],
      self.pred_traj_rel[161:164, :],
      self.non_linear_ped[161:164],
      self.loss_mask[161:164, :],
    ]
  ]
  - self.obs_traj[0:2, :] =
        tensor([[[10.3100,  9.5700,  8.7300,  7.9400,  7.1700,  6.4700,  5.8600,
                5.2400],
              [ 5.9700,  6.2400,  6.3400,  6.5000,  6.6200,  6.6800,  6.8200,
                6.9800]],

              [[12.4900, 11.9400, 11.0300, 10.2100,  9.3600,  8.5900,  7.7800,
                6.9600],
              [ 6.6000,  6.7700,  6.8400,  6.8100,  6.8500,  6.8500,  6.8400,
                6.8400]]])
"""

def seq_collate(data):

    
    """
    $GN:
    - ex:
      res = [['a1', 'b1', 'c1'], ['a2', 'b2', c2']]
      zip(*res)
      # this is the same as calling zip(['a1', 'b1', 'c1'], ['a2', 'b2', c2'])
      ('a1', 'a2')
      ('b1', 'b2')
      ('c1', 'c2')
    - ex: 
      data[]=
      [
        [#1
          self.obs_traj[0:2, :],        
          self.pred_traj[0:2, :],
          self.obs_traj_rel[0:2, :],
          self.pred_traj_rel[0:2, :],
          self.non_linear_ped[0:2],
          self.loss_mask[0:2, :],
        ],
        [#2
          self.obs_traj[2:4, :],        
          self.pred_traj[2:4, :],
          self.obs_traj_rel[2:4, :],
          self.pred_traj_rel[2:4, :],
          self.non_linear_ped[2:4],
          self.loss_mask[2:4, :],
        ],
          ... 
        [#64
          self.obs_traj[161:164, :],        
          self.pred_traj[161:164, :],
          self.obs_traj_rel[161:164, :],
          self.pred_traj_rel[161:164, :],
          self.non_linear_ped[161:164],
          self.loss_mask[161:164, :],
        ]
      ]
      => zip(*data)拆解:
      (
        self.obs_traj[0:2, :],      
        self.obs_traj[2:4, :],
        ...
        self.obs_traj[161:164, :]
      ),
      (
        self.pred_traj[0:2, :],      
        self.pred_traj[2:4, :],
        ...
        self.pred_traj[161:164, :]
      ),
      (
        self.obs_traj_rel[0:2, :],      
        self.obs_traj_rel[2:4, :],
        ...
        self.obs_traj_rel[161:164, :]
      ),
      (
        self.pred_traj_rel[0:2, :],      
        self.pred_traj_rel[2:4, :],
        ...
        self.pred_traj_rel[161:164, :]
      ),
      (
        self.non_linear_ped[0:2, :],      
        self.non_linear_ped[2:4, :],
        ...
        self.non_linear_ped[161:164, :]
      ),
      (
        self.loss_mask[0:2, :],      
        self.loss_mask[2:4, :],
        ...
        self.loss_mask[161:164, :]
      ),
    """
    (
        obs_seq_list,
        pred_seq_list,
        obs_seq_rel_list,
        pred_seq_rel_list,
        non_linear_ped_list,
        loss_mask_list,
    ) = zip(*data)
    '''
    print("$trajectories.py"," 呼叫自定義function:call seq_collate():",
          "\n len(data):", len(data),
          "\n data進行zip()拆解, 將不同batch但相同資料型式(ex:obs_seq類型)合併")
    '''

    # 每個time-frame軌跡數量的list       ex:[2, 2, 3, 3, 3, ...]
    _len = [len(seq) for seq in obs_seq_list]
    # 每個time-frame軌跡數量累加的list    ex:[0, 2, 4, 7, 10,...]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    # 累加的list轉換成: [[0, 2], [2, 4], [4, 7], [7, 10], [10, 13]... [155, 158], [158, 161], [161, 164]]
    seq_start_end = [
        [start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])
    ] 

    """
    $GN:
    - seq_start_end= [[0, 2], [2, 4], [4, 7], ..., [158, 161], [161, 164]]
    - obs_seq_list = 
    
    - obs_seq_list = 
      (                          _
        self.obs_traj[0:2, :],    |  
        self.obs_traj[2:4, :],    | => 64個
        ...                       |
        self.obs_traj[161:164, :]_|
      ),
        self.obs_traj[0:2, :] = 
          tensor([[[10.3100,  9.5700,  8.7300,...],
            [ 5.9700,  6.2400,  6.3400, ...]],
            [[12.4900, 11.9400, 11.0300, ...],
            [ 6.6000,  6.7700,  6.8400,  ...]]]), 

    => torch.cat(obs_seq_list, dim=0):串接成one tensor
     - tensor([
        [[10.3100,  9.5700,  8.7300,  ...],
         [ 5.9700,  6.2400,  6.3400,  ...,  ]],
        [[12.4900, 11.9400, 11.0300,  ...,  ],
         [ 6.6000,  6.7700,  6.8400,  ..., ]],
        [[12.5100, 11.5400, 10.9600,  ..., ],
         [ 6.1900,  6.0300,  5.9700,  ..., ]],
        ...])
     - size(): [164, 2, 8]

    => .permute(2, 0, 1):轉置 
     - tensor([              _   
        [[10.3100,  5.9700],  |
         [12.4900,  6.6000],  |
         [12.5100,  6.1900],  | => obs_t=1的所有軌跡(164個)
         ...,                 |
         [-1.3100, -7.4300],  |
         [-1.6500,  0.9000],  |
         [-1.7200,  1.2800]],_|
                             _
        [[ 9.5700,  6.2400],  |
         [11.9400,  6.7700],  |
         [11.5400,  6.0300],  | => obs_t=2的所有軌跡(164個)
         ...,                 |
         [-1.3100, -7.4300],  |
         [-1.6500,  0.9000],  |
         [-1.7200,  1.2800]],_|
         ...                     ...
                                => obs_t=8的所有軌跡(164個)
         ])
      - size(): [8, 164, 2]
    """
    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    '''
    print("$trajectories.py","\n ex:obs_traj",
          "\n 同類型不同frame合併, size():", (torch.cat(obs_seq_list, dim=0)).size(),
          "\n 進行轉置, size():", (torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)).size())
    '''
    
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

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = [] #儲存每個time-frame(idx)有效的軌跡數量
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []

        index_tmp = 0
        for path in all_files:

            """
            $GN:
            - Step 1: 讀取原始檔案
              - ex: zara2/train/biwi_hotel_train.txt
                data= [
                        [7.800e+02 1.000e+00 8.460e+00 3.590e+00]
                        [7.900e+02 1.000e+00 9.570e+00 3.790e+00]
                        [8.000e+02 1.000e+00 1.067e+01 3.990e+00]
                        ...
                        [1.023e+04 2.540e+02 1.810e+00 5.190e+00]
                        [1.023e+04 2.550e+02 1.200e+01 6.670e+00]
                        [1.023e+04 2.560e+02 1.151e+01 7.530e+00]
                      ]
            """
            data = read_file(path, delim)
            
            """
            $GN:
            - Step 2: 找出不重複的frame number
              - var_name: frames       
              - ex: np.unique(data[:, 0])=> 找出frame_number未重複的
                => frames: [0.0, 10.0, 20.0]
            - np.unique: 排除數組中的重複數字，並進行排序後輸出
            """
            frames = np.unique(data[:, 0]).tolist()

            frame_data = []  #含相同frame number聚合成array的陣列
            """
            $GN:
            - Step 3: 相同time_frame的element合併
              - ex:
                data[]:
                  [
                    [780.     1.     8.46   3.59]
                    [790.     1.     9.57   3.79]
                    [800.     1.    10.67   3.99]
                    [800.     2.    13.64   5.8 ]
                    [810.     1.    11.73   4.32] 
                  ]
                =>
                frame_data[]: 
                  [ 
                    array([[780.  ,   1.  ,   8.46,   3.59]]), 
                    array([[790.  ,   1.  ,   9.57,   3.79]]), 
                    array([[800.  ,   1.  ,  10.67,   3.99], [800.  ,   2.  ,  13.64,   5.8 ]]), 
                    array([[810.  ,   1.  ,  11.73,   4.32], [810.  ,   2.  ,  12.09,   5.75]])
                  ]
                frame_data[0]: array([[780.  ,   1.  ,   8.46,   3.59]]) 
                frame_data[1]: array([[790.  ,   1.  ,   9.57,   3.79]])
                =>
                frame_data[:3]:
                  [
                    array([[780.  ,   1.  ,   8.46,   3.59]]), 
                    array([[790.  ,   1.  ,   9.57,   3.79]]), 
                    array([[800.  ,   1.  ,  10.67,   3.99], [800.  ,   2.  ,  13.64,   5.8 ]]), 
                  ]
            """
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            
            """
            $GN:
            - Step 4: 預留seq_len長度
              - var_name: num_sequences
            """
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))
            

            """
            $GN:
              data[]:
              [
                [780.     1.     8.46   3.59]
                [790.     1.     9.57   3.79]
                [800.     1.    10.67   3.99]
                [800.     2.    13.64   5.8 ]
                [810.     1.    11.73   4.32] 
                ...
              ]
              => 相同frame合併:
              frame_data[]: 
              [ 
                array([[780.  ,   1.  ,   8.46,   3.59]]), 
                array([[790.  ,   1.  ,   9.57,   3.79]]), 
                array([[800.  ,   1.  ,  10.67,   3.99], [800.  ,   2.  ,  13.64,   5.8 ]]), 
                array([[810.  ,   1.  ,  11.73,   4.32], [810.  ,   2.  ,  12.09,   5.75]])
              ]
              => 每次抽出seq_len個array:
              for idx in len(frame_data):
                ->  idx=0 	[
                            array([[780.  ,   1.  ,   8.46,   3.59]]), 
                            array([[790.  ,   1.  ,   9.57,   3.79]]), 
                            ...
                            #0+seq_len
                          ]
                      
                          => 將以上array concat:
                          curr_seq_data[] =   
                          [         
                            [ 7.800e+02  1.000e+00  8.460e+00  3.590e+00]
                            [ 7.900e+02  1.000e+00  9.570e+00  3.790e+00]
                            ...
                          ]

                          => 宣告記錄軌跡的物件:
                          curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))

                    for unique ped in len(curr_seq_data):
                      =>檢查ped軌跡長度, OK: 寫入 curr_seq

                ->檢查 curr_seq長度, OK: seq_list.append(curr_seq[:num_peds_considered])

                ->  idx=1 	array([[790.  ,   1.  ,   9.57,   3.79]]), 
                      array([[800.  ,   1.  ,  10.67,   3.99], [800.  ,   2.  ,  13.64,   5.8 ]]), 
                      ...
                      #1+seq_len
                    ...
            """
            print("$trajectories.py","\n 讀取檔案:",
                  "\n -File_Path:",path.split('datasets')[1],
                  "\n -Raw Data shape:",np.array(data).shape,
                  "\n 相同frame number的行人資訊合併:",
                  "\n -frame_data[] shape:",np.array(frame_data).shape," (unique frame length=",len(frames),")",
                  "\n 資料處理:",
                  "\n for loop: 每次從frame_data抽出[seq_len個array]",
                  "\n   [本次軌跡array]=(num_peds, 2, seq_len)",
                  "\n   for loop: 從[seq_len個array]中找出獨立的行人ID(num_peds)",
                  "\n     檢查此行人在[seq_len個array]內軌跡長度是否符合seq_len",
                  "\n     若是, 此行人軌跡(shape:(2, seq_len))寫入[本次軌跡array]",
                  "\n 檢查[本次軌跡array]數量是否大於min_ped",
                  "\n 若是, (軌跡array)加入seq_list&seq_list_rel",
                  "\n")
            for idx in range(0, num_sequences * self.skip + 1, skip):
                
                """
                $GN:
                - Step 5: 每次抽取frame_data[idx : idx+seq_len]個element, 並concat
                - ex: 
                  frame_data[idx : idx + seq_len] =
               _  [
              |      array([[780.  ,   1.  ,   8.46,   3.59]]), 
              |      array([[790.  ,   1.  ,   9.57,   3.79]]), 
  每個array <-|       array([[800.  ,   1.  ,  10.67,   3.99], [800.  ,   2.  ,  13.64,   5.8 ]]),
  可能有多個行人|           
              |      ...
              |_     array([[9.700e+02,  2.000e+00,  2.010e+00,  8.000e+00],[ 9.700e+02,  3.000e+00,  3.190e+00,  6.890e+00],  [ 9.700e+02,  4.000e+00,  1.101e+01,  5.320e+00], [ 9.700e+02,  5.000e+00,  1.082e+01,  4.490e+00],  [ 9.700e+02,  6.000e+00,  2.130e+00,  6.290e+00],[ 9.700e+02,  7.000e+00,  6.860e+00,  5.830e+00],[ 9.700e+02,  8.000e+00, -9.300e-01,  9.600e-01]])
                  ]

                  =>np.concatenate():
                    curr_seq_data[] =   
                    [         
                      [ 7.800e+02  1.000e+00  8.460e+00  3.590e+00]
                      [ 7.900e+02  1.000e+00  9.570e+00  3.790e+00]
                      [ 8.000e+02  1.000e+00  1.067e+01  3.990e+00]
                      [ 8.000e+02  2.000e+00  1.364e+01  5.800e+00]
                      [ 8.100e+02  1.000e+00  1.173e+01  4.320e+00]
                      [ 8.100e+02  2.000e+00  1.209e+01  5.750e+00]
                      ...
                      [ 9.700e+02  7.000e+00  6.860e+00  5.830e+00]
                      [ 9.700e+02  8.000e+00 -9.300e-01  9.600e-01]
                    ]
                - np.concatenate: 陣列拼接
                """
                # curr_seq_data is a 20 length sequence
                curr_seq_data = np.concatenate(
                    frame_data[idx : idx + self.seq_len], axis=0
                )

                """
                $GN:
                - Step 6: 從concat後的frame_data[idx : idx+seq_len]中, 找出不重複的pedestrain ID   
                  - ex:
                    peds_in_curr_seq: [1. 2. 3. 4. 5. 6. 7. 8.]
                """
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                
                """
                $GN:
                - Step 7: 宣告記錄行人數量(peds_in_curr_seq)之絕對座標及相對座標的np.array
                  - ex:
                    curr_seq(絕對座標)/curr_seq_rel(相對座標) = 
                    [                                                                                                 _
                      [                                                                                                |
                        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]  => 共seq_len個的X絕對(相對)座標    |
                        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]  => 共seq_len個的Y絕對(相對)座標    | 
                      ]                                                                                                | => 共多少的行人(peds_in_curr_seq)                                                                                 |                                                                                        
                      ...                                                                                              | 
                      [                                                                                                |
                        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]                                  |
                        [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]                                  |
                      ]                                                                                               _|
                    ]
                """
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
              
                test_seq = []


                num_peds_considered = 0 #符合條件的行人數量
                _non_linear_ped = []

                for _, ped_id in enumerate(peds_in_curr_seq):

                    """
                    $GN:
                    - Step 8: 從curr_seq_data抽取出相同行人ID的物件
                      - ex: 
                      curr_seq_data[] =   
                      [
                        [ 7.800e+02  1.000e+00  8.460e+00  3.590e+00]
                        [ 7.900e+02  1.000e+00  9.570e+00  3.790e+00]
                        [ 8.000e+02  1.000e+00  1.067e+01  3.990e+00]
                        [ 8.000e+02  2.000e+00  1.364e+01  5.800e+00]
                        [ 8.100e+02  1.000e+00  1.173e+01  4.320e+00]
                        [ 8.100e+02  2.000e+00  1.209e+01  5.750e+00]
                        ...
                        [ 9.700e+02  7.000e+00  6.860e+00  5.830e+00]
                        [ 9.700e+02  8.000e+00 -9.300e-01  9.600e-01]
                      ]
                      =>
                      目標ped_id: 1.0 
                      =>
                      curr_ped_seq[]= 
                      [
                        [780.     1.     8.46   3.59]
                        [790.     1.     9.57   3.79]
                        [800.     1.    10.67   3.99]
                        [810.     1.    11.73   4.32]
                        [820.     1.    12.81   4.61]
                      ]
                    """
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    
                    """
                    $GN:
                    - 四捨五入到decimals=4
                    """
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)

                    """
                    $GN:
                    - curr_ped_seq的index的範圍:
                    - ex:
                      ped_id: 1.0
                      curr_ped_seq: [ [780.     1.     8.46   3.59]
                                      [790.     1.     9.57   3.79]
                                      [800.     1.    10.67   3.99]
                                      [810.     1.    11.73   4.32]
                                      [820.     1.    12.81   4.61]]
                      => pad_front: 0  / pad_end: 5
                    """
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1

                    """
                    $GN:
                    - Step 9: 確認此行人之軌跡長度是否合理
                      => (pad_end - pad_front)至少要長於(obs_len+pre_len)  
                    - continue: 強制跳出 ❮本次❯ 迴圈，繼續進入下一圈
                    """                    
                    if pad_end - pad_front != self.seq_len:
                        continue
                    
                    """
                    $GN:
                    - Step 10: 將curr_ped_seq做transpose, 使X & Y座標獨立成各自的array
                      - ex: 
                            - curr_ped_seq= 
                                [ [8.000e+02 2.000e+00 1.364e+01 5.800e+00]
                                  [8.100e+02 2.000e+00 1.209e+01 5.750e+00]
                                  [8.200e+02 2.000e+00 1.137e+01 5.800e+00]
                                  [8.300e+02 2.000e+00 1.031e+01 5.970e+00]
                                  [8.400e+02 2.000e+00 9.570e+00 6.240e+00]
                                  ...]
                            => curr_ped_seq[:, 2:] = 
                                  [
                                    [13.64  5.8 ]
                                    [12.09  5.75]
                                    [11.37  5.8 ]
                                    [10.31  5.97]
                                    [ 9.57  6.24]
                                    ...
                                  ]
                            => np.transpose(curr_ped_seq[:, 2:]) = 
                               curr_ped_seq = 
                                  [ [13.64 12.09 11.37 10.31  9.57  ...]
                                    [ 5.8   5.75  5.8   5.97  6.24 ...]]
                    """
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq  #shape: (2, 20)
                    
                    """
                    $GN:
                    - Step 11: 計算X & Y座標的相對位置   
                      - ex: 
                        idx 2 
                        ped_id 2.0 
                        curr_ped_seq = 
                        [
                          [13.64 12.09 11.37 10.31  9.57  8.73  7.94  7.17  6.47  5.86  5.24  4.87  4.51  4.2   3.95  3.47  2.82  2.01  1.28  0.54]
                          [ 5.8   5.75  5.8   5.97  6.24  6.34  6.5   6.62  6.68  6.82  6.98  7.16  7.58  7.3   7.71  7.86  8.    8.    7.82  7.4 ]
                        ]
                        rel_curr_ped_seq = 
                        [
                          [ 0.   -1.55 -0.72 -1.06 -0.74 -0.84 -0.79 -0.77 -0.7  -0.61 -0.62 -0.37 -0.36 -0.31 -0.25 -0.48 -0.65 -0.81 -0.73 -0.74]
                          [ 0.   -0.05  0.05  0.17  0.27  0.1   0.16  0.12  0.06 0.14  0.16  0.18   0.42 -0.28  0.41  0.15  0.14  0.   -0.18 -0.42]
                        ]
                    """
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]

                    """
                    $GN:
                    - 記錄通過軌跡數量檢查的行人數量
                    """
                    _idx = num_peds_considered

                    """
                    $GN:
                    - Step 12: 將軌跡數據填入index=_idx的位置
                      - ex: 
                        初始宣告curr_seq_rel=
                        curr_seq = 
                        [                                                                  _
                          [                                                                 |
                            [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]   |
                            [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]   | 
                          ]                                                                 | => 共多少的行人(peds_in_curr_seq)                                                                                 |                                                                                        
                          ...                                                               | 
                          [                                                                 |
                            [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]   |
                            [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]   |
                          ]                                                                _|
                        ]
                        =>
                        軌跡seq: 
                        curr_ped_seq = 
                        [
                          [13.64 12.09 11.37 10.31  9.57  8.73  7.94  7.17  6.47  5.86  5.24  4.87  4.51  4.2   3.95  3.47  2.82  2.01  1.28  0.54]
                          [ 5.8   5.75  5.8   5.97  6.24  6.34  6.5   6.62  6.68  6.82  6.98  7.16  7.58  7.3   7.71  7.86  8.    8.    7.82  7.4 ]
                        ]
                        rel_curr_ped_seq = 
                        [
                          [ 0.   -1.55 -0.72 -1.06 -0.74 -0.84 -0.79 -0.77 -0.7  -0.61 -0.62 -0.37 -0.36 -0.31 -0.25 -0.48 -0.65 -0.81 -0.73 -0.74]
                          [ 0.   -0.05  0.05  0.17  0.27  0.1   0.16  0.12  0.06 0.14  0.16  0.18   0.42 -0.28  0.41  0.15  0.14  0.   -0.18 -0.42]
                        ]
                        => 
                        依照行人順序寫入軌跡資料
                        curr_seq_rel[_idx, :, pad_front:pad_end]=
                        [                                                                                                 
                          [                                       _
                            [ 0.   -1.55 -0.72 -1.06 -0.74 ...]    |
                            [ 0.   -0.05  0.05  0.17  0.27  ...]   | => 第num_peds_considred個行人之軌跡
                          ]                                       _|                                                      
                          ...                                                                                              
                          [                                                                                                
                            [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]                                  
                            [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]                                  
                          ]                                                                                               
                        ]
                    """
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    
                    # Linear vs Non-Linear Trajectory
                    #_non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                """
                $GN:
                - Step 13: 檢查time frame合法軌跡的數量(num_peds_considered)是否大於min_ped
                           => 若是, 加入seq_list / seq_list_rel   
                    - ex:
                    idx=5 
                    符合的行人軌跡數量=2:
                    array([
                          [
                            [10.31,  9.57,  8.73,  7.94,  7.17,  6.47,  5.86,  5.24,  4.87, 4.51,  4.2 ,  3.95,  3.47,  2.82,  2.01,  1.28,  0.54, -0.18,   -0.83, -1.52],
                            [ 5.97,  6.24,  6.34,  6.5 ,  6.62,  6.68,  6.82,  6.98,  7.16,  7.58,  7.3 ,  7.71,  7.86,  8.  ,  8.  ,  7.82,  7.4 ,  7.06,  6.43,  6.05]
                          ],
                          [
                            [12.49, 11.94, 11.03, 10.21,  9.36,  8.59,  7.78,  6.96,  6.29,  5.62,  5.06,  4.69,  4.35,  3.76,  3.19,  2.62,  1.78,  1.01,  0.07, -0.72],
                            [ 6.6 ,  6.77,  6.84,  6.81,  6.85,  6.85,  6.84,  6.84,  7.  ,  7.1 ,  7.04,  7.  ,  7.01,  6.99,  6.89,  7.13,  7.15,  6.96,  6.91,  6.66]
                          ]
                          ]),

                    idx=27
                    符合的行人軌跡數量=2:
                    array([
                          [
                            [ 1.251e+01,  1.154e+01,  1.096e+01,  1.029e+01,  9.880e+00,9.540e+00,  8.870e+00,  8.040e+00,  7.170e+00,  6.430e+00,5.670e+00,  4.940e+00,  4.260e+00,  3.540e+00,  2.820e+00, 2.160e+00,  1.390e+00,  7.100e-01,  1.000e-02, -6.300e-01],
                            [ 6.190e+00,  6.030e+00,  5.970e+00,  6.120e+00,  6.210e+00,6.090e+00,  5.990e+00,  5.660e+00,  5.450e+00,  5.230e+00,5.160e+00,  4.950e+00,  4.710e+00,  4.540e+00,  4.350e+00,4.190e+00,  3.970e+00,  3.720e+00,  3.410e+00,  3.080e+00]
                          ],
                          [
                            [ 1.209e+01,  1.140e+01,  1.070e+01,  1.007e+01,  9.450e+00,8.900e+00,  8.210e+00,  7.490e+00,  6.730e+00,  5.980e+00,5.210e+00,  4.450e+00,  3.760e+00,  3.030e+00,  2.280e+00,1.570e+00,  8.300e-01,  1.100e-01, -6.500e-01, -1.480e+00],
                            [ 6.950e+00,  6.930e+00,  6.870e+00,  6.730e+00,  6.510e+00,6.330e+00,  6.230e+00,  6.060e+00,  5.970e+00,  5.800e+00,5.740e+00,  5.610e+00,  5.600e+00,  5.390e+00,  5.240e+00,5.060e+00,  4.870e+00,  4.680e+00,  4.300e+00,  3.900e+00]]
                          ]),
                    idx=151
                    符合的行人軌跡數量: 3:
                    array([
                          [
                            [...],
                            [...]
                          ],
                          [
                            [...],
                            [...]
                          ],
                          [
                            [...],
                            [...]
                          ]),
                    ...
                """
                if num_peds_considered > min_ped:
                    #non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)  #儲存每個time-frame(idx)有效的軌跡數量
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

                    #print("符合的行人軌跡數量:",num_peds_considered,", 更新seq_list數量:",len(seq_list))
            print("$trajectories.py"," 此檔案軌跡處理完畢\n ->更新有效軌跡數量(seq_list):",len(seq_list),"\n")
        
        #print("$trajectories.py, seq_list長度:",len(seq_list),  seq_list[:2])

        """
        $GN:
        - Step 14: 將各time-frame獨立的軌跡資料(一個time-frame可能有多個軌跡), 融合成各自獨立的軌跡資料
          - ex: 
            seq_list=
              [
                array([[[10.31,  9.57,  8.73,  7.94,  7.17,  ...],  #x座標
                        [ 5.97,  6.24,  6.34,  6.5 ,  6.62,  ...]], #y座標
                      [[12.49, 11.94, 11.03, 10.21,  9.36,  ...],
                        [ 6.6 ,  6.77,  6.84,  6.81,  6.85,  ...]]]), 
                array([[[ 1.251e+01,  1.154e+01,  1.096e+01,  1.029e+01,  9.880e+00, ...],
                        [ 6.190e+00,  6.030e+00,  5.970e+00,  6.120e+00,  6.210e+00, ...]],
                      [[ 1.209e+01,  1.140e+01,  1.070e+01,  1.007e+01,  9.450e+00, ...],
                        [ 6.950e+00,  6.930e+00,  6.870e+00,  6.730e+00,  6.510e+00, ...]]]),
                ...
              ]
              =>(40,)
            seq_list = np.concatenate(seq_list, axis=0) = 
              [
                  [ [ 1.031e+01  9.570e+00  8.730e+00  7.940e+00  7.170e+00  ...]
                    [ 5.970e+00  6.240e+00  6.340e+00  6.500e+00  6.620e+00  ...]]
                  [ [ 1.249e+01  1.194e+01  1.103e+01  1.021e+01  9.360e+00  ...]
                    [ 6.600e+00  6.770e+00  6.840e+00  6.810e+00  6.850e+00  ...]]
                  [ [ 1.251e+01  1.154e+01  1.096e+01  1.029e+01  9.880e+00  ...]
                    [ 6.190e+00  6.030e+00  5.970e+00  6.120e+00  6.210e+00  ...]]
                  [ [ 1.209e+01  1.140e+01  1.070e+01  1.007e+01  9.450e+00  ...]
                    [ 6.950e+00  6.930e+00  6.870e+00  6.730e+00  6.510e+00  ...]
                  ...
              ] 
              =>(101,2,20)
            
        """
        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        print("$trajectories.py"," 將各time-frame軌跡資料,拆成獨立的軌跡資料 \n -shape:",seq_list.shape)

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

        """
        print("$trajectories.py","obs_traj.shape:",np.array(self.obs_traj).shape)
        print("$trajectories.py","obs_traj[:5]:\n",np.array(self.obs_traj)[:5])
        print("$trajectories.py","obs_traj[-5:]:\n",np.array(self.obs_traj)[-5:])

        print("$trajectories.py","pred_traj[:5]:\n",np.array(self.pred_traj)[:5])
        print("$trajectories.py","pred_traj[-5:]:\n",np.array(self.pred_traj)[-5:])

        print("$trajectories.py","obs_traj_rel[:5]:\n",np.array(self.obs_traj_rel)[:5])
        print("$trajectories.py","obs_traj_rel[-5:]:\n",np.array(self.obs_traj_rel)[-5:])

        print("$trajectories.py","pred_traj_rel[:5]:\n",np.array(self.pred_traj_rel)[:5])
        print("$trajectories.py","pred_traj_rel[-5:]:\n",np.array(self.pred_traj_rel)[-5:])
        """

        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        
        """
        $GN:
        - 累加每個time step可用的軌跡數量當做起始idx
        - ex:
        num_peds_in_seq = [2, 2, 3,  3,  3,  3,  3, ...       2,     2,     2] => 儲存每個time-frame有效的軌跡數量
        cum_start_idx =[0, 2, 4, 7, 10, 13, 16, 19, ... , 25503, 25505, 25507]
        """
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        """
        print("$trajectories.py","num_peds_in_seq:",num_peds_in_seq[:5],"...",num_peds_in_seq[-5:])
        print("$trajectories.py","cum_start_idx:",cum_start_idx[:5],"...",cum_start_idx[-5:])
        """ 

        """
        $GN:
        - 建立每個間隔的idx
        - ex: 
          cum_start_idx = [0, 2, 4, 7, 10,...]
          seq_start_end = [(0, 2), (2, 4), (4, 7), (7, 10)...]  => 每個time-frame有效的軌跡數量= (start_index-end_index)
        """
        self.seq_start_end = [
            (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        print("$trajectories.py","記錄seq_list每個time frame行人數量的間隔:",self.seq_start_end[:64],'\n -數量:',len(self.seq_start_end[:64]),"\n -length of Batch:",math.ceil(len(self.seq_start_end)/64) )

    def __len__(self):
        return self.num_seq

    """
    $GN:
    - DataLoader在收集batch資料時(ex:enumerate(train_loader)
      =>會呼叫__getitem__()
    - 負責回傳欲顯示的資料
    - ex:
      out = [
            self.obs_traj[start:end, :],
            self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :],
            self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end],
            self.loss_mask[start:end, :],
        ]

    - 若為多num_workers(多線程), 則起始index間隔batch_size
    - ex:
      call __getitem__(), index: 64 , start: 164 , end: 166  -> 線程1起始
      call __getitem__(), index: 65 , start: 166 , end: 168
      call __getitem__(), index: 66 , start: 168 , end: 170
      call __getitem__(), index: 67 , start: 170 , end: 174
      call __getitem__(), index: 68 , start: 174 , end: 178
      call __getitem__(), index: 0 , start: 0 , end: 2       -> 線程2起始
      call __getitem__(), index: 128 , start: 404 , end: 409 -> 線程3起始
      call __getitem__(), index: 69 , start: 178 , end: 182
      call __getitem__(), index: 1 , start: 2 , end: 4
      call __getitem__(), index: 129 , start: 409 , end: 414
      call __getitem__(), index: 70 , start: 182 , end: 186
      call __getitem__(), index: 130 , start: 414 , end: 419
      call __getitem__(), index: 2 , start: 4 , end: 7
      call __getitem__(), index: 192 , start: 617 , end: 619  -> 線程4起始
      call __getitem__(), index: 71 , start: 186 , end: 190
      call __getitem__(), index: 131 , start: 419 , end: 422
      ...

    - 若為單線程：
      call __getitem__(), index: 0 , start: 0 , end: 2
      ...
      call __getitem__(), index: 63 , start: 161 , end: 164
    """
    def __getitem__(self, index):
        
        """
        $GN:
        - 從Trajectory物件,找出index對應的軌跡, 並組成out物件
        - self.seq_start_end.length: 2112 
          self.batch_size: 64
          => iterator times for batch: 33 (2112/64)
        """
        start, end = self.seq_start_end[index]  # index對應的軌跡
        out = [
            self.obs_traj[start:end, :],        #=> [start:end, :] : 每個time-frame的軌跡item起始~結束
            self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :],
            self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end],
            self.loss_mask[start:end, :],
        ]
        
        """
        if(index == 63 or index == 0 or index==1):
          print("$trajectories.py","call __getitem__():\nindex:",index,"\nstart:",start,", end:",end
                ,"\nobs_traj:", out[0]
                ,"\npred_traj:", out[1]
                ,"\nobs_traj_rel:", out[2]
                ,"\npred_traj_rel:", out[3])
        """

        return out
