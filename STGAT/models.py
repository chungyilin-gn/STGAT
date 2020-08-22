import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import sys 
import numpy as np


def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(*shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


# this efficient implementation comes from https://github.com/xptree/DeepInf/
class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out

        """
        $GN
        - nn.Parameter: 為類型轉換函數，將一個不可訓練的類型Tensor轉換成可以訓練的類型Parameter, 
                        並將這個Parameter綁定到這個module裡面
        - 通過類型轉換, self.w變成了module的一部分，成為了模型中根據訓練可改動的參數
        """
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        """
        $GN
        - xavier初始化方法中服從均勻分佈U(−a，a)
        - 分佈的參數a =增益* sqrt（6 / fan_in + fan_out）
        """
        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)


    def forward(self, h):
        
        """
        $GN:
        bs: obs length
        n:  nums of ped
        """
        bs, n = h.size()[:2] 
        

        """
        $GN:
        - self.w: torch.Tensor(n_head, f_in, f_out) ex: (4,32,16)
        - h.unsqueeze(1) ex: torch.Size([8, 1, 2, 32]) 
          => torch.matmul: (2,32) * (32, 16) => (2,16) 
          => 因為有4個head, 所以變成 (4,2,16)
          => 共有8個step, 所以變成: (8,4,2,16) == h_prime
        """
        h_prime = torch.matmul(h.unsqueeze(1), self.w)

        attn_src = torch.matmul(h_prime, self.a_src)

        attn_dst = torch.matmul(h_prime, self.a_dst)


        print("$models.py.BatchMultiHeadGraphAttention()",
              "\n h.size()",h.size(),
              "\n h.size()[:2]",h.size()[:2],
              "\n h.unsqueeze(1).size()",h.unsqueeze(1).size(),
              "\n self.w.size()",self.w.size(),
              "\n h_prime.size()",h_prime.size(),
              "\n self.a_src.size()",self.a_src.size(),
              "\n attn_src.size()",attn_src.size(),
              "\n self.a_dst.size()",self.a_dst.size(),
              "\n attn_dst.size()",attn_dst.size())

        attn_src_expand = attn_src.expand(-1, -1, -1, n)
        attn_dst_expand = attn_dst.expand(-1, -1, -1, n)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )

        print("$models.py.BatchMultiHeadGraphAttention()",
              "\n attn_src[0]",attn_src[0],
              "\n attn_src_expand[0]",attn_src_expand[0],
              "\n attn_dst[0]",attn_dst[0],
              "\n attn_dst_expand[0]",attn_dst_expand[0],
              "\n attn_dst_expand.permute[0]",attn_dst.expand(-1, -1, -1, n).permute( 0, 1, 3, 2 ),
              "\n attn[0]",attn[0]
              )

        '''
        print("$models.py.BatchMultiHeadGraphAttention()",
              "\n attn_src.expand(-1, -1, -1, n).size",attn_src.expand(-1, -1, -1, n).size(),
              "\n attn_dst.expand(-1, -1, -1, n).permute(  0, 1, 3, 2).size()",attn_dst.expand(-1, -1, -1, n).permute(  0, 1, 3, 2).size(),
              "\n attn.size",attn.size() )
        '''
        attn = self.leaky_relu(attn)
        print("$models.py.BatchMultiHeadGraphAttention()",
              "\n after leaky_relu:",attn.size()  )

        attn = self.softmax(attn)
        print("$models.py.BatchMultiHeadGraphAttention()",
              "\n after softmax:",attn.size()  )
        attn = self.dropout(attn)
        print("$models.py.BatchMultiHeadGraphAttention()",
              "\n after dropout:",attn.size()  )

        output = torch.matmul(attn, h_prime)
        print("$models.py.BatchMultiHeadGraphAttention()",
              "\n after output:",output.size()  )

        if self.bias is not None:
            return output + self.bias, attn
        else:
            return output, attn

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.n_head)
            + " -> "
            + str(self.f_in)
            + " -> "
            + str(self.f_out)
            + ")"
        )

class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout=0.2, alpha=0.2):
        super(GAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()

        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]

            self.layer_stack.append(
                BatchMultiHeadGraphAttention(
                    n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout
                )
            )
        """
        $GN:
        -Applies Instance Normalization over a 3D input 
          Input: (N, C, L)
          Output: (N, C, L) (same shape as input)
        - ex: 
          m = nn.InstanceNorm1d(2)
          input = torch.randn(2, 3, 4)*100
          output = m(input)
        """
        self.norm_list = [
            torch.nn.InstanceNorm1d(32).cuda(),
            torch.nn.InstanceNorm1d(64).cuda(),
        ]

    def forward(self, x):

        """
        $GN:
        x.size() 
        => [0,2] =>  torch.Size([8, 2, 32])
        => [4,7] =>  torch.Size([8, 3, 32])
        """
        bs, n = x.size()[:2] # ex: torch.Size([8, 2, 32]) => bs,n = (8,2)
        
        print("$models.py.GAT()",
              "\n x.size()",x.size(),
              "\n x.size()[:2]",x.size()[:2])

        for i, gat_layer in enumerate(self.layer_stack):

            """
            &GN: 
            - x: torch.Size([8, 2, 32])
            - x.permute(0, 2, 1): torch.Size([8, 32, 2])
            - self.norm_list(): normalize
            """
            x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1) #兩次轉置,回復原狀
            
            print("$models.py.GAT()",
              "\n normalized x.size()",x.size())

            x, attn = gat_layer(x)


            
            if i + 1 == self.n_layer:
                x = x.squeeze(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        else:
            return x


class GATEncoder(nn.Module):
    def __init__(self, n_units, n_heads, dropout, alpha):
        super(GATEncoder, self).__init__()
        self.gat_net = GAT(n_units, n_heads, dropout, alpha)

    def forward(self, obs_traj_embedding, seq_start_end):
        
        graph_embeded_data = []
        count =0
        for start, end in seq_start_end.data:
            curr_seq_embedding_traj = obs_traj_embedding[:, start:end, :]
            print("$modesl.py.GATEncoder()",start,end,"\n curr_seq_embedding_traj.size():" , curr_seq_embedding_traj.size())
            
            
            curr_seq_graph_embedding = self.gat_net(curr_seq_embedding_traj)

            print("curr_seq_graph_embedding.size():" ,curr_seq_graph_embedding.size())
            graph_embeded_data.append(curr_seq_graph_embedding)
            
            if count == 2:
                sys.exit()
            count+=1
        graph_embeded_data = torch.cat(graph_embeded_data, dim=1)
        return graph_embeded_data

"""
$GN: 
- TrajectoryGenerator()
    |_ self.gatencoder=GATEncoder()
        |_ self.gat_net = GAT()
"""
"""
$GN:
  nn.LSTMCell: https://pytorch.org/docs/master/generated/torch.nn.LSTMCell.html
"""
class TrajectoryGenerator(nn.Module):
    def __init__(
        self,
        obs_len,
        pred_len,
        traj_lstm_input_size,   #2
        traj_lstm_hidden_size,  #32
        n_units,   #[32, 16, 32]
        n_heads,   #[4,1]
        graph_network_out_dims, #32
        dropout,
        alpha,
        graph_lstm_hidden_size, #32
        noise_dim=(8,),
        noise_type="gaussian",
    ):
        """
        $GN: 
        - 用父類的初始化方法來初始化"繼承自父類"的屬性
        - 子類繼承了父類的所有屬性和方法，父類屬性自然會用父類方法來進行初始化
        - ex: super(xxx, self).__init__()
        """
        super(TrajectoryGenerator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len

        self.gatencoder = GATEncoder(
            n_units=n_units, n_heads=n_heads, dropout=dropout, alpha=alpha
        )

        self.graph_lstm_hidden_size = graph_lstm_hidden_size
        self.traj_lstm_hidden_size = traj_lstm_hidden_size
        self.traj_lstm_input_size = traj_lstm_input_size

        self.pred_lstm_hidden_size = (
            self.traj_lstm_hidden_size + self.graph_lstm_hidden_size + noise_dim[0]
        )

        """
        $GN:
        - 初始宣告: torch.nn.LSTMCell(input_size: int, hidden_size: int, bias: bool = True)
        - 呼叫: torch.nn.LSTMCell(input_t, (hx, cx))
        """
        self.traj_lstm_model = nn.LSTMCell(traj_lstm_input_size, traj_lstm_hidden_size)
        
        self.graph_lstm_model = nn.LSTMCell(
            graph_network_out_dims, graph_lstm_hidden_size
        )
        self.traj_hidden2pos = nn.Linear(self.traj_lstm_hidden_size, 2)
        self.traj_gat_hidden2pos = nn.Linear(
            self.traj_lstm_hidden_size + self.graph_lstm_hidden_size, 2
        )
        self.pred_hidden2pos = nn.Linear(self.pred_lstm_hidden_size, 2)

        self.noise_dim = noise_dim
        self.noise_type = noise_type

        self.pred_lstm_model = nn.LSTMCell(
            traj_lstm_input_size, self.pred_lstm_hidden_size
        )

    def init_hidden_traj_lstm(self, batch):
        return (
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
            torch.randn(batch, self.traj_lstm_hidden_size).cuda(),
        )

    def init_hidden_graph_lstm(self, batch):
        return (
            torch.randn(batch, self.graph_lstm_hidden_size).cuda(),
            torch.randn(batch, self.graph_lstm_hidden_size).cuda(),
        )

    def add_noise(self, _input, seq_start_end):
        noise_shape = (seq_start_end.size(0),) + self.noise_dim

        z_decoder = get_noise(noise_shape, self.noise_type)

        _list = []
        for idx, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            _vec = z_decoder[idx].view(1, -1)
            _to_cat = _vec.repeat(end - start, 1)
            _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
        decoder_h = torch.cat(_list, dim=0)
        return decoder_h

    def forward(
        self,
        obs_traj_rel,
        obs_traj_pos,
        seq_start_end,
        teacher_forcing_ratio=0.5,
        training_step=3,
    ):
        """
        $GN:
        - obs_traj_rel:torch.Size([8, 164, 2]) 
          => batch: 164
        """
        batch = obs_traj_rel.shape[1]  

        """
        $GN:
        nn.LSTMCell:
        - Outputs: (h_1, c_1)
          - h_1 of shape (batch, hidden_size)
          - c_1 of shape (batch, hidden_size)
        - ex:
          traj_lstm_h_t.size():torch.Size([164, 32])
          traj_lstm_c_t.size():torch.Size([164, 32])
          graph_lstm_h_t.size():torch.Size([164, 32])
          graph_lstm_c_t.size():torch.Size([164, 32])
        """
        traj_lstm_h_t, traj_lstm_c_t = self.init_hidden_traj_lstm(batch)
        graph_lstm_h_t, graph_lstm_c_t = self.init_hidden_graph_lstm(batch)

        pred_traj_rel = []
        traj_lstm_hidden_states = []
        graph_lstm_hidden_states = []

        """
        $GN:
        - ex:
          - obs_traj_rel.size():torch.Size([8, 164, 2])
          - obs_traj_rel[: self.obs_len].size(0): 8
          - obs_traj_rel[: self.obs_len].chunk(
                  obs_traj_rel[: self.obs_len].size(0), dim=0
              )
            => 拆成"obs_traj_rel[: self.obs_len].size(0)"個
              torch.Size([1, 164, 2])
        """
        for i, input_t in enumerate(
            obs_traj_rel[: self.obs_len].chunk(
                obs_traj_rel[: self.obs_len].size(0), dim=0
            )
        ):
            """
            $GN:
            - 呼叫torch.nn.LSTMCell: torch.nn.LSTMCell(input_t, (hx, cx))
            - input_t.squeeze(0)
            ex:
              - input_t.size():torch.Size([1, 164, 2])
              - input_t.squeeze(0).size(): torch.Size([164, 2])
            nn.LSTMCell:
            - Inputs: input, (h_0, c_0)
                - input of shape (batch, input_size)
                - h_0 of shape (batch, hidden_size)
                - c_0 of shape (batch, hidden_size)
            - Outputs: (h_1, c_1)
                - h_1 of shape (batch, hidden_size)
                - c_1 of shape (batch, hidden_size)
            """
            traj_lstm_h_t, traj_lstm_c_t = self.traj_lstm_model(
                input_t.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t)
            )

            if training_step == 1:
                """
                $GN:
                - LSTM輸出.size()= traj_lstm_hidden_size
                  用Linear轉換成size()=2
                - output.size(): torch.Size([164, 32])
                """
                output = self.traj_hidden2pos(traj_lstm_h_t) #traj_hidden2pos: nn.Linear(self.traj_lstm_hidden_size, 2)
                pred_traj_rel += [output]
                
            else:
                traj_lstm_hidden_states += [traj_lstm_h_t]
            
            
        if training_step == 2:
            """
            $GN:
            
            - pred_traj_rel = [
                tensor([164, 2]),
                tensor([164, 2])
                ...
              ]
              => torch.stack(pred_traj_rel): torch.Size([8, 164, 2])  
            """
            
            """
            $GN:
            - seq_start_end = [[0, 2], [2, 4], [4, 7], [7, 10], [10, 13]... [155, 158], [158, 161], [161, 164]]
            - traj_lstm_h_t: torch.Size([164, 32])
            - traj_lstm_hidden_states += [traj_lstm_h_t]:
            [
                torch.Size([164, 32]),  _
                torch.Size([164, 32])    |
                ...                      | => 8個
                torch.Size([164, 32])   _|
            ]
            => torch.stack(traj_lstm_hidden_states): torch.Size([8, 164, 32])
            """
            print("$models.py","torch.stack(traj_lstm_hidden_states):",torch.stack(traj_lstm_hidden_states).size())
            graph_lstm_input = self.gatencoder(
                torch.stack(traj_lstm_hidden_states), seq_start_end
            )
            print("$models.py","graph_lstm_input.size():",graph_lstm_input.size())
            for i in range(self.obs_len):
                graph_lstm_h_t, graph_lstm_c_t = self.graph_lstm_model(
                    graph_lstm_input[i], (graph_lstm_h_t, graph_lstm_c_t)
                )
                encoded_before_noise_hidden = torch.cat(
                    (traj_lstm_hidden_states[i], graph_lstm_h_t), dim=1
                )
                output = self.traj_gat_hidden2pos(encoded_before_noise_hidden)
                pred_traj_rel += [output]

        if training_step == 3:
            graph_lstm_input = self.gatencoder(
                torch.stack(traj_lstm_hidden_states), seq_start_end
            )
            for i, input_t in enumerate(
                graph_lstm_input[: self.obs_len].chunk(
                    graph_lstm_input[: self.obs_len].size(0), dim=0
                )
            ):
                graph_lstm_h_t, graph_lstm_c_t = self.graph_lstm_model(
                    input_t.squeeze(0), (graph_lstm_h_t, graph_lstm_c_t)
                )
                graph_lstm_hidden_states += [graph_lstm_h_t]

        if training_step == 1 or training_step == 2:
            
            """
            $GN:
            pred_traj_rel = [
              tensor([164, 2]),
              tensor([164, 2])
              ...
            ]
            => torch.stack(pred_traj_rel): torch.Size([8, 164, 2])  
            """
            return torch.stack(pred_traj_rel)
        else:
            encoded_before_noise_hidden = torch.cat(
                (traj_lstm_hidden_states[-1], graph_lstm_hidden_states[-1]), dim=1
            )
            pred_lstm_hidden = self.add_noise(
                encoded_before_noise_hidden, seq_start_end
            )
            pred_lstm_c_t = torch.zeros_like(pred_lstm_hidden).cuda()
            output = obs_traj_rel[self.obs_len-1]
            if self.training:
                for i, input_t in enumerate(
                    obs_traj_rel[-self.pred_len :].chunk(
                        obs_traj_rel[-self.pred_len :].size(0), dim=0
                    )
                ):
                    teacher_force = random.random() < teacher_forcing_ratio
                    input_t = input_t if teacher_force else output.unsqueeze(0)
                    pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                        input_t.squeeze(0), (pred_lstm_hidden, pred_lstm_c_t)
                    )
                    output = self.pred_hidden2pos(pred_lstm_hidden)
                    pred_traj_rel += [output]
                outputs = torch.stack(pred_traj_rel)
            else:
                for i in range(self.pred_len):
                    pred_lstm_hidden, pred_lstm_c_t = self.pred_lstm_model(
                        output, (pred_lstm_hidden, pred_lstm_c_t)
                    )
                    output = self.pred_hidden2pos(pred_lstm_hidden)
                    pred_traj_rel += [output]
                outputs = torch.stack(pred_traj_rel)
            return outputs
