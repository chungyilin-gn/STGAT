
Train_Colab.py->train():
	for batch_idx, batch in enumerate(train_loader)  # len(train_loader): 33 (2112/64)
		(
	        obs_traj,
	        pred_traj_gt,
	        obs_traj_rel,
	        pred_traj_gt_rel,
	        non_linear_ped,
	        loss_mask,
	        seq_start_end,
	    ) = batch
		
		model_input = obs_traj_rel #torch.Size([8, 164, 2])
		model(
		    model_input, obs_traj, seq_start_end, 1, training_step
		)

--------
model.py->TrajectoryGenerator().forward(
		    model_input, obs_traj, seq_start_end, 1, training_step
		):

	batch = obs_traj_rel.shape[1]  # 164
	traj_lstm_h_t, traj_lstm_c_t = self.init_hidden_traj_lstm(batch)  	# torch.Size([164, 32])
	graph_lstm_h_t, graph_lstm_c_t = self.init_hidden_graph_lstm(batch) # torch.Size([164, 32])

	for i, input_t in enumerate( # input_t.size(): torch.Size([1, 164, 2])
	    obs_traj_rel[: self.obs_len].chunk(
	        obs_traj_rel[: self.obs_len].size(0), dim=0
	    )
	):

		# LSTM encoding # 
		# traj_lstm_model( input, (h_t, c_t) ) 
		# input_t: torch.Size([1, 164, 2])
		# input_t.squeeze(0): torch.Size([164, 2])
		traj_lstm_h_t, traj_lstm_c_t = self.traj_lstm_model(
	        input_t.squeeze(0), (traj_lstm_h_t, traj_lstm_c_t) 
	    )
		# => traj_lstm_h_t: torch.Size([164, 32])
		traj_lstm_hidden_states += [traj_lstm_h_t]
		"""
		[
	        torch.Size([164, 32]),  _
	        torch.Size([164, 32])    |
	        ...                      | => 8個
	        torch.Size([164, 32])   _|
        ]
		"""

		# torch.stack(traj_lstm_hidden_states): torch.Size([8, 164, 32])
		# seq_start_end= [[0, 2], [2, 4], [4, 7], [7, 10], [10, 13]... [155, 158], [158, 161], [161, 164]]
		# len(seq_start_end): 2112
		graph_lstm_input = self.gatencoder(
            torch.stack(traj_lstm_hidden_states), seq_start_end
        )

--------
model.py->GATEncoder().forward(obs_traj_embedding, seq_start_end):
	for start, end in seq_start_end.data:
		curr_seq_embedding_traj = obs_traj_embedding[:, start:end, :]  # 某個time-step場景的行人軌跡
		"""
		ex: start,end  = tensor(0, device='cuda:0'), tensor(2, device='cuda:0')
		curr_seq_embedding_traj.size(): torch.Size([8, 2, 32])
		"""

		curr_seq_graph_embedding = self.gat_net(curr_seq_embedding_traj)

--------
model.py->GAT().forward(x):

	bs, n = x.size()[:2] # ex: torch.Size([8, 2, 32]) => bs,n = (8,2)
        
    for i, gat_layer in enumerate(self.layer_stack):

        """
        &GN: 
        - x: torch.Size([8, 2, 32])
        - x.permute(0, 2, 1): torch.Size([8, 32, 2])
        - self.norm_list(): normalize
        """
        x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1) #兩次轉置,回復原狀
        
        x, attn = gat_layer(x)


--------
model.py->BatchMultiHeadGraphAttention().forward(h):

	bs, n = x.size()[:2] # ex: h: torch.Size([8, 2, 32]) => bs,n = (8,2)torch.Size([8, 2, 32]) 

	# h: 				torch.Size([8, 2, 32])  => this scene, 2 ped's traj encoding of 8 steps
	# h.unsqueeze(1):	torch.Size([8, 1, 2, 32])
	# w: 				torch.Tensor(4, 32, 16)
	# h_prime: 			torch.Size([8, 4, 2, 16]) 

	#1  (2, 32) * (32, 16) = (2, 16)
	#2  1 step has 4 head: (4,2,16)
	#3  8 steps: (8,4,2,16) 
	h_prime = torch.matmul(h.unsqueeze(1), self.w) 

	# h_prime: 	torch.Size([8, 4, 2, 16]) 
	# a_src: 	torch.Size([4, 16, 1]) 
	# attn_src: torch.Size([8, 4, 2, 1]) 

    attn_src = torch.matmul(h_prime, self.a_src)

    attn_dst = torch.matmul(h_prime, self.a_dst)










