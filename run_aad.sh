# AAD
python3 train.py --data_path ./act_and_drive/ --dataset AAD --num_classes 12 --input_size 39 --batch_size 512 --overall_batch_size 2048 --epochs 80 --mlp_num_blocks 2 --mlp_seq_len 90 --mlp_hidden_dim 512 --channels_mlp_dim 256 --tokens_mlp_dim 64 --convert_coord 2
