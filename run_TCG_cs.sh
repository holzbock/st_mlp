# TCG Cross-Subject
python3 train.py --data_path ./tcg/ --dataset TCG --protocol xs --num_classes 4 --input_size 51 --batch_size 256 --overall_batch_size 1024 --epochs 70 --optimizer ranger --mlp_num_blocks 4 --mlp_seq_len 24 --mlp_hidden_dim 512 --channels_mlp_dim 256 --tokens_mlp_dim 32
