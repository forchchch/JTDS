### implicit args
CUDA_VISIBLE_DEVICES=1 python train_bilevel.py --task_num 313 --epochs 100  --exp_name train_common_seed5  --whether_aux 1 --train_batch_size 64 --aux_weight 1.0 --main_lr 1e-4 --aux_lr 1e-2 --method common --hyperstep 20 --corupted 0 --auxw_decay 5e-5 --n_meta_train_loss_accum 1 --corupted_rate 0.2

