# Mult-Head Co-Attention

### Run

*Note:* Data can be provided on request or obtained directly from ppmi-info.org

> python train_ppmi.py --epoch 200 --batch_size 16 --experiment "ppmi_meth_spect_unbalanced_nobal_aug_01_bs_16_256_kfold1" --missing_data 0.0 --missing_samples 0.0 --feature_size 441 --num_heads 4 --embedding_size 64 --block_shape 16 --meth --spect --num_datasets 2 --log_interval 10 --save --runs 3 --classification --learning_rate 0.00003 --dropout_keep_prob 1.0 --cuda --early_stop_epochs 20 --cuda --hidden_dim 256 --augment --aug_frac 0.1
