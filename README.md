# Mult-Head Co-Attention

### Run

*Note:* Data can be provided on request or obtained directly from ppmi-info.org

> python train_ppmi.py --epoch 200 --batch_size 16 --experiment "<filename>?" --feature_size 441 --num_heads 4 --embedding_size 64 --block_shape 16 --meth --spect --num_datasets 2 --log_interval 10 --save --runs 3 --classification --learning_rate 0.00003 --dropout_keep_prob 1.0 --cuda --early_stop_epochs 20 --hidden_dim 256 --augment --aug_frac 0.1

| parameter | description |
|:---:|:---:|
| epoch | number of training epochs |
| batch_size | mini-batch size |
| experiment | description of experiment |
| feature_size | number of encoder output features |
| num_head | number of heads for multi-head attention mechanism |
| embedding_size | embedding dimension of encoder output |
| block_shape | compressed embedding dimension within MHCA mechanism |
| meth | use methylation data |
| spect | use SPECT data |
| num_datasets | how many unique datasets/modes are being used |
| log_interval | how ofter to print progress to screen |
| save | save the model |
| runs | number of independent runs |
| classification | true = classification, false = regression (currently not fully supported) |
| learning_rate | learning rate |
| dropout_keep_prob | dropout used within MHCA model |
| cuda | execute model on GPU |
| early_stop_epochs | how many epochs to wait until termenanting training |
| hidden_dim | dimensionality of final hidden space |
| augment | perform data augmentation |
| aug_frac | fraction of data augmentation to apply |