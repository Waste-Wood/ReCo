#!/usr/bin/env bash

cd /users5/kxiong/research/causal_chains/experiments

# Training on Chinese CRC dataset
python3 train.py \
  --data_dir "./data/chinese/" \
  --transformer_dir "./data/hfl-bert-wwm-ext" \
  --output_dir "./output" \
  --log_dir "./output" \
  --train "train.json" \
  --test "test.json" \
  --dev "dev.json" \
  --cuda True \
  --gpus "0 1" \
  --batch_size 24 \
  --epochs 200 \
  --evaluation_step 10 \
  --lr 1e-5 \
  --patient 10 \
  --hidden_dim 256 \
  --gmm_size 256 \
  --mode "train" \
  --lambda1 1 \
  --lambda2 0.1 \
  --set_seed True \
  --seed 4096 \
  --log_name "ReCo_Chinese.log" \
  --chain_length 5 \
  --language "chinese" \



# Training on English CRC dataset
 python3 train.py \
   --data_dir "./data/english/" \
   --transformer_dir "./data/bert-base-cased" \
   --output_dir "./output" \
   --log_dir "./output" \
   --train "train.json" \
   --test "test.json" \
   --dev "dev.json" \
   --cuda True \
   --gpus "0 1 2" \
   --batch_size  24\
   --epochs 200 \
   --evaluation_step 10 \
   --lr 1e-5 \
   --patient 10 \
   --hidden_dim 256 \
   --gmm_size 256 \
   --mode "train" \
   --lambda1 1 \
   --lambda2 0.1 \
   --set_seed True \
   --seed 4096 \
   --log_name "ReCo_English.log" \
   --chain_length 5 \
   --language "english" \





