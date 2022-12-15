import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random
import argparse
import tqdm
import numpy as np
from utils.tools import define_logger, load_json_data, process, tokenization, evaluation
import os
from transformers import AdamW
from model.mian_model import ReCo
import pdb


# Define Hyper-Parameters
def define_hps():
    parser = argparse.ArgumentParser(description='Causal Chains')

    # Data Paths
    parser.add_argument('--data_dir', type=str, default='./data/english/', help='Root directory containing the data for training')
    parser.add_argument('--transformer_dir', type=str, default='', help='Pre-trained transformer models')
    parser.add_argument('--output_dir', type=str, default='./output', help='Training outputs directory')
    parser.add_argument('--log_dir', type=str, default='./output', help='Logger file output path')
    parser.add_argument('--apex_dir', type=str, default='', help='Half-Precision repo directory')

    # Data Names
    parser.add_argument('--train', type=str, default='train.json', help='Train file name')
    parser.add_argument('--test', type=str, default='test.json', help='Test file name')
    parser.add_argument('--dev', type=str, default='dev.json', help='Dev file name')

    # Training Settings
    parser.add_argument('--cuda', type=bool, default=True, help='Whether to use gpu for training')
    parser.add_argument('--gpus', type=str, default='0 1 2', help='GPU ids for training')
    parser.add_argument('--apex', type=bool, default=False, help='Whether to use half-precision for training')
    parser.add_argument('--batch_size', type=int, default=24, help='batch_size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=1000, help='Maximum training iterations')
    parser.add_argument('--evaluation_step', type=int, default=10,
                        help='Starting evaluation after training for some steps')
    parser.add_argument('--lr', type=float, default=5e-6, help='The learning rate of gradient decent')
    parser.add_argument('--patient', type=int, default=10, help='The patient of early-stopping')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden_size used in the model')
    parser.add_argument('--gmm_size', type=int, default=256, help='The size of Gaussian Mixture Model ')
    parser.add_argument('--mode', type=str, default='train', help='Training or testing')
    parser.add_argument('--lambda1', type=float, default=1)
    parser.add_argument('--lambda2', type=float, default=0.1)
    parser.add_argument('--kl_loss_weight', type=float, default=0.1)
    parser.add_argument('--threshold_loss_weight', type=float, default=0.1)
    parser.add_argument('--scene_loss_weight', type=float, default=0.1)
    parser.add_argument('--pretrain', type=bool, default=False, help='Whether to pretrain')

    # Model and Data Settings
    parser.add_argument('--shuffle', type=bool, default=False, help='Whether to shuffle data')
    parser.add_argument('--set_seed', type=bool, default=True, help='Whether to fix seed for initialization')
    parser.add_argument('--seed', type=int, default=1004, help='The seed for initialization')
    parser.add_argument('--log_name', type=str, default='ReCo_English_0.1_0.01', help='Log file name')
    parser.add_argument('--chain_length', type=int, default=5, help='causal chain length')
    parser.add_argument('--language', type=str, default="english", help='causal chain length')

    opt = parser.parse_args()
    return opt


# Initialize Hyper-Parameters
hps = define_hps()

# Initialize Logger
logger = define_logger(hps)

# Fix random seed
if hps.set_seed:
    random.seed(hps.seed)
    np.random.seed(hps.seed)
    torch.manual_seed(hps.seed)
    torch.cuda.manual_seed(hps.seed)

logger.info("[HPS] {}".format(hps))
logger.info("[Model] {}".format(hps.transformer_dir))
logger.info("[Language] {}".format(hps.language))

# Loading & Processing Data
logger.info("[INFO] Loading Data")
train_data = load_json_data(os.path.join(hps.data_dir, hps.train))
test_data = load_json_data(os.path.join(hps.data_dir, hps.test))
dev_data = load_json_data(os.path.join(hps.data_dir, hps.dev))
logger.info("[INFO] Processing Data")
train_chains, train_chain_context, train_labels, train_threshold_labels, train_scene_labels, _ = process(hps, train_data)
test_chains, test_chain_context, test_labels, test_threshold_labels, test_scene_labels, _ = process(hps, test_data)
dev_chains, dev_chain_context, dev_labels, dev_threshold_labels, dev_scene_labels, _ = process(hps, dev_data)

# Tokenization & Padding
logger.info("[INFO] Tokenization & Padding for Data")
train_ids, train_mask, train_context_ids, train_context_mask = tokenization(hps, train_chains, train_chain_context)
test_ids, test_mask, test_context_ids, test_context_mask = tokenization(hps, test_chains, test_chain_context)
dev_ids, dev_mask, dev_context_ids, dev_context_mask = tokenization(hps, dev_chains, dev_chain_context)

# Dataset & DataLoader
logger.info("[INFO] Constructing Dataset & DataLoader for Data")

TRAIN = TensorDataset(train_ids, train_mask, train_context_ids, train_context_mask, train_labels, train_threshold_labels, train_scene_labels)
TEST = TensorDataset(test_ids, test_mask, test_context_ids, test_context_mask, test_labels, test_threshold_labels, test_scene_labels)
DEV = TensorDataset(dev_ids, dev_mask, dev_context_ids, dev_context_mask, dev_labels, dev_threshold_labels, dev_scene_labels)

train_dataloader = DataLoader(TRAIN, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)
test_dataloader = DataLoader(TEST, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)
dev_dataloader = DataLoader(DEV, batch_size=hps.batch_size, shuffle=hps.shuffle, drop_last=False)

# Initial Model & Optimizer
logger.info("[INFO] Initializing Model & Optimizer")

model = ReCo(hps)
logger.info('[Total Params] {}'.format(sum(params.numel() for params in model.parameters())))

bert_params = []
other_params = []
for name, para in model.named_parameters():
    if para.requires_grad:
        if 'event_encoder.encoder' in name:
            bert_params.append(para)
        else:
            other_params.append(para)
params = [
    {"params": bert_params, "lr": 1e-5},
    {"params": other_params, "lr": 1e-4}
]

optimizer = AdamW(params)
if hps.cuda:
    gpus = [int(x) for x in hps.gpus.split(' ')]
    model = model.cuda(gpus[0])
    if len(gpus) > 1:
        model = nn.DataParallel(model, device_ids=gpus)

# Starting Training
logger.info("[INFO] Starting Training")
step = 0
patient = 0
best_accuracy = 0.64 if hps.language == 'chinese' else 0.69
stop_train = False
idx = 0

for epoch in range(hps.epochs):
    logger.info("[Epoch] {}".format(epoch))
    bar = tqdm.trange(len(train_dataloader))
    epoch_step = 0
    total_loss = 0
    if epoch == 10:
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=hps.lr)
        logger.info('[Learning Rate]: {}'.format(hps.lr))
    for _, batch in zip(bar, train_dataloader):
        hps.mode = 'train'
        optimizer.zero_grad()
        model.train()

        epslion = torch.randn(batch[0].size()[0], hps.chain_length-1, hps.gmm_size)
        if hps.cuda:
            batch = tuple(term.cuda() for term in batch)
            epslion = epslion.cuda()
        
        chain_loss, _, _, _ = model(batch, epslion)
        if len(gpus) > 1:
            chain_loss = chain_loss.mean()

        total_loss += chain_loss.item()
        bar.set_postfix(avg_loss="{}".format(total_loss/(epoch_step+1)))
        epoch_step += 1

        chain_loss.backward()
        optimizer.step()

        if (step % hps.evaluation_step == 0 and step != 0) or epoch_step == len(train_dataloader)+1:
            logger.info("[Evaluation] Starting Evaluation on Dev Set")
            model.eval()

            epslion = torch.randn(hps.batch_size, hps.chain_length-1, hps.gmm_size)
            if hps.cuda:
                epslion = epslion.cuda()

            hps.mode = 'test'
            outputs = evaluation(hps, model, dev_dataloader, epslion)
            logger.info('[Dev Precision] \t{}'.format(outputs[0]))
            logger.info('[Dev Recall] \t{}'.format(outputs[1]))
            logger.info('[Dev F1] \t{}'.format(outputs[2]))
            logger.info('[Dev Accuracy] \t{}'.format(outputs[3]))

            outputs = evaluation(hps, model, dev_dataloader, 0)
            logger.info('[Dev Precision miu] \t{}'.format(outputs[0]))
            logger.info('[Dev Recall miu] \t{}'.format(outputs[1]))
            logger.info('[Dev F1 miu] \t{}'.format(outputs[2]))
            logger.info('[Dev Accuracy miu] \t{}'.format(outputs[3]))

            if outputs[2] >= best_accuracy:

                patient = 0
                best_accuracy = outputs[2]
                logger.info('[Evaluation] Starting Evaluation on Test Set')
                outputs = evaluation(hps, model, test_dataloader, epslion)
                logger.info('[Test Precision] \t{}'.format(outputs[0]))
                logger.info('[Test Recall] \t{}'.format(outputs[1]))
                logger.info('[Test F1] \t{}'.format(outputs[2]))
                logger.info('[Test Accuracy] \t{}'.format(outputs[3]))

                outputs = evaluation(hps, model, test_dataloader, 0)
                logger.info('[Test Precision miu] \t{}'.format(outputs[0]))
                logger.info('[Test Recall miu] \t{}'.format(outputs[1]))
                logger.info('[Test F1 miu] \t{}'.format(outputs[2]))
                logger.info('[Test Accuracy miu] \t{}'.format(outputs[3]))

                idx += 1

            else:
                patient += 1
                if patient >= hps.patient:
                    logger.info('[INFO] Stopping Training by Early Stopping')
                    stop_train = True
                    break
        step += 1
    if stop_train:
        break






























