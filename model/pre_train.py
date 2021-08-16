import argparse
from datetime import datetime
import os
import pickle
import logging
import time
from sklearn.metrics import accuracy_score, roc_auc_score
import random

import numpy as np
import torch
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.functional as F

from dataset import load_and_process_data
from seq2seq import Encoder,AttentionDecoder,Seq2Seq

timestamp = datetime.now().strftime('%Y-%m%d_%H%M%S')
date, hour = timestamp.split('_')
os.makedirs('log/{}'.format(date), exist_ok=True)
os.makedirs('checkpoints/{}'.format(date), exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--max_stay', type=int, default=16)
    parser.add_argument('--pre_window', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--emb_size', type=int, default=32)
    parser.add_argument('--hid_size', type=int, default=64)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--log_file', type=str, default='log/{}/{}_pretrain.log'.format(date, hour))
    parser.add_argument('--save_model', type=str, default='checkpoints/{}/{}_pretrain.pt'.format(date, hour))
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--load_cache', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--negative_sample', type=bool, default=False)
    args = parser.parse_args()
    return args



def train(model, train_dataset, valid_dataset, args):
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    criterion = F.binary_cross_entropy_with_logits
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    training_steps = args.epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, training_steps)

    log_interval = 500
    start_time = time.time()
    best_valid_acc = float('-inf')

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()
        for idx, (x, x_demo, treatment,_, y, death, mask) in enumerate(train_dataloader):
            # set gradients to zero
            optimizer.zero_grad()
            # TODO: forward pass through model
            _, _, _, ps_output, _ = model(x, y, x_demo, treatment, teacher_forcing_ratio=1)

            # TODO: calculate loss
            treatment = treatment[:, args.max_stay:args.max_stay + args.pre_window]
            loss = criterion(ps_output.reshape(-1), treatment.reshape(-1))

            # send the loss backwards to compute deltas
            loss.backward()
            # do not make the gradients too large
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            # run the optimizer
            optimizer.step()
            scheduler.step()

            if idx % log_interval == 0:
                elapsed = time.time() - start_time
                logging.info('| epoch {:3d} | {:5d}/{:5d} batches '
                             '| loss {:8.3f} '.format(epoch, idx, len(train_dataloader),
                                                      loss))
                start_time = time.time()

        # TODO: evaluation
        valid_acc = evaluate_simple(model, valid_dataloader, args)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            logging.info('Best model. Saving...\n')
            torch.save(model, args.save_model)

        logging.info('-' * 59)
        logging.info('| end of epoch {:3d} | time: {:5.2f}s | '
                     'valid acc {:8.3f} '.format(epoch,
                                                      time.time() - epoch_start_time,
                                                      valid_acc))
        logging.info('-' * 59)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def evaluate_simple(model, dataloader, args):
    model.eval()
    t_true_all, t_pred_all=[],[]
    with torch.no_grad():
        for x, x_demo, treatment,_, y, death, mask in dataloader:
            _, _, _, ps_output, _ = model(x, y, x_demo, treatment, teacher_forcing_ratio=1)

            treatment = treatment[:, args.max_stay:args.max_stay + args.pre_window]
            t_true = treatment.reshape(-1).to('cpu').detach().data.numpy()
            t_pred = ps_output.reshape(-1).to('cpu').detach().data.numpy()
            t_true_all.append(t_true)
            t_pred_all.append(t_pred)

    t_true_all = np.array(np.concatenate(t_true_all))
    t_pred_all = np.array(np.concatenate(t_pred_all))

    total_auc = roc_auc_score(t_true_all, t_pred_all)

    return total_auc


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()

    set_seed(seed=args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        filename=args.log_file,
        filemode='w',
        datefmt='%m/%d/%Y %I:%M:%S')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, dataset = load_and_process_data(args, device, logging, dataset_name='mimic3')

    num_train = int(len(dataset) * 0.7)
    num_val = int(len(dataset) * 0.1)
    split_train_, split_valid_, split_test_ = \
        random_split(dataset, [num_train, num_val, len(dataset) - num_train - num_val])

    enc = Encoder(
        input_dim=len(features),
        output_dim=1,
        x_static_size=4,
        emb_dim=args.emb_size,
        hid_dim=args.hid_size,
        n_layers=2,
        dropout=args.dropout_rate,
        device=device)
    dec = AttentionDecoder(
        output_dim=1,
        x_static_size=4,
        emb_dim=args.emb_size,
        hid_dim=args.hid_size,
        n_layers=2,
        dropout=args.dropout_rate)

    model = Seq2Seq(enc, dec, device).to(device)

    ##training
    train(model, split_train_, split_valid_, args)
    test_dataloader = DataLoader(split_test_, batch_size=args.batch_size, shuffle=True)
    ##testing
    test_metric = evaluate_simple(model, test_dataloader, args)
    logging.info('test metric {:8.3f}'.format(test_metric))




if __name__ == '__main__':
    main()
