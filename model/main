import argparse
from datetime import datetime
import os
import pickle
import logging
import time
from sklearn.metrics.pairwise import euclidean_distances
import random

import numpy as np
import torch
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader,TensorDataset

from dataset import load_and_process_data
from seq2seq import Encoder,AttentionDecoder,Seq2Seq

timestamp = datetime.now().strftime('%Y-%m%d_%H%M%S')
date, hour = timestamp.split('_')
os.makedirs('log/{}'.format(date), exist_ok=True)
os.makedirs('checkpoints/{}'.format(date), exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--max_stay', type=int, default=16)
    parser.add_argument('--pre_window', type=int, default=7)
    parser.add_argument('--mortality_window', type=int, default=60)
    parser.add_argument('--aug_ratio', type=float, default=0.4)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--emb_size', type=int, default=32)
    parser.add_argument('--hid_size', type=int, default=64)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--log_file', type=str, default='log/{}/{}.log'.format(date, hour))
    parser.add_argument('--save_model', type=str, default='checkpoints/{}/{}.pt'.format(date, hour))
    # parser.add_argument('--pretrained_model', type=str, default='checkpoints/{}/{}_pretrain.pt'.format(date, hour))
    parser.add_argument('--pretrained_model', type=str, default='checkpoints/2021-0801/103302_7_pretrain.pt')
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--load_cache', type=bool, default=False)
    parser.add_argument('--output_amsterdamdb', type=str, default='results/{}/{}.csv'.format(date, hour))
    parser.add_argument('--output_mimic', type=str, default='results/{}/{}.csv'.format(date, hour))
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--negative_sample', type=bool, default=False)

    args = parser.parse_args()
    return args


def train(model, train_dataset, valid_dataset, train_data, ps_estimator, args):
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion=torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-8)
    training_steps = args.epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, training_steps)

    log_interval = 500
    start_time = time.time()

    best_valid_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        loss_total = 0
        epoch_start_time = time.time()
        model.train()
        for idx, (x, x_demo, treatment,treatment_label, y, death, mask) in enumerate(train_dataloader):
            # set gradients to zero
            optimizer.zero_grad()

            # augment batch if aug_ratio is not zero
            if train_data:
                ps_train, t_label_train = train_data['ps_outputs'], train_data['t_label']
                nn_idx = get_aug_sample(x, y, x_demo, treatment, treatment_label, ps_estimator, ps_train, t_label_train,
                                        args)
                x = torch.cat((train_data['X'][nn_idx], x), dim=0)
                x_demo = torch.cat((train_data['X_static'][nn_idx], x_demo), dim=0)
                y = torch.cat((train_data['Y'][nn_idx], y), dim=0)
                treatment = torch.cat((train_data['treatment'][nn_idx], treatment), dim=0)

            # TODO: forward pass through model
            output, _, _, ps_output, _ = model(x, y, x_demo, treatment)
            # output_pred=output.argmax(-1)
            loss = criterion(output.reshape(-1), y[:, 1:].reshape(-1))

            # loss = criterion(output.reshape(-1, output.shape[-1]), y[:,1:].long().reshape(-1))
            loss_total += loss.item()
            # send the loss backwards to compute deltas
            loss.backward()
            # do not make the gradients too large
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            # run the optimizer
            optimizer.step()
            scheduler.step()

        logging.info('| epoch {:3d} '
                     '| loss {:8.3f} '.format(epoch,loss_total/len(train_dataloader)))


        # evaluation with factual loss
        valid_loss, valid_acc = evaluate_simple(model, valid_dataloader)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            logging.info('Best model. Saving...\n')
            torch.save(model, args.save_model)

        logging.info('-' * 59)
        logging.info('| end of epoch {:3d} | time: {:5.2f}s | '
                     'valid loss {:8.3f} | valid acc {:8.3f} '.format(epoch,
                                                      time.time() - epoch_start_time,
                                                      valid_loss, valid_acc))
        logging.info('-' * 59)

        # # evaluation with nn PEHE
        # valid_loss = evaluate_nn(model, valid_dataloader, args)
        # if valid_loss < best_valid_loss:
        #     best_valid_loss = valid_loss
        #     logging.info('Best model. Saving...\n')
        #     torch.save(model, args.save_model)
        #
        # logging.info('-' * 59)
        # logging.info('| end of epoch {:3d} | time: {:5.2f}s | '
        #              'valid nn pehe {:8.3f} '.format(epoch,
        #                                           time.time() - epoch_start_time,
        #                                           valid_loss))
        # logging.info('-' * 59)

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


def batch_aug(dataset, ps_estimator, args, device):
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    ps_outputs = []
    X = []
    X_static = []
    treatments, treatment_labels = [], []
    Y = []
    label2treatment = treatment2label2(args.pre_window)
    labels = sorted(label2treatment.keys())
    ps_estimator.eval()
    with torch.no_grad():
        for x, x_demo, treatment,treatment_label, y, death, mask in dataloader:
            _, _, _, ps_output, _ = ps_estimator(x, y, x_demo, treatment)
            ps_output = torch.sigmoid(ps_output).to('cpu').detach().data.numpy()
            ps_output_all = np.zeros(shape=(ps_output.shape[0], len(label2treatment.keys()), ps_output.shape[1]))
            treatments.append(treatment)
            treatment_labels.append(treatment_label)

            for i in labels:
                treatment_list = np.tile(np.array([int(x) for x in label2treatment[i]]), (ps_output.shape[0], 1))
                ps_all = treatment_list * ps_output + (1 - treatment_list) * (1 - ps_output)
                ps_output_all[:, i] = ps_all

            ps_outputs.append(ps_output_all)
            X.append(x)
            X_static.append(x_demo)
            Y.append(y)

    ps_outputs = np.concatenate(ps_outputs, axis=0)
    ps_outputs = torch.from_numpy(ps_outputs.astype(np.float32)).to(device=device)
    X = torch.cat(X)
    X_static = torch.cat(X_static)
    treatments = torch.cat(treatments)
    treatment_labels = torch.cat(treatment_labels)
    Y = torch.cat(Y)

    train_data = {'X': X, 'X_static': X_static, 'treatment': treatments, 't_label': treatment_labels, 'Y': Y,
                  'ps_outputs': ps_outputs}

    return train_data

def get_aug_sample(x, y, x_demo, treatment, treatment_label, ps_estimator, ps_train, t_label_train, args):
    ps_estimator.eval()
    with torch.no_grad():
        _, _, _, ps_batch, _ = ps_estimator(x, y, x_demo, treatment)
    ps_batch = torch.sigmoid(ps_batch).to('cpu').detach().data.numpy()

    label2treatment = treatment2label2(args.pre_window)
    labels = sorted(label2treatment.keys())
    ps_output_all = np.zeros(shape=(ps_batch.shape[0], len(label2treatment.keys()), ps_batch.shape[1]))

    for i in labels:
        treatment_list = np.tile(np.array([int(x) for x in label2treatment[i]]), (ps_batch.shape[0], 1))
        ps_all = treatment_list * ps_batch + (1 - treatment_list) * (1 - ps_batch)
        ps_output_all[:, i] = ps_all

    nn = []
    patient_matched = np.random.choice(np.arange(len(ps_batch)), int(len(ps_batch) * args.aug_ratio))
    ps_output_all = ps_output_all[patient_matched]

    for i, ps in enumerate(ps_output_all):
        t = treatment_label[patient_matched[i]]
        for k in labels:
            if t != k:
                k_idx = torch.where(t_label_train == k)[0]
                if len(k_idx) == 0:
                    continue
                ps_train_k = ps_train[k_idx,k,:]
                ps_train_k = ps_train_k.to('cpu').detach().data.numpy()
                pw_dist = euclidean_distances(ps_train_k, np.expand_dims(ps[k], axis=0))
                nn_k = np.argmin(pw_dist)
                nn.append(k_idx[nn_k].item())
    return np.array(nn)


def evaluate_simple(model, dataloader):
    model.eval()
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()
    total_loss = 0
    total_acc, total_count=0,1
    with torch.no_grad():
        for x, x_demo, treatment,_, y, death, mask in dataloader:
            output, output_cf, patient_rep, _, _ = model(x, y, x_demo, treatment, teacher_forcing_ratio=0)

            # loss = criterion(output.reshape(-1, output.shape[-1]), y[:,1:].long().reshape(-1))
            loss = criterion(output.reshape(-1), y[:, 1:].reshape(-1))
            total_loss += loss.item()

            # y_pred=output.argmax(-1)
            # y_true = y[:,1:]
            # total_acc += (y_pred==y_true).sum().item()
            # total_count += len(y_true.reshape(-1))


    total_loss = total_loss / len(dataloader)

    return total_loss, total_acc/total_count


def evaluate_nn(model, dataloader, args):
    model.eval()
    patient_representatios = []
    ITE_pred = []
    treatments = []
    y_true = []
    with torch.no_grad():
        for x, x_demo, treatment, _, y, death, mask in dataloader:
            output, output_cf, patient_rep, _, _ = model(x, y, x_demo, treatment, teacher_forcing_ratio=0)
            # output_pred, output_cf_pred = output.argmax(-1), output_cf.argmax(-1)
            output_pred, output_cf_pred = output, output_cf
            treatment = treatment[:, args.max_stay:]
            output_1 = output_pred * treatment + output_cf_pred * (1 - treatment)
            output_0 = output_cf_pred * treatment + output_pred * (1 - treatment)
            ite_pred = (output_1 - output_0)

            patient_representatios.append(patient_rep.to('cpu').detach().data.numpy())
            ITE_pred.append(ite_pred.to('cpu').detach().data.numpy())
            treatments.append(treatment.to('cpu').detach().data.numpy())
            y_true.append(y[:,1:].to('cpu').detach().data.numpy())

    patient_representatios = np.concatenate(patient_representatios)
    ITE_pred = np.concatenate(ITE_pred)
    treatments = np.concatenate(treatments)
    y_true = np.concatenate(y_true)

    pehe=[]
    for i in range(len(ITE_pred)):
        for j in range(len(ITE_pred[0])):
            nn_idx=np.where(treatments[:,j]==(1-treatments[i,j]))[0]
            nn_pat = patient_representatios[nn_idx]
            cur_pat=patient_representatios[i]
            pw_dist = euclidean_distances(nn_pat, np.expand_dims(cur_pat, axis=0))
            min_idx = np.argmin(pw_dist)
            y_cf = y_true[nn_idx[min_idx],j]
            y_f=y_true[i,j]
            output_1_true=y_f*treatments[i,j]+y_cf*(1-treatments[i,j])
            output_0_true = y_cf * treatments[i, j] + y_f * (1 - treatments[i, j])
            ite_true=output_1_true-output_0_true
            pehe.append((ite_true-ITE_pred[i,j])**2)

    return sum(pehe)/len(pehe)

def _inference(model, dataloader, args):
    model.train()
    outputs, outputs_cf, targets = [], [], []
    treatments = []
    deaths = []
    patient_representatios = []
    with torch.no_grad():
        for x, x_demo, treatment, _, y, death, mask in dataloader:
            output, output_cf, patient_rep, _, _ = model(x, y, x_demo, treatment, teacher_forcing_ratio=0)

            # y_pred = (torch.argmax(output, -1))
            # y_cf_pred = (torch.argmax(output_cf, -1))

            y_pred = output
            y_cf_pred = output_cf

            outputs.append(y_pred.to('cpu').detach().data.numpy())
            outputs_cf.append(y_cf_pred.to('cpu').detach().data.numpy())

            treatments.append(treatment[:, args.max_stay:].to('cpu').detach().data.numpy())
            deaths.append(death.to('cpu').detach().data.numpy())
            patient_representatios.append(patient_rep.to('cpu').detach().data.numpy())

    outputs, outputs_cf = np.concatenate(outputs), np.concatenate(outputs_cf)
    treatments = np.concatenate(treatments)
    deaths = np.concatenate(deaths)
    patient_representatios = np.concatenate(patient_representatios)

    return outputs, outputs_cf, treatments, deaths, patient_representatios


def evaluate(model, dataloader, args, mortality_window):
    ite_pred = []
    pat_rep_all = []
    for seed in (100, 110):
        set_seed(seed)
        outputs, outputs_cf, treatments, deaths, patient_representatios = _inference(model, dataloader, args)
        output_1 = outputs * treatments + outputs_cf * (1 - treatments)
        output_0 = outputs_cf * treatments + outputs * (1 - treatments)
        pat_rep_all.append(patient_representatios)
        ite_pred.append(output_1-output_0)

    ite_pred = np.array(ite_pred)
    pat_rep_all = np.array(pat_rep_all)

    ite_pred_mean = np.average(ite_pred, axis=0)
    ite_pred_var = np.var(ite_pred, axis=0)
    pat_rep_all = np.average(pat_rep_all, axis=0)

    deaths = np.where(deaths < mortality_window, 1, 0)

    treatment_pre = ((ite_pred_mean+ite_pred_var) < 0) * 1

    diff = (treatments == treatment_pre) * 1
    diff_idx = np.where(np.sum(diff, axis=-1) < args.pre_window)[0]
    same_idx = np.where(np.sum(diff, axis=-1) == args.pre_window)[0]

    treatments_diff = treatments[diff_idx]
    treatment_pre_diff = treatment_pre[diff_idx]

    death_compare = []
    include_idx = []
    for i in range(len(treatments_diff)):
        treatment_pre = treatment_pre_diff[i]
        tmp = np.sum(np.where(((treatments[same_idx] - treatment_pre) == 0), 1, 0), axis=-1)
        compare_idx = np.where(tmp == args.pre_window)
        if len(compare_idx[0]) == 0:
            continue
        include_idx.append(i)
        cur_rep = pat_rep_all[diff_idx[i]]
        compare_rep = pat_rep_all[same_idx[compare_idx]]
        pw_dist = euclidean_distances(compare_rep, np.expand_dims(cur_rep, axis=0))
        nn_idx = np.argmin(pw_dist)
        death_compare.append(deaths[same_idx[compare_idx][nn_idx]])

    diff_idx = diff_idx[np.array(include_idx)]
    deaths_rate_diff = sum(deaths[diff_idx]) / len(deaths[diff_idx])
    deaths_rate = sum(death_compare) / len(death_compare)

    deaths_rate_all = sum(deaths) / len(deaths)

    return deaths_rate_diff,deaths_rate, len(death_compare),deaths_rate_all, len(deaths)


def treatment2label2(max_outcome):

    treatment2label=dict()
    label2treatment = dict()
    n_treatment = 2**max_outcome
    for i in range(n_treatment):
        bin = "{0:b}".format(i)
        if len(bin) < max_outcome:
            bin = '0'*(max_outcome-len(bin))+bin
        treatment2label[bin] = i
        label2treatment[i] = bin

    return label2treatment


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

    # mimic3 for train, validation and internal test
    features, mimic3 = load_and_process_data(args, device, logging, dataset_name='mimic3')

    # amsterdamdb for external test
    _, amsterdamdb = load_and_process_data(args, device, logging, dataset_name='amsterdamdb')

    num_train = int(len(mimic3) * 0.7)
    num_val = int(len(mimic3) * 0.1)
    split_train_, split_valid_, split_test_ = \
        random_split(mimic3, [num_train, num_val, len(mimic3) - num_train - num_val])

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

    if args.aug_ratio and os.path.exists(args.pretrained_model):
        logging.info("Ratio of aug. samples == > {}".format(args.aug_ratio))
        ps_estimator = torch.load(args.pretrained_model)
        train_data = batch_aug(split_train_, ps_estimator, args, device)
    else:
        train_data = None
        ps_estimator = None

    # training
    train(model, split_train_, split_valid_, train_data, ps_estimator, args)

    for mortality_window in [30, 60]:
        # testing on mimic3
        test_dataloader = DataLoader(split_test_, batch_size=args.batch_size, shuffle=False)
        test_metric = evaluate(model, test_dataloader, args, mortality_window)
        deaths_rate_diff, deaths_rate_same, n_diff, deaths_rate_all, n_patient = test_metric
        logging.info('Test MIMIC3 | Mortality window: {} | Diff. rate {:8.3f} | Same rate {:8.3f} |'
                     ' No. Diff. {} | Total rate {:8.3f} | No. patient {}'
                     .format(mortality_window, deaths_rate_diff, deaths_rate_same, n_diff, deaths_rate_all, n_patient))
        output_mimic = open(args.output_mimic + '_{}d_mimic.csv'.format(mortality_window), 'a')
        output_mimic.write('{:.3f},{:.3f},{},{:.3f},{}\n'
                     .format(deaths_rate_diff, deaths_rate_same, n_diff, deaths_rate_all, n_patient))

        # testing on amsterdamdb
        test_dataloader = DataLoader(amsterdamdb, batch_size=args.batch_size, shuffle=False)
        test_metric = evaluate(model, test_dataloader, args, mortality_window)
        deaths_rate_diff, deaths_rate_same, n_diff, deaths_rate_all, n_patient = test_metric
        logging.info('Test AmsterdamDB | Diff. rate {:8.3f} | Same rate {:8.3f} |'
                     ' No. Diff. {} | Total rate {:8.3f} | No. patient {}'
                     .format(deaths_rate_diff, deaths_rate_same, n_diff, deaths_rate_all, n_patient))
        output_amsterdamdb = open(args.output_amsterdamdb + '_{}d_amsterdamdb.csv'.format(mortality_window), 'a')
        output_amsterdamdb.write('{:.3f},{:.3f},{},{:.3f},{}\n'
                           .format(deaths_rate_diff, deaths_rate_same, n_diff, deaths_rate_all, n_patient))




if __name__ == '__main__':
    main()
