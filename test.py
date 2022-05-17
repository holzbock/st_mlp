import os

import torch
import numpy as np

import argparse
import yaml
from tqdm import tqdm
import time
from sklearn import metrics
import pickle

from mlp_mixer import MlpMixer
from dataset_aad import AadDataset
from dataset_tcg import TcgDataset
from utils import check_cuda


# Balanced Accuracy from Act and Drive dataset
def balanced_accuracy(true_labels, pred_labels):
    def accuracy(true_labels, pred_labels):
        accuracy = sum(np.asarray(true_labels) == np.asarray(pred_labels)) / float(len(true_labels))
        return (accuracy)
    class_list = list(set(true_labels))
    total = 0.0
    for curr_class in class_list:
        curr_inds = np.where(true_labels == curr_class)
        total += accuracy(true_labels[curr_inds], pred_labels[curr_inds])
    balanced_accuracy = total/len(class_list)
    return (balanced_accuracy)


def overallAccurayAAD(gt_aad, pred_aad):
    gt = np.concatenate(gt_aad)
    pred = np.concatenate(pred_aad)
    # https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
    cm = metrics.confusion_matrix(gt, pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Confusion Matrix: row = ground truth; col = prediction
    mpc_accuracy = np.nanmean(cm_normalized.diagonal())

    return mpc_accuracy


def eval_TCG(model, val_dataloader, CE_loss, opt, epoch):

    pred_y = list()
    gt_y = list()
    val_loss = 0
    mean_time = 0

    nb = len(val_dataloader)
    model.eval()
    for i, (x, y, padded) in tqdm(enumerate(val_dataloader), total=nb):
        x = x.to(opt.device)
        y = y.to(opt.device)
        padded = padded.to(opt.device)
        with torch.no_grad():
            start = time.time()
            pred = model(x, padded)
            end = time.time()
            loss = CE_loss(pred, y)
            pred = pred.softmax(dim=1)

        val_loss = (i*val_loss + loss) / (i + 1) 
        
        pred = pred.argmax(dim=1)
        
        pred_y.append(pred.detach())
        gt_y.append(y)
        t = end - start
        mean_time = (mean_time * i + t) / (i+1)

    # Compute the validation scores: accuracy, f1 score and jaccard index
    pred_y = torch.cat(pred_y).cpu().numpy()
    gt_y = torch.cat(gt_y).cpu().numpy()
    f1 = metrics.f1_score(gt_y, pred_y, average='macro')
    jaccard = metrics.jaccard_score(gt_y, pred_y, average='macro')
    accuracy = metrics.accuracy_score(gt_y, pred_y)
    matrix = metrics.confusion_matrix(gt_y, pred_y)
    class_accuracy = matrix.diagonal()/matrix.sum(axis=1)
    cm = metrics.confusion_matrix(gt_y, pred_y)

    print('Accuracy: %f, F1-Score: %f, Jaccard Index:%f'%(accuracy, f1, jaccard))
    print("Mean inference time: %f ms \n"%(mean_time * 1000))

    return accuracy, f1, jaccard, val_loss, cm, class_accuracy


def eval_AAD(model, val_dataloader, CE_loss, opt, epoch):

    pred_y = list()
    gt_y = list()
    val_loss = 0
    mean_time = 0

    nb = len(val_dataloader)
    model.eval()
    for i, (x, y, padded) in tqdm(enumerate(val_dataloader), total=nb):
        x = x.to(opt.device)
        y = y.to(opt.device)
        padded = padded.to(opt.device)
        with torch.no_grad():
            start = time.time()
            pred = model(x, padded)
            end = time.time()
            loss = CE_loss(pred, y)
            pred = pred.softmax(dim=1)

        val_loss = (i*val_loss + loss) / (i + 1) 

        pred = pred.argmax(dim=1)
        pred_y.append(pred.detach().cpu().numpy())
        gt_y.append(y.cpu().numpy())
        
        t = end - start
        mean_time = (mean_time * i + t) / (i+1)

    # Compute the validation scores: accuracy, f1 score and jaccard index
    pred_y = np.concatenate(pred_y)
    gt_y = np.concatenate(gt_y)
    # https://stackoverflow.com/questions/39770376/scikit-learn-get-accuracy-scores-for-each-class
    cm = metrics.confusion_matrix(gt_y, pred_y)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Confusion Matrix: row = ground truth; col = prediction
    mpc_accuracy = np.nanmean(cm_normalized.diagonal())

    print("Mean per class accuracy: ", mpc_accuracy)
    print("val loss: ", val_loss.cpu().item())
    print("Mean inference time: %f ms \n"%(mean_time * 1000))

    return mpc_accuracy, val_loss, gt_y, pred_y, cm


def evaluate(opt):
    CE_loss = torch.nn.CrossEntropyLoss().to(opt.device)
    if opt.dataset == 'AAD':
        dataset = AadDataset(opt.data_path, opt.model, mode=opt.eval_mode_aad, split=0, label_type=opt.label_type_aad, 
                        seq_len=opt.mlp_seq_len, num_classes=opt.num_classes, convert_coord=opt.convert_coord)
        eval_runs = 3
    elif opt.dataset == 'TCG':
        dataset = TcgDataset(opt.data_path, opt.subclasses, opt.protocol, train=False, seq_len=opt.mlp_seq_len, 
                        num_classes=opt.num_classes, f=opt.f_step)
        eval_runs = dataset.getNumberRuns()
    else:
        raise ValueError('Unknown dataset: %f'%opt.dataset)

    gt_aad = [np.array([]), np.array([]), np.array([])]
    pred_aad = [np.array([]), np.array([]), np.array([])]
    accuracy_all = list()
    f1_all = list()
    jaccard_all = list()
    cm_all = list()
    class_accuracy_all = list()

    for run_id in range(eval_runs):
        print('Evaluate split %i:'%run_id)

        # Get dataloader
        if opt.dataset == 'AAD':
            dataset.setSplit(run_id)
            dataset.loadData()
        elif opt.dataset == 'TCG':
            dataset.setRunID(run_id)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, pin_memory=True)

        # Get model
        model = MlpMixer(num_classes=opt.num_classes, num_blocks=opt.mlp_num_blocks, hidden_dim=opt.mlp_hidden_dim, 
                    tokens_mlp_dim=opt.tokens_mlp_dim, channels_mlp_dim=opt.channels_mlp_dim, seq_len=opt.mlp_seq_len, 
                    activation=opt.activation, mlp_block_type=opt.mlp_block_type, regularization=opt.regularization, 
                    input_size=opt.input_size, r_se=opt.r_se, use_max_pooling=opt.se_use_max_pooling, use_se=opt.use_se)
            
        name = 'best_performance' + str(run_id) + '.pt' 
        sd_path = os.path.join(opt.data, name)
        state_dict = torch.load(sd_path)
        model.load_state_dict(state_dict)
        model = model.to(opt.device)

        # Evaluate the model
        if opt.dataset == 'AAD':
            mpc_accuracy, _, gt_y, pred_y, cm = eval_AAD(model, dataloader, CE_loss, opt, run_id)
            gt_aad[run_id] = gt_y
            pred_aad[run_id] = pred_y
        elif opt.dataset == 'TCG':
            accuracy, f1, jaccard, _, cm, class_accuracy = eval_TCG(model, dataloader, CE_loss, opt, run_id)
            accuracy_all.append(accuracy)
            f1_all.append(f1)
            jaccard_all.append(jaccard)
            cm_all.append(cm)
            class_accuracy_all.append(class_accuracy)

    if opt.dataset == 'AAD':
        overall_mean_accuracy = overallAccurayAAD(gt_aad, pred_aad)
        accuracy = (np.concatenate(gt_aad) == np.concatenate(pred_aad)).sum() / len(np.concatenate(pred_aad))
        out = '\nOverall %s results:\n Mean Class Accuracy: %f; Accuracy: %f'%(opt.eval_mode_aad, overall_mean_accuracy, accuracy)
    elif opt.dataset == 'TCG':
        class_accuracy_all_mean = np.stack(class_accuracy_all).mean(axis=0)
        class_accuracy_all_std = np.stack(class_accuracy_all).std(axis=0)
        out = '\nOverall %s results:\n Accuracy: Mean: %f, Std: %f;\n F1-Score: Mean: %f, Std: %f;\n Jaccard Index: Mean: %f, \
                Std: %f;\n Class Accuracy: Mean: %f, %f, %f, %f,\n Std: %f, %f, %f, %f'%(opt.protocol, np.array(accuracy_all).mean(), 
                np.array(accuracy_all).std(), np.array(f1_all).mean(), np.array(f1_all).std(), np.array(jaccard_all).mean(), 
                np.array(jaccard_all).std(), class_accuracy_all_mean[0], class_accuracy_all_mean[1], class_accuracy_all_mean[2], 
                class_accuracy_all_mean[3], class_accuracy_all_std[0], class_accuracy_all_std[1], class_accuracy_all_std[2], 
                class_accuracy_all_std[3])
        with open('./tmp/_cm_data.pkl', 'wb') as f:
            pickle.dump(cm_all, f)

    print(out)

    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./runs/exp_0', help='Path where the weights and opt are saved.')
    parser.add_argument('--device', type=str, default='cuda:0', help='device for the model')
    parser.add_argument('--eval_mode_aad', type=str, default='test', help='Choose if the val or the test data are used for the evaluation. Possible: val, test')
    parser.add_argument('--num_workers', type=int, default=4, help='num workers in the dataloader')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    opt = parser.parse_args()

    check_cuda(opt.device)

    config = os.path.join(opt.data, 'opt.yaml')
    with open(config, 'r') as f:
        config = yaml.safe_load(f)

    opt.data_path = config['data_path']
    opt.model = config['model']
    opt.dataset = config['dataset']
    opt.subclasses = config['subclasses']
    opt.protocol = config['protocol']
    opt.label_type_aad = config['label_type_aad']
    opt.mlp_seq_len = config['mlp_seq_len']
    opt.num_classes = config['num_classes']
    opt.convert_coord = config['convert_coord']
    opt.mlp_num_blocks = config['mlp_num_blocks']
    opt.mlp_hidden_dim = config['mlp_hidden_dim']
    opt.channels_mlp_dim = config['channels_mlp_dim']
    opt.tokens_mlp_dim = config['tokens_mlp_dim']
    opt.mlp_seq_len = config['mlp_seq_len']
    opt.input_size = config['input_size']
    opt.activation = config['activation']
    opt.mlp_block_type = config['mlp_block_type']
    opt.regularization = config['regularization']
    opt.r_se = config['r_se']
    opt.se_use_max_pooling = config['se_use_max_pooling']
    opt.use_se = config['use_se']
    opt.f_step = config['f_step']
    opt.conf_matr = True

    evaluate(opt)
