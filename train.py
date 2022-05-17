import numpy as np
import torch
import argparse
from tqdm import tqdm
import time
import yaml
from ranger import Ranger

import seaborn as sn
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from dataset_tcg import TcgDataset
from dataset_aad import AadDataset
from test import evaluate as evaluate_aad_on_test, eval_AAD, eval_TCG, overallAccurayAAD
from mlp_mixer import MlpMixer
from utils import experiment_mkdir, count_parameters, flat_and_anneal_lr_scheduler, check_cuda



def main(opt):
    # Create log dir
    log_dir = experiment_mkdir(opt)
    tb_writer = SummaryWriter(log_dir=log_dir)

    # Load dataset
    if opt.dataset == 'TCG':
        train_dataset = TcgDataset(opt.data_path, opt.subclasses, opt.protocol, train=True, seq_len=opt.mlp_seq_len, 
                                    num_classes=opt.num_classes, f=opt.f_step, augment=opt.augment_tcg)
        val_dataset = TcgDataset(opt.data_path, opt.subclasses, opt.protocol, train=False, seq_len=opt.mlp_seq_len, 
                                    num_classes=opt.num_classes, f=opt.f_step)
        runs = train_dataset.getNumberRuns()
        batch_size_eval = opt.batch_size
        collate_fn = None
    elif opt.dataset == 'AAD':
        train_dataset = AadDataset(opt.data_path, opt.model, mode='train', split=0, label_type=opt.label_type_aad, 
                                    seq_len=opt.mlp_seq_len, num_classes=opt.num_classes, convert_coord=opt.convert_coord, 
                                    noise=opt.noise_augment, noise_std=opt.noise_std)
        val_dataset = AadDataset(opt.data_path, opt.model, mode=opt.eval_mode_aad, split=0, label_type=opt.label_type_aad, 
                                    seq_len=opt.mlp_seq_len, num_classes=opt.num_classes, convert_coord=opt.convert_coord)
        runs = 3
        batch_size_eval = opt.batch_size
        collate_fn = None
    else:
        raise ValueError('Unknown dataset: %s'%opt.dataset)

    accuracy_all = list()
    f1_all = list()
    jaccard_all = list()
    performance_all = list()
    gt_aad = [np.array([]), np.array([]), np.array([])]
    pred_aad = [np.array([]), np.array([]), np.array([])]

    # iterate over the run id's
    # run id's contain different intersection settings or subjects for training 
    # and testing in the TCG and different samples in the AAD
    for run_id in range(runs):
        if opt.dataset == 'TCG':
            # set the run_id in the dataset
            train_dataset.setRunID(run_id)
            val_dataset.setRunID(run_id)
        elif opt.dataset == 'AAD':
            train_dataset.setSplit(run_id)
            val_dataset.setSplit(run_id)
            train_dataset.loadData()
            val_dataset.loadData()
        
        # Create dataloader
        loss_weights = train_dataset.getLossWeights()
        weight_sample = torch.tensor([loss_weights[i] for i in train_dataset.labels])
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_sample, len(train_dataset))

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, 
                                                    pin_memory=True, shuffle=False, sampler=sampler, drop_last=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_eval, num_workers=opt.num_workers, 
                                                        pin_memory=True, collate_fn=collate_fn)

        start_time = time.time()

        # Save hyperparameters to log dir
        with open(log_dir + '/opt.yaml', 'w') as f:
            yaml.dump(vars(opt), f, sort_keys=False)

        # Get model
        model = MlpMixer(num_classes=opt.num_classes, num_blocks=opt.mlp_num_blocks, hidden_dim=opt.mlp_hidden_dim, 
                        tokens_mlp_dim=opt.tokens_mlp_dim, channels_mlp_dim=opt.channels_mlp_dim, seq_len=opt.mlp_seq_len, 
                        activation=opt.activation, mlp_block_type=opt.mlp_block_type, regularization=opt.regularization, 
                        input_size=opt.input_size, r_se=opt.r_se, use_max_pooling=opt.se_use_max_pooling, use_se=opt.use_se)
        model = model.to(opt.device)
        
        params = [p for p in model.parameters() if p.requires_grad]
        if opt.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=opt.lr)
            if opt.lr_scheduler == 'cos':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min=opt.lr*opt.lrf)
            elif opt.lr_scheduler == 'multistep':
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.lr_steps, opt.lr_gamma)
            else:
                raise ValueError('Unknown learning rate scheduler: %s'%opt.lr_scheduler)
        
        elif opt.optimizer == 'ranger':
            optimizer = Ranger(params, lr=opt.lr)
            lambda_scheduler = flat_and_anneal_lr_scheduler(opt.lrf, opt.anneal_factor, opt.epochs)
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_scheduler)

        elif opt.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay, nesterov=opt.nesterov)
            if opt.lr_scheduler == 'cos':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min=opt.lr*opt.lrf)
            elif opt.lr_scheduler == 'multistep':
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, opt.lr_steps, opt.lr_gamma)
            else:
                raise ValueError('Unknown learning rate scheduler: %s'%opt.lr_scheduler)

        else:
            raise ValueError('Unknown optimizer type: %s'%opt.optimizer)

        CE_loss = torch.nn.CrossEntropyLoss().to(opt.device)

        results_file = log_dir + '/results%s.txt' % run_id

        if opt.dataset == 'TCG':
            with open(results_file, 'a') as f:
                f.write('%10.4s' * 7 % ('Epoch', 'Memory', 'Train Loss', 'Accuracy', 'F1 Score', 'Jaccard', 'Val Loss') + '\n')
        elif opt.dataset == 'AAD':
            with open(results_file, 'a') as f:
                f.write('%10.4s' * 5 % ('Epoch', 'Memory', 'Train Loss', 'Accuracy', 'Val Loss') + '\n')

        path_best_performance = log_dir + '/best_performance%s.pt' % run_id
        best_performance = 0
        best_accuracy = 0
        best_f1 = 0
        best_jaccard = 0

        mean_overall_loss = 0

        accumulate = max(round(opt.overall_batch_size / opt.batch_size), 1)

        num_params = count_parameters(model)
        print('Number of trainable parameters in the model: ', num_params)

        for epoch in range(opt.epochs):
            print("Start epoch {}".format(epoch))
            pred_y = list()
            gt_y = list()

            num_batches = len(train_dataloader)
            loss_train = 0
            print(('%10s' * 3)%('Epoch', 'GPU_Mem', 'Loss'))
            pbar = tqdm(enumerate(train_dataloader), total=num_batches)
            model.train()
            optimizer.zero_grad()
            for batch_it, (x, y, padded) in pbar:
                x = x.to(opt.device, non_blocking=True)
                y = y.to(opt.device, non_blocking=True)
                padded = padded.to(opt.device, non_blocking=True)

                y_pred = model(x, padded)
                loss = CE_loss(y_pred, y)

                loss.backward()
                if batch_it % accumulate == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                loss_train = (batch_it * loss_train + loss.detach().cpu()) / (batch_it + 1) 

                # log prediction and GT
                pred_y.append(y_pred.softmax(dim=1).argmax(dim=1).cpu())
                gt_y.append(y.cpu())

                # Write informations to terminal
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 1) % ('%g/%g' % (epoch, opt.epochs - 1), mem, loss_train)
                pbar.set_description(s)

            # Calculate the train metrics
            pred_y = torch.cat(pred_y).numpy()
            gt_y = torch.cat(gt_y).numpy()
            accuracy_train = metrics.accuracy_score(gt_y, pred_y)
            f1_train = metrics.f1_score(gt_y, pred_y, average='macro')
            jaccard_train = metrics.jaccard_score(gt_y, pred_y, average='macro')
            cm_train = metrics.confusion_matrix(gt_y, pred_y)

            print("Evaluate the model after epoch {}".format(epoch))
            if opt.dataset == 'TCG':
                accuracy_val, f1_val, jaccard_val, val_loss, cm_val, class_accuracy = eval_TCG(model, val_dataloader, CE_loss, opt, epoch)
                with open(results_file, 'a') as f:
                    f.write(s + '%10.4g' * 4 % (accuracy_val, f1_val, jaccard_val, val_loss) + '\n')
            elif opt.dataset == 'AAD':
                accuracy_val, val_loss, gt_y, pred_y, cm_val = eval_AAD(model, val_dataloader, CE_loss, opt, epoch)
                with open(results_file, 'a') as f:
                    f.write(s + '%10.4g' * 2 % (accuracy_val, val_loss) + '\n')
            else:
                raise ValueError('Unknown dataset: %'%opt.dataset)

            mean_overall_loss = (epoch * mean_overall_loss + val_loss) / (epoch + 1) 
            lr_scheduler.step()
            lr = optimizer.param_groups[0]['lr']

            tags = ['train_loss/split_%i'%run_id, 'val_loss/split_%i'%run_id, 'val_accuracy/split_%i'%run_id, 
                    'f1_score_val/split_%i'%run_id, 'jaccard_index_val/split_%i'%run_id, 'learning_rate/split_%i'%run_id, 
                    'mean_loss/split_%i'%run_id, 'confusion_matrix_train/split_%i'%run_id, 
                    'confusion_matrix_val/split_%i'%run_id, 'train_accuracy/split_%i'%run_id, 
                    'f1_score_train/split_%i'%run_id, 'jaccard_index_train/split_%i'%run_id]
            tb_writer.add_scalar(tags[0], loss_train, epoch)
            tb_writer.add_scalar(tags[1], val_loss, epoch)
            tb_writer.add_scalar(tags[2], accuracy_val, epoch)
            if opt.dataset == 'TCG':
                tb_writer.add_scalar(tags[3], f1_val, epoch)
                tb_writer.add_scalar(tags[4], jaccard_val, epoch)
                tb_writer.add_scalar(tags[10], f1_train, epoch)
                tb_writer.add_scalar(tags[11], jaccard_train, epoch)
            tb_writer.add_scalar(tags[5], lr, epoch)
            tb_writer.add_scalar(tags[6], mean_overall_loss, epoch)
            tb_writer.add_scalar(tags[9], accuracy_train, epoch)

            plt_size = 40
            fig, ax = plt.subplots(figsize=(plt_size,plt_size)) 
            df_cm = pd.DataFrame(cm_train)
            fig = sn.heatmap(df_cm, annot=True, cbar=False, cmap='Blues', xticklabels=train_dataset.label_names, 
                                yticklabels=train_dataset.label_names) 
            tb_writer.add_figure(tags[7], fig.get_figure(), epoch)

            fig, ax = plt.subplots(figsize=(plt_size,plt_size)) 
            df_cm = pd.DataFrame(cm_val)
            fig = sn.heatmap(df_cm, annot=True, cbar=False, cmap='Blues', xticklabels=train_dataset.label_names, 
                                yticklabels=train_dataset.label_names)
            tb_writer.add_figure(tags[8], fig.get_figure(), epoch)

            if opt.dataset == 'TCG':
                performance = (accuracy_val + f1_val + jaccard_val) / 3
            elif opt.dataset == 'AAD':
                performance = accuracy_val

            if performance > best_performance:
                best_performance = performance
                best_accuracy = accuracy_val
                if opt.dataset == 'TCG':
                    best_f1 = f1_val
                    best_jaccard = jaccard_val
                if opt.dataset == 'AAD':
                    gt_aad[run_id] = gt_y
                    pred_aad[run_id] = pred_y
                torch.save(model.state_dict(), path_best_performance)
        
        end_time = time.time()

        # Write end informations to terminal
        if opt.dataset == 'TCG':
            out = 'Best Model Performance: %g Accuracy: %g; F1-Score: %g; Jaccard Index: %g' % (best_performance, 
                    best_accuracy, best_f1, best_jaccard)
            print(out)
        elif opt.dataset == 'AAD':
            out = 'Best Model Performance: %g Mean Class Accuracy: %g' % (best_performance, best_accuracy)
            print(out)
        trainings_time_out = "Training with %g epochs finished in %g h." % (opt.epochs, (end_time-start_time) / 3600)
        print(trainings_time_out)

        # Write end informations to file
        with open(results_file, 'a') as f:
            f.write(out + '\n')
            f.write(trainings_time_out + '\n')

        accuracy_all.append(best_accuracy)
        performance_all.append(best_performance)
        if opt.dataset == 'TCG':
            f1_all.append(best_f1)
            jaccard_all.append(best_jaccard)


    accuracy_all = np.array(accuracy_all).mean()
    performance_all = np.array(performance_all).mean()
    results_file = log_dir + '/overall_results.txt'


    # Evaluate on the test set
    if opt.dataset == 'AAD':
        print('\nReults on the test set:')
        opt.eval_mode_aad = 'test'
        opt.data = log_dir
        out = evaluate_aad_on_test(opt)
        with open(results_file, 'a') as f:
            f.write(out)

    if opt.dataset == 'TCG':
        f1_all = np.array(f1_all).mean()
        jaccard_all = np.array(jaccard_all).mean()
        out = '\nResults on the val set: \nOverall results:\n Accuracy: %f, F1-Score: %f, Jaccard Index: %f, Performance: %f'%(
                accuracy_all, f1_all, jaccard_all, performance_all)
        with open(results_file, 'a') as f:
            f.write(out)
        print(out)
    elif opt.dataset == 'AAD':
        overall_mean_accuracy = overallAccurayAAD(gt_aad, pred_aad)
        out = '\nResults on the val set: \nOverall results:\n Mean Class Accuracy: %f, Accuracy: %f'%(overall_mean_accuracy, 
                performance_all)
        with open(results_file, 'a') as f:
            f.write(out)
        print(out)


if __name__ == "__main__":
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument('--ouput_path', type=str, default='./runs/', help='Define the output path where the results are saved')
    parser.add_argument('--data_path', type=str, default='./tcg/', help='path to the dataset; ./tcg/ or ./act_and_drive/')
    parser.add_argument('--dataset', type=str, default='TCG', choices=['TCG', 'AAD'], help='Choose between TCG, AAD')
    parser.add_argument('--subclasses', type=int, default=0, help='if 1 uses the subclasses else only the major classes')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp'], help='Defines the model type.')
    parser.add_argument('--protocol', type=str, default='xv', help='which protocol for the evaluation is used cross view (xv) or cross subject (xs)')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size on the gpu for the training.')
    parser.add_argument('--overall_batch_size', type=int, default=2048, help='batch size after which the parameters are updated.')
    parser.add_argument('--epochs', type=int, default=70, help='number of training epochs')
    parser.add_argument('--optimizer', type=str, default='adam', help='Posible optimizers are: adam, ranger, sgd')
    parser.add_argument('--lr_scheduler', type=str, default='cos', choices=['cos', 'multistep'], help='Posible lr scheduler for SGD and Adam Optimizer.')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for the training')
    parser.add_argument('--lr_steps', type=str, default='50,', help='epochs to reduce the lr by the factor of gamma; !!!No space at the end!!!')
    parser.add_argument('--lr_gamma', type=float, default=0.2, help='factor to redcue the lr')
    parser.add_argument('--lrf', type=float, default=0.1, help='cosine annealing trainings_lr = lr * lrf')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum of the SGD optimizer')
    parser.add_argument('--weight_decay', type=str, default=0.0005, help='Weight Decay for the SGD optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=True, help='Nesterov for the SGD optimizer')
    parser.add_argument('--num_classes', type=int, default=4, help='Number action classes; TCG: 4, AAD: 34')
    parser.add_argument('--input_size', type=int, default=51, help='input size of the skeletons (17 joints in 3D); TCG: 51, AAD: 39. Automatically adapted for velocity and acceleration training.')
    parser.add_argument('--mlp_seq_len', type=int, default=8, help='Sequence lenght for the mlp')
    parser.add_argument('--mlp_num_blocks', type=int, default=1, help='MLP blocks used in the mlp-mixture model')
    parser.add_argument('--mlp_hidden_dim', type=int, default=128, help='Hidden dim of the mlp-mixture model')
    parser.add_argument('--tokens_mlp_dim', type=int, default=30, help='hidden dim in the mlp block for the sequence mlp')
    parser.add_argument('--channels_mlp_dim', type=int, default=512, help='hidden dim in the mlp block for the channel mlp')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use. cuda:? or cpu')
    parser.add_argument('--runtime', action='store_true', help='defines if a runtime evaluation is done')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers in the dataloader.')
    parser.add_argument('--anneal_factor', type=float, default=0.72, help='start of the cos lr decay epochs * anneal_factor > actual_epoch')
    parser.add_argument('--activation', type=str, default='gelu', help='Choose type of activation function. Possible: gelu and mish')
    parser.add_argument('--mlp_block_type', type=str, default='normal', choices=['normal', 'temporal', 'spatial'], help='Choose which MLP-block should be used.')
    parser.add_argument('--regularization', type=float, default=0.1, help='Choose regularisation method. -1 -> BatchNormalization; 0 -> No regularisation; 0.001 - 1 -> Dropout probability')
    parser.add_argument('--label_type_aad', type=str, default='coarse', choices=['fine', 'coarse'], help='Choose if the fine or coarse labels are used. Adapt also num classes. fine: 34 classes, coarse: 12 classes')
    parser.add_argument('--eval_mode_aad', type=str, default='val', help='Choose if the val or the test data are used for the evaluation. Possible: val, test')
    parser.add_argument('--convert_coord', type=int, default=0, help='0: csv data; 1: normalized csv data; 2: world coord data; 3: normalized world coord data')
    parser.add_argument('--split_train', type=str, default='train0', help='Train split for the AAD dataset. Possible: train0, train1, train2')
    parser.add_argument('--split_test', type=str, default='test0', help='Test split for the AAD dataset. Possible: test0, test1, test2, val0, val1, val2')
    parser.add_argument('--f_step', type=int, default=20, help='sample frequency')
    parser.add_argument('--rec_weight', type=float, default=0.1, help='reconstruction loss weight')
    parser.add_argument('--noise_augment', type=int, default=0, help='Use the noise augmentation. Possible: 0 no augmentation, 1 only for the underrepresented classes, 2 for all classes')
    parser.add_argument('--noise_std', type=float, default=0.001, help='std deviation in the noise augmentation')
    parser.add_argument('--se_use_max_pooling', type=str2bool, default=False, help='if False use Average_pooling if true use Max_pooling')
    parser.add_argument('--r_se', type=int, default=4, help='reduction parameter in the se module')
    parser.add_argument('--use_se', type=str2bool, default=True, help='Use the se block in the mixer')
    parser.add_argument('--augment_tcg', type=str2bool, default=False, help='Use rotation in go gesture and flip in clear gesture as augmentation.')
    opt = parser.parse_args()

    opt.lr_steps = [int(val) for val in opt.lr_steps.split(',') if len(val)>0.0001]

    check_cuda(opt.device)

    main(opt)
