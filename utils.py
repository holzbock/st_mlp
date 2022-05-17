import os

import torch
import numpy as np

import math as m


def check_cuda(device):
    # Check if cuda is available
    if 'cuda' in device:
        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            device_number = int(device.split(':')[-1])
            if device_number >= num_devices:
                raise ValueError('You have only %i cuda devices but set cuda device %i.'%(num_devices, device_number))
        else:
            raise ValueError('You set a cuda device but cuda is not available.')


def experiment_mkdir(opt):
    # Generate a new folder for each run with the log files
    dirs = [x[0] for x in os.walk(opt.ouput_path)]
    log_dir = opt.ouput_path + 'exp_0'

    if len(dirs) == 1:
        os.mkdir(log_dir)
    else:
        num = 0
        for d in dirs:
            d = d.split('_')
            if d[-1] != opt.ouput_path and int(d[-1]) > num:
                num = int(d[-1])
        log_dir = opt.ouput_path + 'exp_' + str(num+1)
        os.makedirs(log_dir, exist_ok=True)

    return log_dir


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flat_and_anneal_lr_scheduler(lrf, anneal_factor, total_iters):
    anneal_point = total_iters * anneal_factor
    def f(x):
        if x < anneal_point:
            return 1 # No lr factor is needed
        else:
            return lrf + 0.5 * (1 - lrf) * (1 + m.cos(m.pi * ((float(x) - anneal_point) / (total_iters - anneal_point))))

    return f


def pad_sequence(sequence, max_len):
    # This function is used from the TCG dataset: 
    #   https://github.com/againerju/tcg_recognition/blob/a5470aae9bef2facbe75358fe5c62e37dc547154/utils.py#L539

    # validity check
    assert sequence.shape[0] <= max_len

    # zero padding
    pad_seq = np.zeros((max_len, sequence.shape[1]))
    pad_seq[0:sequence.shape[0], :] = sequence

    return pad_seq


def one_hot_encoding(targets, nb_classes):
    # This function is used from the TCG dataset: 
    #   https://github.com/againerju/tcg_recognition/blob/a5470aae9bef2facbe75358fe5c62e37dc547154/utils.py#L561
    
    targets_one_hot = np.eye(nb_classes)[targets]

    return targets_one_hot


def subsampling(sequence, sampling_factor=5):
    # This function is used from the TCG dataset: 
    #   https://github.com/againerju/tcg_recognition/blob/a5470aae9bef2facbe75358fe5c62e37dc547154/utils.py#L577

    sequence = sequence[::sampling_factor]

    return sequence
