import os
import sys

import torch
import numpy as np

from tqdm import tqdm
import random
import math as m
import json

from utils import one_hot_encoding, subsampling, pad_sequence

TCG_F = 20

class TcgDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, subclasses, protocol, train=True, seq_len=8, num_classes=4, f=TCG_F, 
                augment=False):
        # Load TCG data
        self.protocol = protocol
        self.train = train
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.db = TCGDB(data_path)
        self.db.open()
        self.db.set_subclasses(train_subclasses=int(subclasses))
        print("Finished opening TCG dataset. Dont forget to set the RunID!!!")
        self.class_list = {'idle': 0, 'stop': 1, 'go': 2, 'clear': 3}
        self.label_names = ['idle', 'stop', 'go', 'clear']

        self.data = list()
        self.labels = list()
        self.f = f
        self.dt = 1 / self.f
        self.augment = augment
        self.left_side_idx = [3, 4, 5, 6, 7, 8]
        self.right_side_idx = [10, 11, 12, 13, 14, 15]
        self.num_joints = 17


    def getNumberRuns(self):
        return self.db.get_number_runs(protocol_type=self.protocol)


    def setRunID(self, run_id):
        # Call setRunID when you want to change the evaluation data
        X_train, Y_train, X_test, Y_test = self.db.get_train_test_data(run_id=run_id, protocol_type=self.protocol)

        self.data = list()
        self.labels = list()

        if self.train:
            x_data = X_train
            y_data = Y_train
        else:
            x_data = X_test
            y_data = Y_test

        if self.f != TCG_F:
            print("Resampling to f={:4.2f} Hz".format(self.f))
            import numpy as np
            from scipy.interpolate import interp1d
            interval = (x_data.shape[1] - 1) / TCG_F
            num_new_samples = int(np.floor(interval * self.f))
            x_cur = np.linspace(0.0, interval, x_data.shape[1])
            x_new = np.linspace(0.0, interval, num_new_samples)
            #print(x_cur.shape, y_data.shape)
            f1 = interp1d(x_cur, x_data, axis=1)
            f2 = interp1d(x_cur, y_data, axis=1)
            x_data = f1(x_new)
            y_data = f2(x_new)

        pbar = tqdm(enumerate(zip(x_data, y_data)), total=len(x_data))
        for batch_it, (x, y) in pbar:
            good = y.sum(1) > 0.0
            x = torch.from_numpy(x).float()[good]     # shape: seq_len, 51
            y = torch.from_numpy(y).float()[good]     # shape: seq_len, 4
            y = y.argmax(dim=1)
            for i in range(x.shape[0] - self.seq_len + 1):
                self.data.append(x[i:i+self.seq_len])
                self.labels.append(y[i+self.seq_len-1])

        print('Finished setting the RunID.')


    def __len__(self):
        if len(self.data) == 0:
            raise ValueError('Call setRunID to set the run ID.')
        return len(self.data)


    def __getitem__(self, index):
        if len(self.data) == 0:
            raise ValueError('Call setRunID to set the run ID.')
        
        data_out = self.data[index]
        label_out = self.labels[index]
        padded = torch.ones(self.seq_len)

        # Augment skeletons
        # Do this to get a more general model for the real world szenario
        if self.augment:
            # flip the clear gesture
            left_side_idx = [3, 4, 5, 6, 7, 8]
            right_side_idx = [10, 11, 12, 13, 14, 15]
            if label_out == 3:
                if random.random() > 0.5:
                    left_side = data_out.reshape(self.seq_len, 17, 3)[: ,left_side_idx, :].clone()
                    right_side = data_out.reshape(self.seq_len, 17, 3)[: ,right_side_idx, :].clone()
                    data_out = data_out.reshape(self.seq_len, 17, 3)
                    data_out[:, right_side_idx, :] = left_side
                    data_out[:, left_side_idx, :] = right_side
                    data_out = data_out.reshape(self.seq_len, 51)
            # rotate the start gesture by 10 degree
            if label_out == 2:
                if random.random() > 0.:
                    angle = random.random() * 20 - 10
                    R = torch.tensor([[m.cos(angle), -m.sin(angle), 0], [m.sin(angle), m.cos(angle), 0], [0, 0, 1]])
                    data_out = (R @ data_out.reshape(self.seq_len,17,3).transpose(1,2)).transpose(1,2).reshape(self.seq_len, 51)

        return data_out, label_out, padded


    def getLossWeights(self):
        # occurence of the single classes | weights (1 - num/all) | 1/(num/all)
        # 0 (idle): 201350  | 0,326 | 1,48
        # 1 (stop): 41515   | 0,86  | 7,2
        # 2 (go): 47674     | 0,84  | 6,7
        # 3 (clear): 8375   | 0,972 | 35,69
        # all: 298914
        # CE_loss = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1.48, 7.2, 6.7, 35.69])).to(opt.device) 
        # Combines LogSoftmax and Cross Entropy
        weights = torch.zeros(self.num_classes)
        all = len(self.labels)
        labels = torch.stack(self.labels)

        for i in range(self.num_classes):
            num = (labels == i).sum().item()
            weights[i] = 1 / (num / all)

        return weights


class TCGDB(object):
    # This class is used from the TCG dataset: 
    #   https://github.com/againerju/tcg_recognition/blob/a5470aae9bef2facbe75358fe5c62e37dc547154/TCGDB.py#L8

    def __init__(self, root):

        self.root = root
        self.name = "tcg_dataset_2"

        self.path = os.path.join(root, self.name)

        self.db_file = os.path.join(self.path, "tcg.json")
        self.data_file = os.path.join(self.path, "tcg_data.npy")

        # database
        self.dataset = ""
        self.description = ""
        self.joint_dictionary = {}
        self.version = []
        self.sequences = []

        # learning protocols
        self.sampling_factor = 5  # to subsample 100 Hz to 20 Hz
        self.train_subclasses = False
        self.train_test_sets = {"xs": [[[1, 2, 3, 4], [5]],
                                       [[1, 3, 4, 5], [2]],
                                       [[1, 2, 4, 5], [3]],
                                       [[1, 2, 3, 5], [4]],
                                       [[2, 3, 4, 5], [1]]],
                                "xv": [[["right", "bottom", "left"], ["top"]],
                                       [["right", "bottom", "top"], ["left"]],
                                       [["bottom", "left", "top"], ["right"]],
                                       [["right", "left", "top"], ["bottom"]]]}

    def set_subclasses(self, train_subclasses=0):

        if train_subclasses == 0:
            self.train_subclasses = False
        elif train_subclasses == 1:
            self.train_subclasses = True
        else:
            sys.exit("Subclasses parameter has to be boolean (0/1), got {}. Abort!".format(train_subclasses))

    def get_nb_classes(self):

        if self.train_subclasses:
            return 15
        else:
            return 4

    def get_number_runs(self, protocol_type="xs"):

        return len(self.train_test_sets[protocol_type])

    def get_train_test_set(self, run_id=1, protocol_type="xs"):

        return self.train_test_sets[protocol_type][run_id]

    def get_train_test_data(self, run_id=1, protocol_type="xs"):

        # initialize train and test
        X_train = []
        Y_train = []

        X_test = []
        Y_test = []

        # sets
        train_sets = self.train_test_sets[protocol_type][run_id][0]
        test_sets = self.train_test_sets[protocol_type][run_id][1]


        # iterate through sequences
        for _, seq in enumerate(self.sequences):

            # poses & labels
            poses = np.array([f.pose.flatten() for f in seq.frames])

            if self.train_subclasses:
                targets = np.array([f.min_cls for f in seq.frames])
            else:
                targets = np.array([f.maj_cls for f in seq.frames])

            targets_one_hot = one_hot_encoding(targets, nb_classes=self.get_nb_classes())

            # subsampling
            poses = subsampling(poses, sampling_factor=self.sampling_factor)
            targets_one_hot = subsampling(targets_one_hot, sampling_factor=self.sampling_factor)

            # assign to set
            if protocol_type == "xs":

                if seq.subject in train_sets:

                    X_train.append(poses)
                    Y_train.append(targets_one_hot)

                elif seq.subject in test_sets:

                    X_test.append(poses)
                    Y_test.append(targets_one_hot)

                else:

                    print("Sequence is not contained in TRAIN neither in TEST...")

            elif protocol_type == "xv":

                if seq.viewpoint in train_sets:

                    X_train.append(poses)
                    Y_train.append(targets_one_hot)

                elif seq.viewpoint in test_sets:

                    X_test.append(poses)
                    Y_test.append(targets_one_hot)

                else:

                    print("Sequence is not contained in TRAIN neither in TEST...")

        # maximal sequence length
        max_seq_len_train = max([len(s) for s in X_train])
        max_seq_len_test = max([len(s) for s in X_test])
        max_seq_len = max([max_seq_len_train, max_seq_len_test])

        # zero padding
        X_train = np.array([pad_sequence(s, max_seq_len) for s in X_train])
        Y_train = np.array([pad_sequence(s, max_seq_len) for s in Y_train])
        X_test = np.array([pad_sequence(s, max_seq_len) for s in X_test])
        Y_test = np.array([pad_sequence(s, max_seq_len) for s in Y_test])

        return X_train, Y_train, X_test, Y_test

    def open(self):

        # load
        with open(self.db_file, "rb") as fin:
            db = json.load(fin)

        with open(self.data_file, "rb") as fin:
            data = np.load(fin, allow_pickle=True)

        # meta data
        self.dataset = db["dataset"]
        self.description = db["description"]
        self.version = db["version"]
        self.joint_dictionary = db["joint_dictionary"]

        # sequences
        for sid, seq in enumerate(db["sequences"]):

            sequence = TCGSequence()

            # sequence description
            sequence.subject = seq["subject_id"]
            sequence.junction = seq["junction"]
            sequence.scene = seq["scene_id"]

            sequence.scene_agents = seq["scene_agents"]

            agent_description = sequence.scene_agents[seq["agent_number"]]
            sequence.id = agent_description["id"]
            sequence.viewpoint = agent_description["position"]
            sequence.intention = agent_description["intention"]
            sequence.queue = agent_description["queue"]

            sequence.annotation = seq["annotation"]

            sequence.num_frames = seq["num_frames"]
            sequence.frames = []

            # annotations
            maj_class_name = [None] * sequence.num_frames
            min_class_name = [None] * sequence.num_frames

            for li, label_interval in enumerate(sequence.annotation):
                for f in range(label_interval[2], label_interval[3]):
                    maj_class_name[f] = label_interval[0]
                    min_class_name[f] = label_interval[1]

            # create frame instance
            pose_sequence = data[sid]
            for p, pose in enumerate(pose_sequence):
                frame = TCGFrame()
                frame.pose = pose
                frame.maj_cls_name = maj_class_name[p]
                frame.min_cls_name = min_class_name[p]
                frame.encode_majlabel()
                frame.encode_minlabel()
                sequence.frames.append(frame)

            # append
            self.sequences.append(sequence)


class TCGSequence(object):
    # This class is used from the TCG dataset: 
    #   https://github.com/againerju/tcg_recognition/blob/a5470aae9bef2facbe75358fe5c62e37dc547154/TCGDB.py#L203

    def __init__(self):

        self.subject = int()

        self.junction = ""
        self.scene = int()
        self.id = int()
        self.viewpoint = int()
        self.intention = int()
        self.queue = int()

        self.annotation = []

        self.num_frames = int()
        self.frames = []


class TCGFrame(object):
    # This class is used from the TCG dataset: 
    #   https://github.com/againerju/tcg_recognition/blob/a5470aae9bef2facbe75358fe5c62e37dc547154/TCGDB.py#L222

    def __init__(self):
        self.pose = []
        self.maj_cls_name = ""
        self.maj_cls = -1
        self.min_cls_name = ""
        self.min_cls = -1

    def encode_majlabel(self):

        maj_label_dict = {"inactive": 0, "stop": 1, "go": 2, "clear": 3}

        self.maj_cls = maj_label_dict[self.maj_cls_name]

    def encode_minlabel(self):

        min_label_dict = {"inactive_normal-pose": 0, "inactive_out-of-vocabulary": 0, "inactive_transition": 0,
                          "stop_both-static": 1, "stop_both-dynamic": 2, "stop_left-static": 3,
                          "stop_left-dynamic": 4, "stop_right-static": 5, "stop_right-dynamic": 6,
                          "clear_left-static": 7, "clear_right-static": 8, "go_both-static": 9,
                          "go_both-dynamic": 10, "go_left-static": 11, "go_left-dynamic": 12,
                          "go_right-static": 13, "go_right-dynamic": 14}

        self.min_cls = min_label_dict[self.maj_cls_name + "_" + self.min_cls_name]
