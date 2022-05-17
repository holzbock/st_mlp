import os

import torch
import numpy as np

import pickle
import copy


class AadDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, model, mode='train', split=0, label_type='fine', seq_len=90, num_classes=34, convert_coord=0, 
                noise=0, noise_std=0):
        # Load AAD data
        self.data_path = data_path
        self.model = model
        self.mode = mode
        self.split = split
        self.label_type = label_type
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.convert_coord = convert_coord  # 0: world coodinate system 1: world coordinate system mid_hip first frame 
                                            # 2: convert to image coordinates 3: image coordinates and normalize mid_hip first frame
        self.data = list()
        self.labels = list()
        self.padded = list()
        self.noise = noise # 0: no noise added; 1: noise only to the underrepresented classes; 2: noise to all classes
        self.noise_std = noise_std
        self.noise_classes = [0, 4, 5, 7, 8, 10, 11]
        if self.label_type == 'fine':
            self.class_list = {'taking_laptop_from_backpack': 0, 'putting_laptop_into_backpack': 1, 'opening_backpack': 2, 
                                'closing_laptop': 3, 'opening_laptop': 4, 'placing_an_object': 5, 'fetching_an_object': 6, 
                                'closing_door_outside': 7, 'closing_door_inside': 8, 'opening_door_outside': 9,
                                'opening_door_inside': 10, 'entering_car': 11, 'exiting_car': 12, 'unfastening_seat_belt': 13, 
                                'fastening_seat_belt': 14, 'preparing_food': 15, 'closing_bottle': 16, 'opening_bottle': 17, 
                                'drinking': 18, 'eating': 19, 'taking_off_sunglasses': 20, 'putting_on_sunglasses': 21,
                                'taking_off_jacket': 22, 'putting_on_jacket': 23, 'looking_or_moving_around (e.g. searching)': 24, 
                                'sitting_still': 25, 'pressing_automation_button': 26, 'using_multimedia_display': 27, 
                                'writing': 28, 'working_on_laptop': 29, 'talking_on_phone': 30, 'interacting_with_phone': 31, 
                                'reading_newspaper': 32, 'reading_magazine': 33}
            self.label_names = ['taking_laptop_from_backpack', 'putting_laptop_into_backpack', 'opening_backpack', 'closing_laptop', 
                                'opening_laptop', 'placing_an_object', 'fetching_an_object', 'closing_door_outside', 
                                'closing_door_inside', 'opening_door_outside', 'opening_door_inside', 'entering_car', 'exiting_car', 
                                'unfastening_seat_belt', 'fastening_seat_belt', 'preparing_food', 'closing_bottle', 'opening_bottle',
                                'drinking', 'eating', 'taking_off_sunglasses', 'putting_on_sunglasses', 'taking_off_jacket', 
                                'putting_on_jacket', 'looking_or_moving_around (e.g. searching)', 'sitting_still', 
                                'pressing_automation_button', 'using_multimedia_display', 'writing', 'working_on_laptop', 
                                'talking_on_phone', 'interacting_with_phone', 'reading_newspaper', 'reading_magazine']
        elif self.label_type == 'coarse':
            self.class_list = {'final_task': 0, 'work': 1, 'eat_drink': 2, 'read_write_magazine': 3, 'put_on_jacket': 4, 
                                'take_off_sunglasses': 5, 'read_write_newspaper': 6, 'fasten_seat_belt': 7, 'put_on_sunglasses': 8, 
                                'watch_video': 9, 'take_off_jacket': 10, 'hand_over': 11, }
            self.label_names = ['final_task', 'work', 'eat_drink', 'read_write_magazine', 'put_on_jacket', 'take_off_sunglasses', 
                                'read_write_newspaper', 'fasten_seat_belt', 'put_on_sunglasses', 'watch_video', 'take_off_jacket', 
                                'hand_over']
        else:
            raise ValueError('Unkown label_type %s in the initialization of the Act and Drive dataset.'%self.label_type)
        
        # It is possible to change the split using the same dataset object. 
        # Set the new split_id with the setSplit() method and then call 
        # loadData() to load the data from the new split.
        self.loadData()


    def loadData(self):
        self.data = list()
        self.labels = list()
        self.padded = list()

        with open(os.path.join(self.data_path, 'skeletons.pkl'), 'rb') as f:
            raw_data = pickle.load(f)

        label_file = self.label_type + '_labels.pkl'
        with open(os.path.join(self.data_path, label_file), 'rb') as f:
            raw_label = pickle.load(f)

        key = self.mode + str(self.split)
        label_split = raw_label[key]

        with open(os.path.join(self.data_path, 'calib_data.pkl'), 'rb') as f:
            calib_data = pickle.load(f)
        
        for lab in label_split:
            # lab contains: person, run, start frame, end frame, label
            frame_ids, skeletons = raw_data['vp' + str(lab[0])]['run' + str(lab[1])]
            part = copy.deepcopy(skeletons[int(lab[2]):int(lab[3])+1, :])
            # interpolate missing points
            part = self.interpolate(part)

            padding = self.seq_len - part.shape[0]
            if padding < 0:
                part = part[part.shape[0] - self.seq_len:, :]

            if self.convert_coord == 0:     # world coordinate system
                v=0 # No action
            elif self.convert_coord == 1:   
                # world coordinate system and normalized on the mid_hip of the first frame, mid_hip is col 24, 25, 26
                if part.shape[0] != 0:
                    first_frame_mide_hip = np.repeat(np.tile(part[0, [24, 25, 26]], 13).reshape(1, -1), part.shape[0], axis=0)
                    part = part - first_frame_mide_hip
            elif self.convert_coord == 2:   
                # camera coordinate system
                rot = calib_data['vp' + str(lab[0])]['run' + str(lab[1])]['rotation']
                trans = calib_data['vp' + str(lab[0])]['run' + str(lab[1])]['translation'].reshape(3,1)
                for i in range(13):
                    part[:, i*3:(i+1)*3] = (rot.T.dot(part[:, i*3:(i+1)*3].T) + trans).T
            elif self.convert_coord == 3:   
                # camera coordinate system and normalized on the mid_hip of the first frame, mid_hip is col 24, 25, 26
                rot = calib_data['vp' + str(lab[0])]['run' + str(lab[1])]['rotation']
                trans = calib_data['vp' + str(lab[0])]['run' + str(lab[1])]['translation'].reshape(3,1)
                for i in range(13):
                    translated = (rot.T.dot(part[:, i*3:(i+1)*3].T) + trans).T
                    part[:, i*3:(i+1)*3] = translated
                if part.shape[0] != 0:
                    first_frame_mide_hip = np.repeat(np.tile(part[0, [24, 25, 26]], 13).reshape(1, -1), part.shape[0], axis=0)
                    part = part - first_frame_mide_hip
            else:   # Unknown coordinate system and normalization
                raise ValueError('Unknown data convertion in the dataloader: %i'%self.convert_coord)

            label_num = self.class_list[lab[-1]]
            
            if padding > 0:
                part = np.pad(part, ((0, padding), (0, 0)))

            pad_mask = torch.zeros(self.seq_len)
            pad_mask[:self.seq_len - torch.tensor(padding).clamp(0)] = 1
            self.labels.append(torch.tensor(label_num))
            self.data.append(torch.from_numpy(part).float())
            self.padded.append(pad_mask)


    def interpolate(self, skel):
        l_seq = skel.shape[0]
        for i in range(skel.shape[1]):
            x_corr_inter = np.array(list(range(l_seq)))
            x_corr_orig = np.array(list(range(l_seq)))
            tmp = np.nonzero(skel[:, i])[0]
            if tmp.shape[0] == 0:
                continue
            # Define start and end of the interpolation; it's possible that at the beginning and the end are zeros
            start = tmp.min()
            end = tmp.max()
            x_corr_orig = x_corr_orig[skel[:, i] != 0]
            x_corr_inter = x_corr_inter[start:end+1]
            part = skel[skel[:, i] != 0, i]
            skel[start:end+1, i] = np.interp(x_corr_inter, x_corr_orig, part)

        return skel


    def setSplit(self, num):
        self.split = num


    def getSplit(self):
        return self.split


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        data_out = self.data[index]
        label_out = self.labels[index]
        good_out = self.padded[index]
        if self.noise == 1: # noise only to the underrepresented classes
            if label_out in self.noise_classes:
                noise = torch.normal(0, self.noise_std, size=data_out.shape)
                data_out += noise
        elif self.noise == 2: # add noise to all classes
            noise = torch.normal(0, self.noise_std, size=data_out.shape)
            data_out += noise
        return data_out, label_out, good_out


    def getLossWeights(self):
        # occurence of the single classes | weights (1 - num/all) | 1/(num/all)
        weights = torch.zeros(self.num_classes)
        all = len(self.labels)
        labels = torch.stack(self.labels)

        nums = [(labels == i).sum().item() for i in range(self.num_classes)]
        nums_max = max(nums)
        weights = nums_max / torch.tensor(nums)

        return weights
