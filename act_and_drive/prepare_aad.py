import os

import numpy as np

import csv
import pickle
import argparse


def processData():
    # Content csv-file
    # frame_id (col: 0); timestep (col:1); nose (col: 2, 3, 4, 5) YES; left_elbow (col: 6, 7, 8, 9) YES; left_wrist (col: 10, 11, 12, 13) YES; right_heel (col: 14, 15, 16, 17) NO; 
    # right_hip (col: 18, 19, 20, 21) YES; right_small_toe (col: 22, 23, 24 ,25) NO; neck(col: 26,27,28,29) YES; left_small_toe (col: 30,31,32,33) NO; 
    # right_wrist (col: 34,35,36,37) YES; right_ankle (col: 38,39,40,41) NO; left_hip (col: 42,43,44,45) YES; left_heel (col: 46,47,48,49) NO; left_knee (col:50,51,52,53) YES; 
    # left_eye (col:54,55,56,57) YES; mid_hip(col: 58,59,60,61) YES; background (col: 62,63,64,65) NO; left_ear (col: 66,67,68,69) YES; right_elbow (col: 70,71,72,73) YES; 
    # right_shoulder (col: 74,75,76,77) YES; right_knee (col:78,79,80,81) YES; left_shoulder (col:82,83,84,85) YES; left_big_toe(col:86,87,88,89) NO; rigth_eye (col: 90,91,92,93) YES; 
    # right_ear(colo:94,95,96,97) YES; right_big_tow (col: 98,99,100,101) NO; left_ankle (col: 102,103,104,105) NO

    column_frame_id = 0
    columns_upper_body = [2, 3, 4, 6, 7, 8, 10, 11, 12, 18, 19, 20, 26, 27, 28, 34, 35, 36, 42, 43, 44, 54, 55, 56, 58, 59, 60, 70, 71, 72, 74, 75, 76, 82, 83, 84, 90, 91, 92]

    output = dict()
    for i in range(1, 16):
        output['vp' + str(i)] = dict()

    for root, dir, files in os.walk('./openpose_3d'):
        for i, file in enumerate(files):
            person_id = root.split('/')[-1]
            run_id = file[:4]
            data = np.genfromtxt(os.path.join(root, file), delimiter=',')
            # Remove first row with nan's
            data = data[1:, :]
            # Remove frames with no skeleton
            upper_skeleton = data[:, columns_upper_body]
            frame_ids = data[:, column_frame_id]
            output[person_id][run_id] = [frame_ids, upper_skeleton]

    with open('./skeletons.pkl', 'wb') as f:
        pickle.dump(output, f)


def processLabels(root_dir, files, out_file):
    labels = dict()
    for i in range(3):
        labels['test' + str(i)] = list()
        labels['train' + str(i)] = list()
        labels['val' + str(i)] = list()

    for file in files:
        with open(os.path.join(root_dir, file), newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            split = file.split('.')[-2]
            split_number = file.split('.')[-3].split('_')[-1]
            id = split + split_number
            for i, row in enumerate(spamreader):
                # Skip the header
                if i == 0:
                    continue

                data = list() # person, run, start, end, label
                data.append(row[0])
                run_id = int(row[1].split('/')[1].split('_')[0][3])
                data.append(run_id)
                data.append(row[3])
                data.append(row[4])
                data.append(row[5])
                labels[id].append(data)

    with open(out_file, 'wb') as f:
        pickle.dump(labels, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action='store_true', help='Process the data of the dataset.')
    parser.add_argument('--fine_label', action='store_true', help='Prepare fine graded label')
    parser.add_argument('--coarse_label', action='store_true', help='Process the coarse label')
    opt = parser.parse_args()
    
    if opt.data:
        processData()

    if opt.fine_label:
        root_dir = './activities_3s/a_column_driver'
        files = ['midlevel.chunks_90.split_0.test.csv', 'midlevel.chunks_90.split_0.train.csv', 'midlevel.chunks_90.split_0.val.csv',
                'midlevel.chunks_90.split_1.test.csv', 'midlevel.chunks_90.split_1.train.csv', 'midlevel.chunks_90.split_1.val.csv',
                'midlevel.chunks_90.split_2.test.csv', 'midlevel.chunks_90.split_2.train.csv', 'midlevel.chunks_90.split_2.val.csv']
        out_file = './fine_labels.pkl'
        processLabels(root_dir, files, out_file)

    if opt.coarse_label:
        root_dir = './activities_3s/a_column_driver'
        files = ['tasklevel.chunks_90.split_0.test.csv', 'tasklevel.chunks_90.split_0.train.csv', 'tasklevel.chunks_90.split_0.val.csv',
                'tasklevel.chunks_90.split_1.test.csv', 'tasklevel.chunks_90.split_1.train.csv', 'tasklevel.chunks_90.split_1.val.csv',
                'tasklevel.chunks_90.split_2.test.csv', 'tasklevel.chunks_90.split_2.train.csv', 'tasklevel.chunks_90.split_2.val.csv']
        out_file = './coarse_labels.pkl'
        processLabels(root_dir, files, out_file)
