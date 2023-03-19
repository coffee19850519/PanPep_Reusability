import argparse
import random
import os
import numpy as np
import pandas as pd
import torch
from sklearn.utils import shuffle
from General_Model import Memory_Meta
from collections import Counter

from function import PepTCRdict, train_epoch, valid_epoch, add_negative_data, project_path, data_config, device, data_output, aa_dict

argparser = argparse.ArgumentParser()
# argparser.add_argument('--positive_input', type=str, help='the path to the positive input data file (*.csv)',
#                        default=os.path.join(project_path, eval(data_config['Train']['General']['Training_dataset'])))
# argparser.add_argument('--negative_input', type=str, help='the path to the negative input data file (*.csv)',
#                        default=os.path.join(project_path, data_config['dataset']['Negative_dataset']))
argparser.add_argument('--epoch', type=int, help='epoch number', default=data_config['Train']['General']['epoch'])
argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate',
                       default=data_config['Train']['Meta_learning']['Model_parameter']['meta_lr'])
args = argparser.parse_args()
# Set the random seed
# torch.manual_seed(222)
# torch.cuda.manual_seed_all(222)
# np.random.seed(222)
# random.seed(222)
# torch.cuda.manual_seed(222)

# This is the model parameters
config = [
    ('self_attention', [[1, 5, 5], [1, 5, 5], [1, 5, 5]]),
    ('linear', [5, 5]),
    ('relu', [True]),
    ('conv2d', [16, 1, 2, 1, 1, 0]),
    ('relu', [True]),
    ('bn', [16]),
    ('max_pool2d', [2, 2, 0]),
    ('flatten', []),
    ('linear', [2, 608])
]

if __name__ == '__main__':
    # Initialize a new model
    model = Memory_Meta(config).to(device)
    optimizer = torch.optim.Adam(model.net.parameters(), lr=args.meta_lr)

    for kf_time in range(data_config['dataset']['current_fold'][0], data_config['dataset']['current_fold'][1]):
        genetal_test_path = os.path.join(project_path, data_output, 'kfold' + str(kf_time), 'KFold_' + str(kf_time) + '_test_general_data.csv')
        if not os.path.exists(genetal_test_path):
            test_data_path = os.path.join(project_path, data_output, 'kfold' + str(kf_time), 'KFold_' + str(kf_time) + '_test.csv')
            test_data = pd.read_csv(test_data_path)
            test_data_all = add_negative_data(test_data, os.path.join(project_path, 'Control_dataset.txt'))
            test_data_all.to_csv(genetal_test_path)

    for kf_time in range(data_config['dataset']['current_fold'][0], data_config['dataset']['current_fold'][1]):
        train_data_4fold_path = os.path.join(project_path, data_output, 'kfold' + str(kf_time), 'KFold_' + str(kf_time) + '_train.csv')
        train_data_4fold = pd.read_csv(train_data_4fold_path)

        train_data_all = add_negative_data(train_data_4fold[: int(len(train_data_4fold) * 0.8)], os.path.join(project_path, 'Control_dataset.txt'))
        valid_data_all = add_negative_data(train_data_4fold[int(len(train_data_4fold) * 0.8):], os.path.join(project_path, 'Control_dataset.txt'))

        Train_data = PepTCRdict(train_data_all, aa_dict_path=aa_dict)
        Train_db = torch.utils.data.DataLoader(Train_data, data_config['Train']['General']['train_batch_size'], shuffle=True, num_workers=1,
                                               pin_memory=True)

        Valid_data = PepTCRdict(valid_data_all, aa_dict_path=aa_dict)
        Valid_db = torch.utils.data.DataLoader(Valid_data, data_config['Train']['General']['valid_batch_size'], shuffle=True, num_workers=1,
                                               pin_memory=True)

        best_loss = float('inf')
        for epoch in range(args.epoch):
            print(f"Epoch: {epoch + 1}")
            model.train()
            train_loss = train_epoch(model, Train_db, optimizer)
            model.eval()
            with torch.no_grad():  # valid
                valid_loss = valid_epoch(model, Valid_db)

            if valid_loss.avg < best_loss:
                best_loss = valid_loss.avg
                torch.save(model.state_dict(), 'KFold_' + str(kf_time) + '_best.pt')
                print("Saved Best Model!")
