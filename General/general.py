import random
import os, sys
import numpy as np
import pandas as pd
import torch
from General_Model import Memory_Meta
from collections import Counter

from function import PepTCRdict, train_epoch, valid_epoch, select_data_add_negative_data
sys.path.append("..")
from utils import Data_config, Project_path, Data_output, Device, Aa_dict, Model_config, Train_Round, MLogger, add_negative_data



if __name__ == '__main__':
    # Initialize a new model
    model = Memory_Meta(Model_config).to(Device)
    optimizer = torch.optim.Adam(model.net.parameters(), lr=Data_config['Train']['Meta_learning']['Model_parameter']['meta_lr'])
    # negative data
    negative_data = np.loadtxt(os.path.join(Project_path, 'Control_dataset.txt'), dtype=str)

    for kf_time in range(Data_config['dataset']['current_fold'][0], Data_config['dataset']['current_fold'][1]):
        # create test file (positive data is same as other training)
        genetal_test_path = os.path.join(Project_path, Data_output, 'kfold' + str(kf_time), 'KFold_' + str(kf_time) + '_test_general_data.csv')
        if not os.path.exists(genetal_test_path):
            test_data_path = os.path.join(Project_path, Data_output, 'kfold' + str(kf_time), 'KFold_' + str(kf_time) + '_test.csv')
            test_data = pd.read_csv(test_data_path)
            test_data_all = add_negative_data(test_data, negative_data)
            test_data_all.to_csv(genetal_test_path, index=False)
    for index in range(Train_Round):
        for kf_time in range(Data_config['dataset']['current_fold'][0], Data_config['dataset']['current_fold'][1]):
            # Set the random seed
            seed = random.randint(0, 10000)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.cuda.manual_seed(seed)

            # print(kf_time)
            train_data_4fold_path = os.path.join(Project_path, Data_output, 'kfold' + str(kf_time), 'KFold_' + str(kf_time) + '_train.csv')
            train_data_4fold = pd.read_csv(train_data_4fold_path)
            all_peptide = list(Counter(train_data_4fold['peptide']).keys())
            train_peptide = [all_peptide[i] for i in np.random.choice(len(all_peptide), int(len(all_peptide) * 0.8))]

            train_data_all = select_data_add_negative_data(train_data_4fold[train_data_4fold['peptide'].isin(train_peptide)], negative_data, spt_num=2)
            valid_data_all = select_data_add_negative_data(train_data_4fold[~train_data_4fold['peptide'].isin(train_peptide)], negative_data, spt_num=2)

            Train_data = PepTCRdict(train_data_all, aa_dict_path=Aa_dict)
            Train_db = torch.utils.data.DataLoader(Train_data, Data_config['Train']['General']['train_batch_size'], shuffle=Data_config['Train']['General']['sample_shuffle'], num_workers=0)

            Valid_data = PepTCRdict(valid_data_all, aa_dict_path=Aa_dict)
            Valid_db = torch.utils.data.DataLoader(Valid_data, Data_config['Train']['General']['valid_batch_size'], shuffle=Data_config['Train']['General']['sample_shuffle'], num_workers=0)

            model_save_path = os.path.join(Data_config['Train']['General']['Train_output_dir'], 'Round' + str(index + 1), 'kfold' + str(kf_time))
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            # logger
            logger = MLogger(os.path.join(model_save_path, 'kfold' + str(kf_time) + '_training.log'))
            logger.info('Round:' + str(index + 1) + ', KFold:' + str(kf_time))
            best_loss = float('inf')
            for epoch in range(Data_config['Train']['General']['epoch']):
                logger.info(f"Epoch: {epoch + 1}")
                model.train()
                train_loss = train_epoch(model, Train_db, optimizer, logger)
                model.eval()
                with torch.no_grad():  # valid
                    valid_loss = valid_epoch(model, Valid_db, logger)

                if valid_loss.avg < best_loss:
                    best_loss = valid_loss.avg
                    torch.save(model.state_dict(), os.path.join(model_save_path, 'KFold_' + str(kf_time) + '_best.pt'))
                    logger.info("Saved Best Model!")
