# import numpy as np
# import pickle
# from Loader import SignedPairsDataset, get_index_dicts
# from Trainer import ERGOLightning
# from torch.utils.data import DataLoader
# from argparse import Namespace
# import torch
# import pandas as pd
# import os
# from os import listdir
# from os.path import isfile, join
# import sys

# def read_input_file(datafile):
#     amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
#     all_pairs = []
#     def invalid(seq):
#         return pd.isna(seq) or any([aa not in amino_acids for aa in seq])
#     data = pd.read_csv(datafile)
#     for index in range(len(data)):
#         sample = {}
#         sample['tcra'] = data['TRA'][index]
#         sample['tcrb'] = data['TRB'][index]
#         sample['va'] = data['TRAV'][index]
#         sample['ja'] = data['TRAJ'][index]
#         sample['vb'] = data['TRBV'][index]
#         sample['jb'] = data['TRBJ'][index]
#         sample['t_cell_type'] = data['T-Cell-Type'][index]
#         sample['peptide'] = data['Peptide'][index]
#         sample['mhc'] = data['MHC'][index]
#         # we do not use the sign
#         sample['sign'] = 0
#         if invalid(sample['tcrb']) or invalid(sample['peptide']):
#             continue
#         if invalid(sample['tcra']):
#             sample['tcra'] = 'UNK'
#         all_pairs.append(sample)
#     return all_pairs, data


# def load_model(hparams, checkpoint_path):
#     # 检查是否有可用的 GPU
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # 使用 map_location 来正确加载模型
#     model = ERGOLightning(hparams)
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(checkpoint['state_dict'])
#     model.to(device)
#     model.eval()
#     return model


# def get_model(dataset):
#     if dataset == 'vdjdb':
#         version = '1veajht'
#     if dataset == 'mcpas':
#         version = '1meajht'
#     # get model file from version
#     checkpoint_path = os.path.join('Models', 'version_' + version, 'checkpoints')
#     files = [f for f in listdir(checkpoint_path) if isfile(join(checkpoint_path, f))]
#     checkpoint_path = os.path.join(checkpoint_path, files[0])
#     # get args from version
#     args_path = os.path.join('Models', 'version_' + version, 'meta_tags.csv')
#     with open(args_path, 'r') as file:
#         lines = file.readlines()
#         args = {}
#         for line in lines[1:]:
#             key, value = line.strip().split(',')
#             if key in ['dataset', 'tcr_encoding_model', 'cat_encoding']:
#                 args[key] = value
#             else:
#                 args[key] = eval(value)
#     hparams = Namespace(**args)
#     checkpoint = checkpoint_path
#     model = load_model(hparams, checkpoint)
#     train_pickle = 'Samples/' + model.dataset + '_train_samples.pickle'
#     test_pickle = 'Samples/' + model.dataset + '_test_samples.pickle'
#     return model, train_pickle


# def get_train_dicts(train_pickle):
#     with open(train_pickle, 'rb') as handle:
#         train = pickle.load(handle)
#     train_dicts = get_index_dicts(train)
#     return train_dicts


# def predict(dataset, test_file):
#     # 创建输出文件夹（如果不存在）
#     output_dir = 'prediction1'
#     os.makedirs(output_dir, exist_ok=True)
    
#     model, train_file = get_model(dataset)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     model.eval()
    
#     train_dicts = get_train_dicts(train_file)
#     test_samples, dataframe = read_input_file(test_file)
#     test_dataset = SignedPairsDataset(test_samples, train_dicts)
    
#     batch_size = 1000
#     loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
#                        collate_fn=lambda b: test_dataset.collate(b, 
#                                                                tcr_encoding=model.tcr_encoding_model,
#                                                                cat_encoding=model.cat_encoding))
    
#     outputs = []
#     with torch.no_grad():
#         for batch_idx, batch in enumerate(loader):
#             # 将批次数据移到正确的设备上
#             batch = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in batch)
#             output = model.validation_step(batch, batch_idx)
#             if output:
#                 outputs.extend(output['y_hat'].cpu().numpy().tolist())
    
#     # 构建输出文件路径
#     input_filename = os.path.basename(test_file)  # 获取输入文件名
#     output_filename = f'{os.path.splitext(input_filename)[0]}_predicted.csv'
#     output_path = os.path.join(output_dir, output_filename)
    
#     dataframe['Score'] = outputs
#     # 保存预测结果
#     dataframe.to_csv(output_path, index=False)
#     return dataframe


# if __name__ == '__main__':
#     if len(sys.argv) != 3:
#         print("Usage: python Predict.py dataset input_file")
#         print("Example: python Predict.py mcpas input.csv")
#         sys.exit(1)
        
#     df = predict(sys.argv[1], sys.argv[2])
#     print(f"Predictions saved in predictions folder")


# # NOTE: fix sklearn import problem with this in terminal:
# # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dsi/speingi/anaconda3/lib/
# # or just conda install libgcc



# import numpy as np
# import pickle
# from Loader import SignedPairsDataset, get_index_dicts
# from Trainer import ERGOLightning
# from torch.utils.data import DataLoader
# from argparse import Namespace
# import torch
# import pandas as pd
# import os
# from os import listdir
# from os.path import isfile, join
# import sys
# import glob
# import argparse
# from tqdm import tqdm  # 添加进度条支持

# def read_input_file(datafile):
#     """读取输入CSV文件"""
#     amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
#     all_pairs = []
    
#     def invalid(seq):
#         return pd.isna(seq) or any([aa not in amino_acids for aa in seq])
    
#     try:
#         data = pd.read_csv(datafile)
#         for index in range(len(data)):
#             sample = {}
#             sample['tcra'] = data['TRA'][index]
#             sample['tcrb'] = data['TRB'][index]
#             sample['va'] = data['TRAV'][index]
#             sample['ja'] = data['TRAJ'][index]
#             sample['vb'] = data['TRBV'][index]
#             sample['jb'] = data['TRBJ'][index]
#             sample['t_cell_type'] = data['T-Cell-Type'][index]
#             sample['peptide'] = data['Peptide'][index]
#             sample['mhc'] = data['MHC'][index]
#             sample['sign'] = 0  # we do not use the sign
            
#             if invalid(sample['tcrb']) or invalid(sample['peptide']):
#                 continue
#             if invalid(sample['tcra']):
#                 sample['tcra'] = 'UNK'
#             all_pairs.append(sample)
#         return all_pairs, data
#     except Exception as e:
#         print(f"读取文件 {datafile} 时出错: {str(e)}")
#         return None, None

# def load_model(hparams, checkpoint_path):
#     """加载模型"""
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     try:
#         model = ERGOLightning(hparams)
#         checkpoint = torch.load(checkpoint_path, map_location=device)
#         model.load_state_dict(checkpoint['state_dict'])
#         model.to(device)
#         model.eval()
#         return model
#     except Exception as e:
#         print(f"加载模型时出错: {str(e)}")
#         return None

# def get_model(dataset):
#     """获取对应数据集的模型"""
#     if dataset == 'vdjdb':
#         version = '1veajht'
#     elif dataset == 'mcpas':
#         version = '1meajht'
#     else:
#         raise ValueError(f"不支持的数据集: {dataset}")

#     try:
#         # 获取模型文件
#         checkpoint_path = os.path.join('Models', 'version_' + version, 'checkpoints')
#         files = [f for f in listdir(checkpoint_path) if isfile(join(checkpoint_path, f))]
#         if not files:
#             raise FileNotFoundError("找不到检查点文件")
#         checkpoint_path = os.path.join(checkpoint_path, files[0])

#         # 获取参数
#         args_path = os.path.join('Models', 'version_' + version, 'meta_tags.csv')
#         with open(args_path, 'r') as file:
#             lines = file.readlines()
#             args = {}
#             for line in lines[1:]:
#                 key, value = line.strip().split(',')
#                 if key in ['dataset', 'tcr_encoding_model', 'cat_encoding']:
#                     args[key] = value
#                 else:
#                     args[key] = eval(value)

#         hparams = Namespace(**args)
#         model = load_model(hparams, checkpoint_path)
#         train_pickle = 'Samples/' + model.dataset + '_train_samples.pickle'
        
#         return model, train_pickle
#     except Exception as e:
#         print(f"获取模型时出错: {str(e)}")
#         return None, None

# def get_train_dicts(train_pickle):
#     """获取训练数据字典"""
#     try:
#         with open(train_pickle, 'rb') as handle:
#             train = pickle.load(handle)
#         return get_index_dicts(train)
#     except Exception as e:
#         print(f"加载训练数据时出错: {str(e)}")
#         return None

# def process_folder(dataset, input_dir):
#     """处理文件夹中的所有CSV文件"""
#     # 创建输出目录
#     output_dir = 'zeropr'
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 加载模型
#     print("正在加载模型...")
#     model, train_file = get_model(dataset)
#     if model is None:
#         print("模型加载失败")
#         return
        
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"使用设备: {device}")
    
#     model = model.to(device)
#     model.eval()
    
#     # 加载训练数据字典
#     print("正在加载训练数据...")
#     train_dicts = get_train_dicts(train_file)
#     if train_dicts is None:
#         print("训练数据加载失败")
#         return
    
#     # 获取所有CSV文件
#     csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
#     if not csv_files:
#         print(f"在 {input_dir} 中没有找到CSV文件！")
#         return
    
#     print(f"找到 {len(csv_files)} 个CSV文件待处理")
    
#     # 处理每个CSV文件
#     for i, csv_file in enumerate(csv_files, 1):
#         try:
#             file_name = os.path.basename(csv_file)
#             print(f"\n正在处理 [{i}/{len(csv_files)}]: {file_name}")
            
#             # 读取数据
#             test_samples, dataframe = read_input_file(csv_file)
#             if test_samples is None or dataframe is None:
#                 print(f"跳过文件 {file_name}")
#                 continue
                
#             test_dataset = SignedPairsDataset(test_samples, train_dicts)
            
#             # 创建数据加载器
#             batch_size = 1000
#             loader = DataLoader(
#                 test_dataset, 
#                 batch_size=batch_size, 
#                 shuffle=False,
#                 collate_fn=lambda b: test_dataset.collate(
#                     b, 
#                     tcr_encoding=model.tcr_encoding_model,
#                     cat_encoding=model.cat_encoding
#                 )
#             )
            
#             # 进行预测
#             outputs = []
#             with torch.no_grad():
#                 for batch in tqdm(loader, desc="预测进度"):
#                     batch = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in batch)
#                     output = model.validation_step(batch, 0)
#                     if output:
#                         outputs.extend(output['y_hat'].cpu().numpy().tolist())
            
#             # 保存预测结果
#             output_filename = f'{os.path.splitext(file_name)[0]}_predicted.csv'
#             output_path = os.path.join(output_dir, output_filename)
            
#             dataframe['Score'] = outputs
#             dataframe.to_csv(output_path, index=False)
            
#             print(f"文件 {file_name} 处理完成，结果保存在: {output_path}")
            
#         except Exception as e:
#             print(f"处理文件 {file_name} 时发生错误: {str(e)}")
#             continue
    
#     print("\n所有文件处理完成！")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, required=True, choices=['vdjdb', 'mcpas'],
#                       help='使用的数据集类型 (vdjdb 或 mcpas)')
#     parser.add_argument('--input_dir', type=str, required=True,
#                       help='包含CSV文件的文件夹路径')
    
#     args = parser.parse_args()
         
#     process_folder(args.dataset, args.input_dir)

# if __name__ == '__main__':
#     main()




import numpy as np
import pickle
from Loader import SignedPairsDataset, get_index_dicts
from Trainer import ERGOLightning
from torch.utils.data import DataLoader
from argparse import Namespace
import torch
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import sys
import glob
import argparse
from tqdm import tqdm

def read_input_file(datafile):
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    all_pairs = []
    
    def invalid(seq):
        return pd.isna(seq) or any([aa not in amino_acids for aa in seq])
    
    try:
        data = pd.read_csv(datafile)
        for index in range(len(data)):
            sample = {}
            sample['tcra'] = data['TRA'][index]
            sample['tcrb'] = data['TRB'][index]
            sample['va'] = data['TRAV'][index]
            sample['ja'] = data['TRAJ'][index]
            sample['vb'] = data['TRBV'][index]
            sample['jb'] = data['TRBJ'][index]
            sample['t_cell_type'] = data['T-Cell-Type'][index]
            sample['peptide'] = data['Peptide'][index]
            sample['mhc'] = data['MHC'][index]
            sample['sign'] = 0
            
            if invalid(sample['tcrb']) or invalid(sample['peptide']):
                continue
            if invalid(sample['tcra']):
                sample['tcra'] = 'UNK'
            all_pairs.append(sample)
        return all_pairs, data
    except Exception as e:
        print(f"Error reading file {datafile}: {str(e)}")
        return None, None

def load_model(hparams, checkpoint_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        model = ERGOLightning(hparams)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def get_model(dataset):
    if dataset == 'vdjdb':
        version = '1veajht'
    elif dataset == 'mcpas':
        version = '1meajht'
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    try:
        checkpoint_path = os.path.join('Models', 'version_' + version, 'checkpoints')
        files = [f for f in listdir(checkpoint_path) if isfile(join(checkpoint_path, f))]
        if not files:
            raise FileNotFoundError("No checkpoint files found")
        checkpoint_path = os.path.join(checkpoint_path, files[0])

        args_path = os.path.join('Models', 'version_' + version, 'meta_tags.csv')
        with open(args_path, 'r') as file:
            lines = file.readlines()
            args = {}
            for line in lines[1:]:
                key, value = line.strip().split(',')
                if key in ['dataset', 'tcr_encoding_model', 'cat_encoding']:
                    args[key] = value
                else:
                    args[key] = eval(value)

        hparams = Namespace(**args)
        model = load_model(hparams, checkpoint_path)
        train_pickle = 'Samples/' + model.dataset + '_train_samples.pickle'
        
        return model, train_pickle
    except Exception as e:
        print(f"Error getting model: {str(e)}")
        return None, None

def get_train_dicts(train_pickle):
    try:
        with open(train_pickle, 'rb') as handle:
            train = pickle.load(handle)
        return get_index_dicts(train)
    except Exception as e:
        print(f"Error loading training data: {str(e)}")
        return None

# def process_folder(dataset, input_dir):
#     output_dir = 'prediction1'
#     os.makedirs(output_dir, exist_ok=True)
    
#     print("Loading model...")
#     model, train_file = get_model(dataset)
#     if model is None:
#         print("Model loading failed")
#         return
        
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     model = model.to(device)
#     model.eval()
    
#     print("Loading training data...")
#     train_dicts = get_train_dicts(train_file)
#     if train_dicts is None:
#         print("Training data loading failed")
#         return
    
#     csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
#     if not csv_files:
#         print(f"No CSV files found in {input_dir}!")
#         return
    
#     print(f"Found {len(csv_files)} CSV files to process")
    
#     for i, csv_file in enumerate(csv_files, 1):
#         try:
#             file_name = os.path.basename(csv_file)
#             print(f"\nProcessing [{i}/{len(csv_files)}]: {file_name}")
            
#             test_samples, dataframe = read_input_file(csv_file)
#             if test_samples is None or dataframe is None:
#                 print(f"Skipping file {file_name}")
#                 continue
                
#             test_dataset = SignedPairsDataset(test_samples, train_dicts)
            
#             batch_size = 1000
#             loader = DataLoader(
#                 test_dataset, 
#                 batch_size=batch_size, 
#                 shuffle=False,
#                 collate_fn=lambda b: test_dataset.collate(
#                     b, 
#                     tcr_encoding=model.tcr_encoding_model,
#                     cat_encoding=model.cat_encoding
#                 )
#             )
            
#             outputs = []
#             with torch.no_grad():
#                 for batch in tqdm(loader, desc="Prediction progress"):
#                     batch = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in batch)
#                     output = model.validation_step(batch, 0)
#                     if output:
#                         outputs.extend(output['y_hat'].cpu().numpy().tolist())
            
#             output_filename = f'{os.path.splitext(file_name)[0]}_predicted.csv'
#             output_path = os.path.join(output_dir, output_filename)
            
#             dataframe['Score'] = outputs
#             dataframe.to_csv(output_path, index=False)
            
#             print(f"File {file_name} processed, results saved to: {output_path}")
            
#         except Exception as e:
#             print(f"Error processing file {file_name}: {str(e)}")
#             continue
    
#     print("\nAll files processed!")

def process_file(dataset, input_file, output_dir):
    # output_dir = 'prediction1'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading model...")
    model, train_file = get_model(dataset)
    if model is None:
        print("Model loading failed")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    model.eval()
    
    print("Loading training data...")
    train_dicts = get_train_dicts(train_file)
    if train_dicts is None:
        print("Training data loading failed")
        return
    
    try:
        file_name = os.path.basename(input_file)
        print(f"\nProcessing file: {file_name}")
        
        test_samples, dataframe = read_input_file(input_file)
        if test_samples is None or dataframe is None:
            print(f"Failed to process file {file_name}")
            return
            
        test_dataset = SignedPairsDataset(test_samples, train_dicts)
        
        batch_size = 1000
        loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            collate_fn=lambda b: test_dataset.collate(
                b, 
                tcr_encoding=model.tcr_encoding_model,
                cat_encoding=model.cat_encoding
            )
        )
        
        outputs = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Prediction progress"):
                batch = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in batch)
                output = model.validation_step(batch, 0)
                if output:
                    outputs.extend(output['y_hat'].cpu().numpy().tolist())
        
        output_filename = f'{os.path.splitext(file_name)[0]}_predicted.csv'
        output_path = os.path.join(output_dir, output_filename)
        
        dataframe['Score'] = outputs
        dataframe.to_csv(output_path, index=False)
        
        print(f"File processed, results saved to: {output_path}")
        
    except Exception as e:
        print(f"Error processing file {file_name}: {str(e)}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['vdjdb', 'mcpas'],
                      help='Dataset type (vdjdb or mcpas)')
    parser.add_argument('--input_file', type=str, required=True,
                      help='Directory containing CSV files')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory containing CSV files')
    args = parser.parse_args()
         
    process_file(args.dataset, args.input_file, args.output_dir)

if __name__ == '__main__':
    main()