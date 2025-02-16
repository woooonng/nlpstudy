from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import omegaconf
from typing import Union, List, Tuple, Literal

class IMDBDataset(Dataset):
    def __init__(self, data_config : omegaconf.DictConfig, split : Literal['train', 'valid', 'test']):
        """
        Inputs :
            data_config : omegaconf.DictConfig{
                model_name : str
                max_len : int
                valid_size : float
            }
            split : Literal['train', 'valid', 'test']
        Outputs : None
        """
        self.data_config = data_config
        self.split = split

        tokenizer_dict = {
            'bert': "bert-base-uncased",
            'modern_bert': "answerdotai/ModernBERT-base"
        }
        tokenizer_name = tokenizer_dict.get(data_config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        imdb = load_dataset('imdb')
        self.total_data = concatenate_datasets([imdb['train'], imdb['test']])

        # split the data into train, val, test
        train_temp_split = self.total_data.train_test_split(test_size=data_config.valid_size*2)
        train, temp = train_temp_split['train'], train_temp_split['test']

        val_test_split = temp.train_test_split(test_size=0.5)
        val, test = val_test_split['train'], val_test_split['test']

        split_map = {'train': train, 'val': val, 'test': test}
        self.data = split_map[split]

        pos_neg_count = [0, 0]
        for example in test:
            pos_neg_count[example['label']] += 1
        print(f"{split} label distribution for train dataset : {pos_neg_count}")
        print(f">> SPLIT : {self.split} | '{split}' Data Length : ", len(self.data['text']))

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[dict, int]:
        """
        Inputs :
            idx : int
        Outputs :
            inputs : dict{
                input_ids : torch.Tensor
                token_type_ids : torch.Tensor
                attention_mask : torch.Tensor
            }
            label : int
        """
        inputs = self.tokenizer(self.data[idx]['text'], return_tensors='pt', truncation=True, padding=True, max_length=self.data_config.max_len)
        label = self.data[idx]['label']
        return inputs, label

    @staticmethod
    def collate_fn(batch : List[Tuple[dict, int]]) -> dict:
        """
        Inputs :
            batch : List[Tuple[dict, int]]
        Outputs :
            data_dict : dict{
                input_ids : torch.Tensor
                token_type_ids : torch.Tensor
                attention_mask : torch.Tensor
                label : torch.Tensor
            }
        """
        available_keys = batch[0][0].keys()
        data_dict = {}
        for key in available_keys:
            temp = [item[0][key].squeeze(0) for item in batch]
            temp_padded = pad_sequence(temp, batch_first=True, padding_value=0)
            data_dict[key] = temp_padded

        labels = torch.tensor([item[1] for item in batch])
        data_dict['label'] = labels
        return data_dict
    
def get_dataloader(data_config : omegaconf.DictConfig, split : Literal['train', 'valid', 'test']) -> torch.utils.data.DataLoader:
    """
    Output : torch.utils.data.DataLoader
    """
    dataset = IMDBDataset(data_config, split)
    dataloader = DataLoader(dataset, batch_size=data_config.batch_size, shuffle=(split=='train'), collate_fn=IMDBDataset.collate_fn)
    return dataloader