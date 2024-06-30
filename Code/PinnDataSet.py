from typing import Any
from torch.utils.data.dataset import Dataset
from pathlib import Path
import chardet
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch


def read_data(path: str) -> list:
    with open(path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
        data = pd.read_csv(path, encoding=encoding)
    return data


class PINNsDataset(Dataset):
    def __init__(self,
                 root: str,
                 batch_size: int,
                 ) -> None:
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size
        self.file_paths = [str(p) for p in self.root.glob('*.CSV')]
        self.data_len = len(self.file_paths)
        
        # load all data(*.CSV)
        self.datas = []

        with tqdm(self.file_paths, total=self.data_len, desc='Loading data *.CSV') as pbar:
            for path in pbar:
                data = read_data(path)
                self.datas.append(data)


    def __getitem__(self, index: Any) -> Any:
        data = self.datas[index]
        start = np.random.randint(0, max(len(data) - self.batch_size, 1))
        end = start + self.batch_size
        batch_data = data.iloc[start:end]


        t_omega_k_h_xdata = torch.tensor(batch_data.iloc[:, :5].values).float()
        U_data = torch.tensor(batch_data.iloc[:, 5].values).float()
        return t_omega_k_h_xdata, U_data


    def __len__(self) -> int:
        return self.data_len
    

def pinns_collect_fn(x):
    return x



        



    








