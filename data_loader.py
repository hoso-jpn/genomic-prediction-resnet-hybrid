import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class GenomeDataset(Dataset):
    def __init__(self, num_samples=100, num_snps=1000):
        # 本番はここで pd.read_csv や np.load を行います
        # 現在はテスト用にランダムな値を生成
        self.genotypes = torch.randn(num_samples, num_snps)
        self.phenotypes = torch.randn(num_samples, 1)

    def __len__(self):
        return len(self.genotypes)

    def __getitem__(self, idx):
        return self.genotypes[idx], self.phenotypes[idx]

def get_loader(batch_size=32):
    dataset = GenomeDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)