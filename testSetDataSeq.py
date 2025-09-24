import torch
from torch.utils.data import SequentialSampler, Dataset, DataLoader,BatchSampler


class SimpleDataset(Dataset):
    def __init__(self):
        self.data = torch.arange(20)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


dataset = SimpleDataset()

sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=4)

drop_last = False
batch_sampler = BatchSampler(sampler, batch_size=4, drop_last=drop_last)

print("---- Using DataLoader ----")

for batch in dataloader:
    print(batch)

print("---- Using BatchSampler ----")

for batch in batch_sampler:
    print(batch)    