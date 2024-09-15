from torch.utils.data import Dataset


class ListDatasetWrapper(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def collate_fn(self, batch):
        return batch

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)
