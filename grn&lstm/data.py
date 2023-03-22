import torch
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, data_len=300, split=150, batch_size=5):

        self.data_len = data_len
        self.split = split
        self.batch_size = batch_size

        self.t = torch.linspace(0, 12*3.14, self.data_len)
        self.sin_t = torch.sin(self.t)
        self.cos_t = torch.cos(self.t)

    def __len__(self):
        return self.data_len

    def get_train_data(self):
        split, batch_size = self.split, self.batch_size

        train_t = self.t[0:split].reshape(-1, batch_size, 1)
        train_x = self.sin_t[0:split].reshape(-1, batch_size, 1)
        train_y = self.cos_t[0:split].reshape(-1, batch_size, 1)
        return train_t, train_x, train_y

    def get_test_data(self):
        split, batch_size = self.split, self.batch_size

        test_t = self.t[split:].reshape(-1, batch_size, 1)
        test_x = self.sin_t[split:].reshape(-1, batch_size, 1)
        test_y = self.cos_t[split:].reshape(-1, batch_size, 1)
        return test_t, test_x, test_y


if __name__ == '__main__':
    data = Dataset(data_len=300, split=150, batch_size=5)
    train_t, train_x, train_y = data.get_train_data()
    test_t, test_t, txst_y = data.get_test_data()
