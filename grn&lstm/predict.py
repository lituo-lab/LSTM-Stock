import torch
from data import Dataset
from net_gru import gruRNN
import matplotlib.pyplot as plt

DEVICE = 'cpu'

data = Dataset(data_len=300, split=150, batch_size=5)
train_t, train_x, train_y = data.get_train_data()
net = gruRNN(1, 16, 1, 5).to(DEVICE)
net.load_state_dict(torch.load('net.param'))

#%% test
test_t, test_x, test_y = data.get_test_data()

train_y_pre = net(train_x).detach().numpy().reshape(-1, 1)
train_x = train_x.detach().numpy().reshape(-1, 1)
train_y = train_y.detach().numpy().reshape(-1, 1)

test_y_pre = net(test_x).detach().numpy().reshape(-1, 1)
test_x = test_x.detach().numpy().reshape(-1, 1)
test_y = test_y.detach().numpy().reshape(-1, 1)

train_t = train_t.numpy().reshape(-1, 1)
test_t = test_t.numpy().reshape(-1, 1)

#%% plot
plt.figure()
plt.plot(train_t, train_x, 'g', label='sin_trn')
plt.plot(train_t, train_y, 'b', label='ref_cos_trn')
plt.plot(train_t, train_y_pre, 'y--', label='pre_cos_trn')

plt.plot(test_t, test_x, 'c', label='sin_tst')
plt.plot(test_t, test_y, 'k', label='ref_cos_tst')
plt.plot(test_t, test_y_pre, 'm--', label='pre_cos_tst')

plt.plot([test_t[0], test_t[0]], [-1.2, 4.0], 'r--', label='separation line')

plt.xlabel('t')
plt.ylabel('sin(t) and cos(t)')
plt.xlim(train_t[0], test_t[-1])
plt.ylim(-1.2, 3)
plt.legend(loc='upper right')
plt.text(14, 2, "train", size=15, alpha=1.0)
plt.text(20, 2, "test", size=15, alpha=1.0)

plt.show()
