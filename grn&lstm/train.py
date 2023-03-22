import torch
from torch import nn, optim
from data import Dataset
from net_gru import gruRNN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = Dataset(data_len=300, split=150, batch_size=5)
train_t, train_x, train_y = data.get_train_data()

net = gruRNN(1, 16, 1, 5).to(DEVICE)

opt = optim.Adam(net.parameters(), lr=1e-2)
max_epochs = 10000

for epoch in range(max_epochs):

    X, Y = train_x.to(DEVICE), train_y.to(DEVICE)

    out_Y = net(X)

    train_loss = nn.MSELoss()(out_Y, Y)

    opt.zero_grad()
    train_loss.backward()
    opt.step()

    if train_loss.item() < 1e-4:
        print(f'Epoch [{epoch+1}/{max_epochs}], Loss: {train_loss.item()}')
        print("The loss value is reached")
        break

    elif (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{max_epochs}], Loss: {train_loss.item()}')

torch.save(net.state_dict(), 'net.param')
