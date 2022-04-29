import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

torch.manual_seed(1)

class LSTMTagger(nn.Module):

    def __init__(self, input_size = 6, hidden_size = 3, target_size = 1):

        super(LSTMTagger, self).__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_size, target_size)

    def forward(self, input, hidden):
        lstm_out, _ = self.lstm(input, hidden)
        tag_space = self.hidden2tag(lstm_out)
        # TODO: 需要之后把target_size变成某一个数值，把tag_size作为softmax的输出。
        # tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_space


def main():
    """
    Run some basic tests on the API
    """

    # define batch_size, channels, height, width
    b, c, h, w = 1, 3, 4, 8
    d = 5           # hidden state size
    max_epoch = 20  # number of epochs

    # set manual seed
    torch.manual_seed(1)

    print('Instantiate model')
    model = LSTMTagger(6, 3, 1)
    print(repr(model))

    # ---------------- Train Phase ---------------- #
    print('---------------- Starting Train Phase ----------------')

    print('Create input and target Variables')
    input = [Variable(torch.randn(1, 1, 6)) for _ in range(10)]
    hidden = (Variable(torch.randn(1, 1, 3)),
              Variable(torch.randn(1, 1, 3)))
    y = [Variable(torch.randn(1, 1, 1)) for _ in range(10)]

    print('Create a MSE criterion')
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    print('Run for', max_epoch, 'iterations')
    for epoch in range(max_epoch):  # again, normally you would NOT do 300 epochs, it is toy data
        for i in range(10):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network

            # Step 3. Run our forward pass.
            output = model(input[i], hidden)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(output, y[i])
            loss.backward()
            optimizer.step()

    # print('Input size:', list(x.data.size()))
    # print('Target size:', list(y.data.size()))
    # print('Last hidden state size:', list(state[0].size()))


    # ---------------- Test Phase ---------------- #
    print('---------------- Starting Test Phase ----------------')

    print('Create test_input and test_target Variables')
    test_input = [Variable(torch.randn(1, 1, 6)) for _ in range(10)]
    test_hidden = (Variable(torch.randn(1, 1, 3)),
              Variable(torch.randn(1, 1, 3)))
    test_y = [Variable(torch.randn(1, 1, 1)) for _ in range(10)]

    for i in range(10):
        test_output = model(test_input, test_hidden)
        loss += loss_function(test_output, test_y)
    loss = loss/10

if __name__ == '__main__':
    main()