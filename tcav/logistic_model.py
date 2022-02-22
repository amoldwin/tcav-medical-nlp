import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch import nn, optim
from torch.nn import functional as F

# Defining neural network structure
class BoWClassifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_labels, vocab_size):
        # needs to be done everytime in the nn.module derived class
        super(BoWClassifier, self).__init__()

        # Define the parameters that are needed for linear model ( Ax + b)
        self.linear = nn.Linear(vocab_size, num_labels)

        # NOTE! The non-linearity log softmax does not have parameters! So we don't need
        # to worry about that here

    def forward(self, bow_vec): # Defines the computation performed at every call.
        # Pass the input through the linear layer,
        # then pass that through log_softmax.

        return F.log_softmax(self.linear(bow_vec), dim=1)