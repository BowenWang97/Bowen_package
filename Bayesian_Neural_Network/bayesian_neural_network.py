import pyro 
import torch
import torch.nn as nn
from pyro.nn import PyroModule
from torch.distributions import Normal

class one_layer_ANN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, nonlinear_layer_name = "sigmoid"):

        super(one_layer_ANN, self).__init__()

        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.nonlinear_layer_name = nonlinear_layer_name

        self.all_nonlinear_layer = {
            "sigmoid": nn.Sigmoid()
        }

    def forward(self, input):

        out = self.hidden(input)
        out = self.all_nonlinear_layer[self.nonlinear_layer_name](out)
        output = self.output(out)

        return output

class BayesianLinear(nn.Module):

    def __init__(self, input_size, output_size, prior_var = 1):

        super().__init__()

        self.weight_mu = nn.Parameter(torch.zeros(output_size, input_size))
        self.bias_mu =  nn.Parameter(torch.zeros(output_size))

        self.weight_sigma = nn.Parameter(torch.zeros(output_size, input_size))        
        self.bias_sigma = nn.Parameter(torch.zeros(output_size))

        self.weight = None
        self.bias = None

        self.weight_prior = Normal(0,prior_var)
        self.bias_prior = Normal(0,prior_var)

    def forward(self, input):

        weight_epsilon = Normal(0,1).sample(self.weight_mu.shape)
        bias_epsilon = Normal(0,1).sample(self.weight_mu.shape)

        self.weight = self.weight_mu + self.weight_sigma * weight_epsilon
        self.bias = self.bias_mu + self.bias_sigma * bias_epsilon

        output = torch.nn.functional(input, self.weight, self.bias)

        return output