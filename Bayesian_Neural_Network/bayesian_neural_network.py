import torch
import torch.nn as nn
from torch.distributions import Normal

class one_layer_ANN(nn.Module):

    def __init__(self, input_dimension, hidden_dimension, output_dimension, nonlinear_layer_name = "sigmoid"):

        super(one_layer_ANN, self).__init__()

        self.hidden = nn.Linear(input_dimension, hidden_dimension)
        self.output = nn.Linear(hidden_dimension, output_dimension)

        self.nonlinear_layer_name = nonlinear_layer_name

        self.all_nonlinear_layer = {
            "sigmoid": nn.Sigmoid(),
            "relu": nn.ReLU()
        }

    def forward(self, input):

        out = self.hidden(input)
        out = self.all_nonlinear_layer[self.nonlinear_layer_name](out)
        output = self.output(out)

        return output
    
class two_layer_ANN(nn.Module):

    def __init__(self, input_dimension, hidden_dimension, output_dimension, nonlinear_layer_name = ["sigmoid", "sigmoid"]):

        super(two_layer_ANN, self).__init__()

        self.hidden_1 = nn.Linear(input_dimension, hidden_dimension[0])
        self.hidden_2 = nn.Linear(hidden_dimension[0], hidden_dimension[1])
        self.output = nn.Linear(hidden_dimension[1], output_dimension)

        self.nonlinear_layer_name = nonlinear_layer_name

        self.all_nonlinear_layer = {
            "sigmoid": nn.Sigmoid(),
            "relu": nn.ReLU()
        }

    def forward(self, input):

        out = self.hidden_1(input)
        out = self.all_nonlinear_layer[self.nonlinear_layer_name[0]](out)
        out = self.hidden_2(out)
        out = self.all_nonlinear_layer[self.nonlinear_layer_name[1]](out)
        output = self.output(out)

        return output

class Bayesian_Layer(nn.Module):

    def __init__(self, input_dimension, output_dimension, prior_var = 1.):

        super(Bayesian_Layer, self).__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.weight_mu = nn.Parameter(torch.zeros(self.output_dimension, self.input_dimension))
        self.bias_mu =  nn.Parameter(torch.zeros(self.output_dimension))

        self.weight_rho = nn.Parameter(torch.zeros(self.output_dimension, self.input_dimension) * -3.)        
        self.bias_rho = nn.Parameter(torch.zeros(self.output_dimension) * -3.)

        self.weight = None
        self.bias = None

        self.weight_prior = Normal(0, prior_var)
        self.bias_prior = Normal(0, prior_var)

    def forward(self, input):

        weight_epsilon = Normal(0, 1).sample(self.weight_mu.shape)
        bias_epsilon = Normal(0, 1).sample(self.bias_mu.shape)

        self.weight = self.weight_mu + torch.log(1 + torch.exp(self.weight_rho)) * weight_epsilon
        self.bias = self.bias_mu + torch.log(1 + torch.exp(self.bias_rho)) * bias_epsilon

        weight_log_prior = self.weight_prior.log_prob(self.weight)
        bias_log_prior = self.bias_prior.log_prob(self.bias)
        self.log_prior = torch.sum(weight_log_prior) + torch.sum(bias_log_prior)

        self.weight_posterior = Normal(self.weight_mu.data, torch.log(1 + torch.exp(self.weight_rho)))
        self.bias_posterior = Normal(self.bias_mu.data, torch.log(1 + torch.exp(self.bias_rho)))
        self.log_posterior = self.weight_posterior.log_prob(self.weight).sum() + self.bias_posterior.log_prob(self.bias).sum()

        output = torch.nn.functional.linear(input, self.weight, self.bias)

        return output
    
class one_layer_BNN(nn.Module):

    def __init__(self, input_dimension, hidden_dimension, output_dimension, nonlinear_layer_name = "sigmoid", prior_var = 1.):

        super(one_layer_BNN, self).__init__()

        self.hidden = Bayesian_Layer(input_dimension, hidden_dimension, prior_var = prior_var)
        self.output = Bayesian_Layer(hidden_dimension, output_dimension, prior_var = prior_var)

        self.nonlinear_layer_name = nonlinear_layer_name

        self.all_nonlinear_layer = {
            "sigmoid": nn.Sigmoid(),
            "relu": nn.ReLU()
        }

    def forward(self, input):

        out = self.hidden(input)
        out = self.all_nonlinear_layer[self.nonlinear_layer_name](out)
        output = self.output(out)

        return output
    
    def log_prior(self):

        return self.hidden.log_prior + self.output.log_prior
    
    def log_posterior(self):

        return self.hidden.log_posterior + self.output.log_posterior
    
    def sample_evidence_lower_bound(self, input, output, sample_number, sample_noise = 0.1):

        sample_output = torch.zeros(sample_number, output.shape[0])
        sample_prior = torch.zeros(sample_number)
        sample_posterior = torch.zeros(sample_number)
        sample_likelihood = torch.zeros(sample_number)

        for n in range (sample_number):

            sample_output[n] = self(input).reshape(-1)
            sample_prior[n] = self.log_prior()
            sample_posterior[n] = self.log_posterior()
            sample_likelihood[n] = Normal(sample_output[n], sample_noise).log_prob(output.reshape(-1)).sum()

        log_prior = sample_prior.mean()
        log_posterior = sample_posterior.mean()
        log_likelihood = sample_likelihood.mean()

        loss = log_posterior - log_prior - log_likelihood

        return loss
    
class two_layer_BNN(nn.Module):

    def __init__(self, input_dimension, hidden_dimension, output_dimension, nonlinear_layer_name = ["sigmoid", "sigmoid"], prior_var = 1.):

        super(two_layer_BNN, self).__init__()

        self.hidden_1 = Bayesian_Layer(input_dimension, hidden_dimension[0], prior_var = prior_var)
        self.hidden_2 = Bayesian_Layer(hidden_dimension[0], hidden_dimension[1], prior_var = prior_var)
        self.output = Bayesian_Layer(hidden_dimension[1], output_dimension, prior_var = prior_var)

        self.nonlinear_layer_name = nonlinear_layer_name

        self.all_nonlinear_layer = {
            "sigmoid": nn.Sigmoid(),
            "relu": nn.ReLU()
        }

    def forward(self, input):

        out = self.hidden_1(input)
        out = self.all_nonlinear_layer[self.nonlinear_layer_name[0]](out)
        out = self.hidden_2(out)
        out = self.all_nonlinear_layer[self.nonlinear_layer_name[1]](out)
        output = self.output(out)

        return output
    
    def log_prior(self):

        return self.hidden_1.log_prior + self.hidden_2.log_prior + self.output.log_prior
    
    def log_posterior(self):

        return self.hidden_1.log_posterior + self.hidden_2.log_posterior + self.output.log_posterior
    
    def sample_evidence_lower_bound(self, input, output, sample_number, sample_noise = 0.1):

        sample_output = torch.zeros(sample_number, output.shape[0])
        sample_prior = torch.zeros(sample_number)
        sample_posterior = torch.zeros(sample_number)
        sample_likelihood = torch.zeros(sample_number)

        for n in range (sample_number):

            sample_output[n] = self(input).reshape(-1)
            sample_prior[n] = self.log_prior()
            sample_posterior[n] = self.log_posterior()
            sample_likelihood[n] = Normal(sample_output[n], sample_noise).log_prob(output.reshape(-1)).sum()

        log_prior = sample_prior.mean()
        log_posterior = sample_posterior.mean()
        log_likelihood = sample_likelihood.mean()

        loss = log_posterior - log_prior - log_likelihood

        return loss