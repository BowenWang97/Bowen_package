import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

class data_scaler():

    def __init__(self, input, output, predicted_input = None):

        super(data_scaler, self).__init__()

        self.input = input
        self.output = output
        self.predicted_input = predicted_input

    def standardscaler(self):

        self.input_mean = torch.mean(self.input, dim = 0, keepdim=True)
        self.input_std = torch.std(self.input, dim = 0, keepdim=True)
        self.output_mean = torch.mean(self.output, dim = 0, keepdim=True)
        self.output_std = torch.std(self.output, dim = 0, keepdim=True)

        scaler_input = (self.input - self.input_mean) / self.input_std
        scaler_output = (self.output - self.output_mean) / self.output_std

        if (self.predicted_input is not None):

            scaler_predicted_input = ( self.predicted_input - self.input_mean) / self.input_std        

            return scaler_input, scaler_output, scaler_predicted_input
        
        else:

            return scaler_input, scaler_output

    def inverse_standardscaler(self, scaler_predicted_output):

        predicted_output = scaler_predicted_output * self.output_std + self.output_mean

        return predicted_output
    
    def minmaxscaler(self, input_min, input_max):

        self.input_min = input_min
        self.input_max = input_max
        self.output_min = torch.min(self.output)
        self.output_max = torch.max(self.output)

        scaler_input = (self.input - self.input_min) / (self.input_max - self.input_min)
        scaler_output = (self.output - self.output_min) / (self.output_max - self.output_min)
        
        if (self.predicted_input is not None):

            scaler_predicted_input = (self.predicted_input - self.input_min) / (self.input_max - self.input_min)

            return scaler_input, scaler_output, scaler_predicted_input
        
        else:

            return scaler_input, scaler_output
    
    def inverse_minmaxscaler(self, scaler_predicted_output):

        predicted_output = scaler_predicted_output * (self.output_max - self.output_min) + self.output_min

        return predicted_output
    
    def inverse_minmaxscaler_noise(self, scaler_predicted_mean, scaler_predicted_noise):

        predicted_mean = scaler_predicted_mean * (self.output_max - self.output_min) + self.output_min
        predicted_noise = scaler_predicted_noise * (self.output_max - self.output_min)

        return predicted_mean, torch.abs(predicted_noise)
    
    def inverse_minmaxscaler_theta(self, scaler_weight, scaler_bias):

        weight = scaler_weight * (self.output_max - self.output_min) / (self.input_max - self.input_min)
        bias = scaler_bias * (self.output_max - self.output_min) + self.output_min - weight * self.input_min

        return weight, bias
    
    def minmax_input_standard_output_scaler(self, input_min, input_max):

        self.input_min = input_min
        self.input_max = input_max
        self.output_mean = torch.mean(self.output, dim = 0, keepdim=True)
        self.output_std = torch.std(self.output, dim = 0, keepdim=True)

        scaler_input = (self.input - self.input_min) / (self.input_max - self.input_min)
        scaler_output = (self.output - self.output_mean) / self.output_std

        if (self.predicted_input is not False):

            scaler_predicted_input = (self.predicted_input - self.input_min) / (self.input_max - self.input_min)        

            return scaler_input, scaler_output, scaler_predicted_input
        
        else:

            return scaler_input, scaler_output
    
    def inverse_minmax_input_standard_output_scaler(self, scaler_predicted_output):

        predicted_output = scaler_predicted_output * self.output_std + self.output_mean

        return predicted_output
    
    def inverse_minmax_input_standard_output_scaler_theta(self, scaler_weight, scaler_bias):

        weight = scaler_weight * self.output_std / (self.input_max - self.input_min)
        bias = scaler_bias * self.output_std + self.output_mean - weight * self.input_min

        return weight, bias

class linear_regression(nn.Module):

    def __init__(self):

        super(linear_regression, self).__init__()

        self.linear = nn.Linear(1, 1)
        
    def forward(self, input):

        return self.linear(input)
    
class linear_regression_with_one_layer_ANN(nn.Module):

    def __init__(self, input_dimension, hidden_dimension, output_dimension, nonlinear_layer_name = "relu"):

        super(linear_regression_with_one_layer_ANN, self).__init__()

        self.linear = nn.Linear(1, 1)

        self.hidden = nn.Linear(input_dimension, hidden_dimension)
        self.output = nn.Linear(hidden_dimension, output_dimension)

        self.nonlinear_layer_name = nonlinear_layer_name

        self.all_nonlinear_layer = {
            "sigmoid": nn.Sigmoid(),
            "relu": nn.ReLU()
        }

    def forward(self, input):

        out = self.hidden(input[:, 1:])
        out = self.all_nonlinear_layer[self.nonlinear_layer_name](out)
        out = self.output(out)

        output = out + self.linear(input[:, 0].unsqueeze(-1))

        return output
    
class linear_regression_with_two_layer_ANN(nn.Module):

    def __init__(self, input_dimension, hidden_dimension, output_dimension, nonlinear_layer_name = ["relu", "relu"]):

        super(linear_regression_with_two_layer_ANN, self).__init__()

        self.linear = nn.Linear(1, 1)

        self.hidden_1 = nn.Linear(input_dimension, hidden_dimension[0])
        self.hidden_2 = nn.Linear(hidden_dimension[0], hidden_dimension[1])
        self.output = nn.Linear(hidden_dimension[1], output_dimension)

        self.nonlinear_layer_name = nonlinear_layer_name

        self.all_nonlinear_layer = {
            "sigmoid": nn.Sigmoid(),
            "relu": nn.ReLU()
        }

    def forward(self, input):

        out = self.hidden_1(input[:, 1:])
        out = self.all_nonlinear_layer[self.nonlinear_layer_name[0]](out)
        out = self.hidden_2(out)
        out = self.all_nonlinear_layer[self.nonlinear_layer_name[1]](out)
        out = self.output(out)

        output = out + self.linear(input[:, 0].unsqueeze(-1))

        return output

class one_layer_ANN(nn.Module):
    
    def __init__(self, input_dimension, hidden_dimension, output_dimension, nonlinear_layer_name = "relu"):

        super(one_layer_ANN, self).__init__()

        self.hidden = nn.Linear(input_dimension, hidden_dimension)
        self.output = nn.Linear(hidden_dimension, output_dimension)

        self.nonlinear_layer_name = nonlinear_layer_name

        self.all_nonlinear_layer = {
            "sigmoid": nn.Sigmoid(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh()
        }   

    def forward(self, input):

        out = self.hidden(input)
        out = self.all_nonlinear_layer[self.nonlinear_layer_name](out)
        output = self.output(out)

        return output
    
class two_layer_ANN(nn.Module):

    def __init__(self, input_dimension, hidden_dimension, output_dimension, nonlinear_layer_name = ["relu", "relu"]):

        super(two_layer_ANN, self).__init__()

        self.hidden_1 = nn.Linear(input_dimension, hidden_dimension[0])
        self.hidden_2 = nn.Linear(hidden_dimension[0], hidden_dimension[1])
        self.output = nn.Linear(hidden_dimension[1], output_dimension)

        self.nonlinear_layer_name = nonlinear_layer_name

        self.all_nonlinear_layer = {
            "sigmoid": nn.Sigmoid(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "softplus": nn.Softplus(),
            "gelu": nn.GELU()
        }

    def forward(self, input):

        out = self.hidden_1(input)
        out = self.all_nonlinear_layer[self.nonlinear_layer_name[0]](out)
        out = self.hidden_2(out)
        out = self.all_nonlinear_layer[self.nonlinear_layer_name[1]](out)
        output = self.output(out)

        return output
    
class three_layer_ANN(nn.Module):

    def __init__(self, input_dimension, hidden_dimension, output_dimension, nonlinear_layer_name = ["relu", "relu", "relu"]):

        super(three_layer_ANN, self).__init__()

        self.hidden_1 = nn.Linear(input_dimension, hidden_dimension[0])
        self.hidden_2 = nn.Linear(hidden_dimension[0], hidden_dimension[1])
        self.hidden_3 = nn.Linear(hidden_dimension[1], hidden_dimension[2])
        self.output = nn.Linear(hidden_dimension[2], output_dimension)

        self.nonlinear_layer_name = nonlinear_layer_name

        self.all_nonlinear_layer = {
            "sigmoid": nn.Sigmoid(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "softplus": nn.Softplus(),
            "gelu": nn.GELU()
        }

    def forward(self, input):

        out = self.hidden_1(input)
        out = self.all_nonlinear_layer[self.nonlinear_layer_name[0]](out)
        out = self.hidden_2(out)
        out = self.all_nonlinear_layer[self.nonlinear_layer_name[1]](out)
        out = self.hidden_3(out)
        out = self.all_nonlinear_layer[self.nonlinear_layer_name[2]](out)
        output = self.output(out)

        return output
    
class three_layer_ANN_with_Sine(nn.Module):

    def __init__(self, input_dimension, hidden_dimension, output_dimension, omega = torch.pi):

        super(three_layer_ANN_with_Sine, self).__init__()

        self.hidden_1 = nn.Linear(input_dimension, hidden_dimension[0])
        self.hidden_2 = nn.Linear(hidden_dimension[0], hidden_dimension[1])
        self.hidden_3 = nn.Linear(hidden_dimension[1], hidden_dimension[2])
        self.output = nn.Linear(hidden_dimension[2], output_dimension)

        self.omega = omega

    def forward(self, input):

        out = self.hidden_1(input)
        out = torch.sin(self.omega * out)
        out = self.hidden_2(out)
        out = torch.relu(out)
        out = self.hidden_3(out)
        out = torch.relu(out)
        output = self.output(out)

        return output
    
class four_layer_ANN(nn.Module):

    def __init__(self, input_dimension, hidden_dimension, output_dimension, nonlinear_layer_name = ["relu", "relu", "relu", "relu"]):

        super(four_layer_ANN, self).__init__()

        self.hidden_1 = nn.Linear(input_dimension, hidden_dimension[0])
        self.hidden_2 = nn.Linear(hidden_dimension[0], hidden_dimension[1])
        self.hidden_3 = nn.Linear(hidden_dimension[1], hidden_dimension[2])
        self.hidden_4 = nn.Linear(hidden_dimension[2], hidden_dimension[3])
        self.output = nn.Linear(hidden_dimension[3], output_dimension)

        self.nonlinear_layer_name = nonlinear_layer_name

        self.all_nonlinear_layer = {
            "sigmoid": nn.Sigmoid(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "softplus": nn.Softplus(),
            "gelu": nn.GELU()
        }

    def forward(self, input):

        out = self.hidden_1(input)
        out = self.all_nonlinear_layer[self.nonlinear_layer_name[0]](out)
        out = self.hidden_2(out)
        out = self.all_nonlinear_layer[self.nonlinear_layer_name[1]](out)
        out = self.hidden_3(out)
        out = self.all_nonlinear_layer[self.nonlinear_layer_name[2]](out)
        out = self.hidden_4(out)
        out = self.all_nonlinear_layer[self.nonlinear_layer_name[3]](out)
        output = self.output(out)

        return output
    
class five_layer_ANN(nn.Module):

    def __init__(self, input_dimension, hidden_dimension, output_dimension, nonlinear_layer_name = ["relu", "relu", "relu", "relu", "relu"]):

        super(five_layer_ANN, self).__init__()

        self.hidden_1 = nn.Linear(input_dimension, hidden_dimension[0])
        self.hidden_2 = nn.Linear(hidden_dimension[0], hidden_dimension[1])
        self.hidden_3 = nn.Linear(hidden_dimension[1], hidden_dimension[2])
        self.hidden_4 = nn.Linear(hidden_dimension[2], hidden_dimension[3])
        self.hidden_5 = nn.Linear(hidden_dimension[3], hidden_dimension[4])
        self.output = nn.Linear(hidden_dimension[4], output_dimension)

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
        out = self.hidden_3(out)
        out = self.all_nonlinear_layer[self.nonlinear_layer_name[2]](out)
        out = self.hidden_4(out)
        out = self.all_nonlinear_layer[self.nonlinear_layer_name[3]](out)
        out = self.hidden_5(out)
        out = self.all_nonlinear_layer[self.nonlinear_layer_name[4]](out)
        output = self.output(out)

        return output

class Bayesian_Layer_VI(nn.Module):

    def __init__(self, input_dimension, output_dimension, prior_var = 1.):

        super(Bayesian_Layer_VI, self).__init__()

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.weight_mu = nn.Parameter(torch.zeros(self.output_dimension, self.input_dimension))
        self.bias_mu =  nn.Parameter(torch.zeros(self.output_dimension))

        self.weight_sigma = nn.Parameter(torch.ones(self.output_dimension, self.input_dimension) * -3.)        
        self.bias_sigma = nn.Parameter(torch.ones(self.output_dimension) * -3.)

        self.weight = None
        self.bias = None

        self.weight_prior = Normal(0, prior_var)
        self.bias_prior = Normal(0, prior_var)

    def forward(self, input):

        weight_epsilon = Normal(0, 1).sample(self.weight_mu.shape)
        bias_epsilon = Normal(0, 1).sample(self.bias_mu.shape)

        self.weight = self.weight_mu + torch.log(1 + torch.exp(self.weight_sigma)) * weight_epsilon
        self.bias = self.bias_mu + torch.log(1 + torch.exp(self.bias_sigma)) * bias_epsilon

        weight_log_prior = self.weight_prior.log_prob(self.weight)
        bias_log_prior = self.bias_prior.log_prob(self.bias)
        self.log_prior = torch.sum(weight_log_prior) + torch.sum(bias_log_prior)

        self.weight_posterior = Normal(self.weight_mu.data, torch.log(1 + torch.exp(self.weight_sigma)))
        self.bias_posterior = Normal(self.bias_mu.data, torch.log(1 + torch.exp(self.bias_sigma)))
        self.log_posterior = self.weight_posterior.log_prob(self.weight).sum() + self.bias_posterior.log_prob(self.bias).sum()

        output = torch.nn.functional.linear(input, self.weight, self.bias)

        return output
    
class one_layer_BNN_VI(nn.Module):

    def __init__(self, input_dimension, hidden_dimension, output_dimension, nonlinear_layer_name = "sigmoid", prior_var = 1.):

        super(one_layer_BNN_VI, self).__init__()

        self.hidden = Bayesian_Layer_VI(input_dimension, hidden_dimension, prior_var = prior_var)
        self.output = Bayesian_Layer_VI(hidden_dimension, output_dimension, prior_var = prior_var)

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
    
    def sample_evidence_lower_bound(self, input, output, sample_number, output_noise = 1.):

        sample_output = torch.zeros(sample_number, output.shape[0])
        sample_prior = torch.zeros(sample_number)
        sample_posterior = torch.zeros(sample_number)
        sample_likelihood = torch.zeros(sample_number)

        for n in range (sample_number):

            sample_output[n] = self(input).reshape(-1)
            sample_prior[n] = self.log_prior()
            sample_posterior[n] = self.log_posterior()
            sample_likelihood[n] = Normal(sample_output[n], output_noise).log_prob(output.reshape(-1)).sum()

        log_prior = sample_prior.mean()
        log_posterior = sample_posterior.mean()
        log_likelihood = sample_likelihood.mean()

        loss = log_posterior - log_prior - log_likelihood

        return loss
    
class two_layer_BNN_VI(nn.Module):

    def __init__(self, input_dimension, hidden_dimension, output_dimension, nonlinear_layer_name = ["sigmoid", "sigmoid"], prior_var = 1.):

        super(two_layer_BNN_VI, self).__init__()

        self.hidden_1 = Bayesian_Layer_VI(input_dimension, hidden_dimension[0], prior_var = prior_var)
        self.hidden_2 = Bayesian_Layer_VI(hidden_dimension[0], hidden_dimension[1], prior_var = prior_var)
        self.output = Bayesian_Layer_VI(hidden_dimension[1], output_dimension, prior_var = prior_var)

        self.nonlinear_layer_name = nonlinear_layer_name

        self.all_nonlinear_layer = {
            "sigmoid": nn.Sigmoid(),
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "softplus": nn.Softplus(),
            "gelu": nn.GELU()
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

class MCMC(nn.Module):

    def __init__(self, module, input, output, output_noise = 1., prior_sigma = 1., proposal_step = 0.1):

        super(MCMC, self).__init__()

        self.module = module
        self.input = input
        self.output = output
        self.output_noise = output_noise
        self.prior_sigma = prior_sigma
        self.proposal_step = proposal_step
        self.initial_theta = torch.cat([parameter.detach().clone().view(-1) for parameter in self.module.parameters()])
    
    def set_theta(self, theta):

        with torch.no_grad():

            assert theta.numel() == sum(p.numel() for p in self.module.parameters())

            theta_offset = 0

            for parameter in self.module.parameters():

                theta_number = parameter.numel()                

                parameter.copy_(theta[theta_offset : theta_offset+theta_number].view_as(parameter))

                theta_offset = theta_offset + theta_number

    def log_prior(self, theta):

        return -0.5 * torch.sum((theta / self.prior_sigma) **2)

    def log_likelihood(self):
        
        predict_output = self.module(self.input)

        n = self.output.numel()

        return -0.5 * torch.sum((self.output - predict_output) ** 2) / (self.output_noise ** 2) - 0.5 * n * torch.log(2 * torch.pi * (self.output_noise ** 2))
    
    def log_posterior(self, theta):
        
        self.set_theta(theta)

        return self.log_prior(theta) + self.log_likelihood()

    # def potential_energy_gradient(self, theta):

    #     theta = theta.detach().clone().requires_grad_()
    #     self.set_theta(theta)
        
    #     loss = -self.log_posterior(theta)
        
    #     loss.backward()
    #     gradient = theta.grad.clone()

    #     return gradient.detach()

    def potential_energy_gradient(self, theta):

        self.set_theta(theta)
        self.module.zero_grad()

        log_likelihood = self.log_likelihood()
        
        flat_params = torch.cat([p.view(-1) for p in self.module.parameters()])
        log_prior = self.log_prior(flat_params)
        
        loss = -(log_prior + log_likelihood)
        loss.backward()

        gradient = torch.cat([p.grad.clone().view(-1) for p in self.module.parameters()])

        return gradient.detach()
    
    def leapfrog(self, proposal_theta, proposal_momentum, direction):

        gradient = self.potential_energy_gradient(proposal_theta)
        proposal_momentum = proposal_momentum - 0.5 * direction * self.proposal_step * gradient

        proposal_theta = proposal_theta + direction * self.proposal_step * proposal_momentum

        gradient = self.potential_energy_gradient(proposal_theta)
        proposal_momentum = proposal_momentum - 0.5 * direction * self.proposal_step * gradient

        return proposal_theta, proposal_momentum, gradient

    def binary_tree_building(self, theta, momentum, gradient, depth, hamilton_threshold, direction):

        if (depth == 0):

            proposal_theta, proposal_momentum, gradient = self.leapfrog(theta, momentum, direction)

            hamilton = - self.log_posterior (proposal_theta) + 0.5 * torch.sum(proposal_momentum **2)

            valid = (hamilton_threshold <= -hamilton)

            return proposal_theta, proposal_momentum, gradient, proposal_theta, proposal_momentum, gradient, proposal_theta, valid, 1

        else:

            theta_minus, momentum_minus, gradient_minus, theta_plus, momentum_plus, gradient_plus, proposal_theta, valid_1, n1 = \
                self.binary_tree_building(theta, momentum, gradient, depth - 1, hamilton_threshold, direction)

            if (direction == -1):

                theta_minus, momentum_minus, gradient_minus, _, _, _, proposal_theta_2, valid_2, n2 = \
                    self.binary_tree_building(theta_minus, momentum_minus, gradient_minus, depth - 1, hamilton_threshold, direction)

            else:

                _, _, _, theta_plus, momentum_plus, gradient_plus, proposal_theta_2, valid_2, n2 = \
                    self.binary_tree_building(theta_plus, momentum_plus, gradient_plus, depth - 1, hamilton_threshold, direction)

            if ((n1 + n2) > 0):

                accept_ratio = n2 / (n1 + n2) 
                
            else:

                accept_ratio = 0

            if torch.rand(1) < accept_ratio:

                proposal_theta = proposal_theta_2

            valid = (valid_1 and valid_2 and not (torch.dot((theta_plus - theta_minus), momentum_minus) < 0 or torch.dot((theta_plus - theta_minus), momentum_plus) < 0))

            return theta_minus, momentum_minus, gradient_minus, theta_plus, momentum_plus, gradient_plus, proposal_theta, valid, n1+n2

    def metropolis_hasting(self, sample_number = 10000):

        theta_samples = []
        accept_count = 0

        current_theta = self.initial_theta
        current_log_posterior = self.log_posterior(current_theta)

        for n in range(sample_number):

            proposal_theta = current_theta + self.proposal_step * torch.randn_like(current_theta)

            proposal_log_posterior = self.log_posterior(proposal_theta)

            accept_ratio = torch.exp(proposal_log_posterior - current_log_posterior)
            accept_ratio = torch.clamp(accept_ratio, max = 1.0)

            if torch.rand(1) < accept_ratio:

                current_theta = proposal_theta
                current_log_posterior = proposal_log_posterior

                accept_count = accept_count + 1

            theta_samples.append(current_theta.clone())

            if ((n+1) % 1000 == 0):

                print(f"\rSample {n+1}, Acceptance Rate: {accept_count / (n+1):.3f}", end = '', flush = True)

        print()

        return theta_samples
    
    def hamiltonian_monte_carlo(self, sample_number = 10000, leapfrog_number = 10):

        theta_samples = []
        accept_count = 0

        current_theta = self.initial_theta 

        for n in range(sample_number):

            proposal_theta = current_theta.clone()
            current_momentum = torch.randn_like(current_theta)

            gradient = self.potential_energy_gradient(current_theta)
            proposal_momentum = current_momentum - 0.5 * self.proposal_step * gradient

            for lp in range(leapfrog_number):

                proposal_theta = proposal_theta + self.proposal_step * proposal_momentum

                gradient = self.potential_energy_gradient(proposal_theta)

                if (lp != leapfrog_number-1):

                    proposal_momentum = proposal_momentum - self.proposal_step * gradient

            proposal_momentum = proposal_momentum - 0.5 * self.proposal_step * gradient

            current_potential_energy = - self.log_posterior(current_theta)
            current_kinetic_energy = 0.5* torch.sum(current_momentum **2)

            proposal_potential_energy = - self.log_posterior(proposal_theta)
            proposal_kinetic_energy = 0.5* torch.sum(proposal_momentum **2)

            accept_ratio = torch.exp(current_potential_energy + current_kinetic_energy - proposal_potential_energy - proposal_kinetic_energy)
            accept_ratio = torch.clamp(accept_ratio, max = 1.0)

            if torch.rand(1) < accept_ratio:

                current_theta = proposal_theta
                current_momentum = proposal_momentum

                accept_count = accept_count + 1

            theta_samples.append(current_theta.clone())

            if ((n+1) % 1000 == 0):

                print(f"\rSample {n+1}, Acceptance Rate: {accept_count / (n+1):.3f}", end = '', flush = True)

        print()

        return theta_samples
    
    def no_u_turn_sampler(self, sample_number = 10000, max_depth = 10):

        theta_samples = []

        current_theta = self.initial_theta

        gradient = self.potential_energy_gradient(current_theta)

        for n in range(sample_number):

            current_momentum = torch.randn_like(current_theta)

            hamilton = - self.log_posterior (current_theta) + 0.5 * torch.sum(current_momentum **2)

            hamilton_threshold = torch.log(torch.rand(1)) - hamilton

            theta_minus = current_theta.clone()
            theta_plus = current_theta.clone()
            momentum_minus = current_momentum.clone()
            momentum_plus = current_momentum.clone()
            gradient_minus = gradient.clone()
            gradient_plus = gradient.clone()
            proposal_theta = current_theta.clone()
            depth = 0
            current_number = 1
            current_valid = True

            while (current_valid and (depth < max_depth)):

                direction = 2 * torch.randint(0, 2, ()).item() - 1

                if (direction == -1):

                    theta_minus, momentum_minus, gradient_minus, _, _, _, proposal_theta, valid, proposal_number = self.binary_tree_building(theta_minus, momentum_minus, gradient_minus, depth, hamilton_threshold, direction)

                else:

                    _, _, _, theta_plus, momentum_plus, gradient_plus, proposal_theta, valid, proposal_number = self.binary_tree_building(theta_plus, momentum_plus, gradient_plus, depth, hamilton_threshold, direction)

                if (valid and (torch.rand(1) < proposal_number/current_number)):

                    current_theta = proposal_theta

                current_number = current_number + proposal_number
                depth = depth +1
                current_valid = (valid and not (torch.dot((theta_plus - theta_minus), momentum_minus) < 0 or torch.dot((theta_plus - theta_minus), momentum_plus) < 0))

            gradient = self.potential_energy_gradient(current_theta)

            theta_samples.append(current_theta.clone())

            if ((n+1) % 1000 == 0):

                print(f"\rSample {n+1}, Acceptance Rate: {current_number / (n+1):.3f}", end = '', flush = True)

        print()

        return theta_samples
    
    def predict(self, input_test, theta_samples):

        predictions = []

        for parameter in theta_samples:

            self.set_theta(parameter)

            with torch.no_grad():

                predictions.append(self.module(input_test))
        
        return torch.stack(predictions)

class MCMC_heteroscedastic(nn.Module):

    def __init__(self, module, input, output, prior_sigma = 1., proposal_step = 0.1):

        super(MCMC_heteroscedastic, self).__init__()

        self.module = module
        self.input = input
        self.output = output
        self.prior_sigma = prior_sigma
        self.proposal_step = proposal_step
        self.initial_theta = torch.cat([parameter.detach().clone().view(-1) for parameter in self.module.parameters()])
    
    def set_theta(self, theta):

        with torch.no_grad():

            assert theta.numel() == sum(p.numel() for p in self.module.parameters())

            theta_offset = 0

            for parameter in self.module.parameters():

                theta_number = parameter.numel()                

                parameter.copy_(theta[theta_offset : theta_offset+theta_number].view_as(parameter))

                theta_offset = theta_offset + theta_number

    def log_prior(self, theta):

        return -0.5 * torch.sum((theta / self.prior_sigma) **2)

    def log_likelihood(self):
        
        predict_output = self.module(self.input)

        predict_mu = predict_output[:, 0].unsqueeze(-1)
        predict_sigma = predict_output[:, 1].unsqueeze(-1)

        log_l = -0.5 * (((self.output - predict_mu) ** 2) / (predict_sigma ** 2) + torch.log(2 * torch.pi * (predict_sigma ** 2)))

        return torch.sum(log_l)
    
    def log_posterior(self, theta):
        
        self.set_theta(theta)

        return self.log_prior(theta) + self.log_likelihood()

    # def potential_energy_gradient(self, theta):

    #     theta = theta.detach().clone().requires_grad_()
    #     self.set_theta(theta)
        
    #     loss = -self.log_posterior(theta)
        
    #     loss.backward()
    #     gradient = theta.grad.clone()

    #     return gradient.detach()

    def potential_energy_gradient(self, theta):

        self.set_theta(theta)
        self.module.zero_grad()

        log_likelihood = self.log_likelihood()
        
        flat_params = torch.cat([p.view(-1) for p in self.module.parameters()])
        log_prior = self.log_prior(flat_params)
        
        loss = -(log_prior + log_likelihood)
        loss.backward()

        gradient = torch.cat([p.grad.clone().view(-1) for p in self.module.parameters()])

        return gradient.detach()
    
    def leapfrog(self, proposal_theta, proposal_momentum, direction):

        gradient = self.potential_energy_gradient(proposal_theta)
        proposal_momentum = proposal_momentum - 0.5 * direction * self.proposal_step * gradient

        proposal_theta = proposal_theta + direction * self.proposal_step * proposal_momentum

        gradient = self.potential_energy_gradient(proposal_theta)
        proposal_momentum = proposal_momentum - 0.5 * direction * self.proposal_step * gradient

        return proposal_theta, proposal_momentum, gradient

    def binary_tree_building(self, theta, momentum, gradient, depth, hamilton_threshold, direction):

        if (depth == 0):

            proposal_theta, proposal_momentum, gradient = self.leapfrog(theta, momentum, direction)

            hamilton = - self.log_posterior (proposal_theta) + 0.5 * torch.sum(proposal_momentum **2)

            valid = (hamilton_threshold <= -hamilton)

            return proposal_theta, proposal_momentum, gradient, proposal_theta, proposal_momentum, gradient, proposal_theta, valid, 1

        else:

            theta_minus, momentum_minus, gradient_minus, theta_plus, momentum_plus, gradient_plus, proposal_theta, valid_1, n1 = \
                self.binary_tree_building(theta, momentum, gradient, depth - 1, hamilton_threshold, direction)

            if (direction == -1):

                theta_minus, momentum_minus, gradient_minus, _, _, _, proposal_theta_2, valid_2, n2 = \
                    self.binary_tree_building(theta_minus, momentum_minus, gradient_minus, depth - 1, hamilton_threshold, direction)

            else:

                _, _, _, theta_plus, momentum_plus, gradient_plus, proposal_theta_2, valid_2, n2 = \
                    self.binary_tree_building(theta_plus, momentum_plus, gradient_plus, depth - 1, hamilton_threshold, direction)

            if ((n1 + n2) > 0):

                accept_ratio = n2 / (n1 + n2) 
                
            else:

                accept_ratio = 0

            if torch.rand(1) < accept_ratio:

                proposal_theta = proposal_theta_2

            valid = (valid_1 and valid_2 and not (torch.dot((theta_plus - theta_minus), momentum_minus) < 0 or torch.dot((theta_plus - theta_minus), momentum_plus) < 0))

            return theta_minus, momentum_minus, gradient_minus, theta_plus, momentum_plus, gradient_plus, proposal_theta, valid, n1+n2

    def metropolis_hasting(self, sample_number = 10000):

        theta_samples = []
        accept_count = 0

        current_theta = self.initial_theta
        current_log_posterior = self.log_posterior(current_theta)

        for n in range(sample_number):

            proposal_theta = current_theta + self.proposal_step * torch.randn_like(current_theta)

            proposal_log_posterior = self.log_posterior(proposal_theta)

            accept_ratio = torch.exp(proposal_log_posterior - current_log_posterior)
            accept_ratio = torch.clamp(accept_ratio, max = 1.0)

            if torch.rand(1) < accept_ratio:

                current_theta = proposal_theta
                current_log_posterior = proposal_log_posterior

                accept_count = accept_count + 1

            theta_samples.append(current_theta.clone())

            if ((n+1) % 1000 == 0):

                print(f"\rSample {n+1}, Acceptance Rate: {accept_count / (n+1):.3f}", end = '', flush = True)

        print()

        return theta_samples
    
    def hamiltonian_monte_carlo(self, sample_number = 10000, leapfrog_number = 10):

        theta_samples = []
        accept_count = 0

        current_theta = self.initial_theta 

        for n in range(sample_number):

            proposal_theta = current_theta.clone()
            current_momentum = torch.randn_like(current_theta)

            gradient = self.potential_energy_gradient(current_theta)
            proposal_momentum = current_momentum - 0.5 * self.proposal_step * gradient

            for lp in range(leapfrog_number):

                proposal_theta = proposal_theta + self.proposal_step * proposal_momentum

                gradient = self.potential_energy_gradient(proposal_theta)

                if (lp != leapfrog_number-1):

                    proposal_momentum = proposal_momentum - self.proposal_step * gradient

            proposal_momentum = proposal_momentum - 0.5 * self.proposal_step * gradient

            current_potential_energy = - self.log_posterior(current_theta)
            current_kinetic_energy = 0.5* torch.sum(current_momentum **2)

            proposal_potential_energy = - self.log_posterior(proposal_theta)
            proposal_kinetic_energy = 0.5* torch.sum(proposal_momentum **2)

            accept_ratio = torch.exp(current_potential_energy + current_kinetic_energy - proposal_potential_energy - proposal_kinetic_energy)
            accept_ratio = torch.clamp(accept_ratio, max = 1.0)

            if torch.rand(1) < accept_ratio:

                current_theta = proposal_theta
                current_momentum = proposal_momentum

                accept_count = accept_count + 1

            theta_samples.append(current_theta.clone())

            if ((n+1) % 1000 == 0):

                print(f"\rSample {n+1}, Acceptance Rate: {accept_count / (n+1):.3f}", end = '', flush = True)

        print()

        return theta_samples
    
    def no_u_turn_sampler(self, sample_number = 10000, max_depth = 10):

        theta_samples = []

        current_theta = self.initial_theta

        gradient = self.potential_energy_gradient(current_theta)

        for n in range(sample_number):

            current_momentum = torch.randn_like(current_theta)

            hamilton = - self.log_posterior (current_theta) + 0.5 * torch.sum(current_momentum **2)

            hamilton_threshold = torch.log(torch.rand(1)) - hamilton

            theta_minus = current_theta.clone()
            theta_plus = current_theta.clone()
            momentum_minus = current_momentum.clone()
            momentum_plus = current_momentum.clone()
            gradient_minus = gradient.clone()
            gradient_plus = gradient.clone()
            proposal_theta = current_theta.clone()
            depth = 0
            current_number = 1
            current_valid = True

            while (current_valid and (depth < max_depth)):

                direction = 2 * torch.randint(0, 2, ()).item() - 1

                if (direction == -1):

                    theta_minus, momentum_minus, gradient_minus, _, _, _, proposal_theta, valid, proposal_number = self.binary_tree_building(theta_minus, momentum_minus, gradient_minus, depth, hamilton_threshold, direction)

                else:

                    _, _, _, theta_plus, momentum_plus, gradient_plus, proposal_theta, valid, proposal_number = self.binary_tree_building(theta_plus, momentum_plus, gradient_plus, depth, hamilton_threshold, direction)

                if (valid and (torch.rand(1) < proposal_number/current_number)):

                    current_theta = proposal_theta

                current_number = current_number + proposal_number
                depth = depth +1
                current_valid = (valid and not (torch.dot((theta_plus - theta_minus), momentum_minus) < 0 or torch.dot((theta_plus - theta_minus), momentum_plus) < 0))

            gradient = self.potential_energy_gradient(current_theta)

            theta_samples.append(current_theta.clone())

            if ((n+1) % 100 == 0):

                print(f"\rSample {n+1}, Acceptance Rate: {current_number / (n+1):.3f}", end = '', flush = True)

        print()

        return theta_samples
    
    def predict(self, input_test, theta_samples):

        predictions = []

        for parameter in theta_samples:

            self.set_theta(parameter)

            with torch.no_grad():

                predictions.append(self.module(input_test))
        
        return torch.stack(predictions)
    
    def predict_noise(self, input_test, theta_samples):

        predictions = []

        for parameter in theta_samples:

            self.set_theta(parameter)

            with torch.no_grad():

                output = self.module(input_test)

                output_mean = output[:, 0]
                output_noise = output[:, 1]

                outputs = output_mean + output_noise * torch.randn(1)

                predictions.append(outputs)
        
        return torch.stack(predictions)

class VI_MCMC(nn.Module):

    def __init__(self, vi_module, mcmc_module, input, output, output_noise = 1.):

        super(VI_MCMC, self).__init__()

        self.mcmc_module = mcmc_module
        self.input = input
        self.output = output
        self.output_noise = output_noise

        self.theta_mu = torch.cat([param.view(-1) for name, param in vi_module.named_parameters() if 'mu' in name])
        self.theta_sigma = torch.cat([param.view(-1) for name, param in vi_module.named_parameters() if 'sigma' in name])

        self.proposal_step = abs(0.1 * self.theta_sigma.mean())

    def set_theta(self, theta):

        with torch.no_grad():

            assert theta.numel() == sum(p.numel() for p in self.mcmc_module.parameters())

            theta_offset = 0

            for parameter in self.mcmc_module.parameters():

                theta_number = parameter.numel()                

                parameter.copy_(theta[theta_offset : theta_offset+theta_number].view_as(parameter))

                theta_offset = theta_offset + theta_number

    def log_prior(self, theta):

        return -0.5 * torch.sum((theta / self.theta_sigma) **2)

    def log_likelihood(self):
        
        predict_output = self.mcmc_module(self.input)

        n = self.output.numel()

        return -0.5 * torch.sum((self.output - predict_output) ** 2) / self.output_noise ** 2 - 0.5 * n * torch.log(torch.tensor(2 * torch.pi * self.output_noise ** 2))
    
    def log_posterior(self, theta):
        
        self.set_theta(theta)

        return self.log_prior(theta) + self.log_likelihood()

    def metropolis_hasting(self, sample_number = 10000):

        theta_samples = []
        accept_count = 0

        current_theta = self.theta_mu
        current_log_posterior = self.log_posterior(current_theta)

        for n in range(sample_number):

            proposal_theta = current_theta + self.proposal_step * torch.randn_like(current_theta)

            proposal_log_posterior = self.log_posterior(proposal_theta)

            accept_ratio = torch.exp(proposal_log_posterior - current_log_posterior)
            accept_ratio = torch.clamp(accept_ratio, max = 1.0)

            if torch.rand(1) < accept_ratio:

                current_theta = proposal_theta
                current_log_posterior = proposal_log_posterior

                accept_count = accept_count + 1

            theta_samples.append(current_theta.clone())

            if (n % 100 == 0):

                print(f"Sample {n}, Acceptance Rate: {accept_count / (n+1):.3f}")

        return theta_samples
    
    def predict(self, input_test, theta_samples):

        predictions = []

        for parameter in theta_samples:

            self.set_theta(parameter)

            with torch.no_grad():

                predictions.append(self.mcmc_module(input_test))

        return predictions

# class VI(nn.Module):

#     def __init__(self, module, input, output, output_noise = 1., prior_sigma = 1.):

#         super(VI, self).__init__()

#         self.module = module
#         self.input = input
#         self.output = output
#         self.output_noise = output_noise
#         self.prior_sigma = prior_sigma

#         # self.means = nn.Parameter(torch.cat([torch.randn_like(parameter).detach().clone().view(-1) for parameter in module.parameters()]))
#         # self.log_std = nn.Parameter(torch.ones_like(self.means) * -3.)

#     def sample_parameter(self):

#         means = nn.Parameter(torch.cat([torch.randn_like(parameter).detach().clone().view(-1) for parameter in self.module.parameters()]))
#         log_std = nn.Parameter(torch.ones_like(means) * -3.)

#         epsilon = torch.randn_like(log_std)

#         theta_sample = means + torch.exp(log_std) * epsilon

#         return theta_sample
    
#     def pack_theta(self, theta):

#         theta_dict = {}
#         theta_offset = 0

#         for name, parameter in self.module.named_parameters():
            
#             numel = parameter.numel()
#             theta_dict[name] = theta[theta_offset:theta_offset + numel].view_as(parameter)
#             theta_offset += numel

#         return theta_dict

#     def evidence_lower_bound(self, sample_number = 5):

#         loss = 0

#         for _ in range(sample_number):
            
#             theta_sample = self.sample_parameter()

#             # theta_dict = self.pack_theta(theta_sample)

#             predict_output = self.module(self.input)
#             n = self.output.numel()

#             means = nn.Parameter(torch.cat([torch.randn_like(parameter).detach().clone().view(-1) for parameter in self.module.parameters()]))
#             log_std = nn.Parameter(torch.ones_like(means) * -3.)

#             log_likelihood = -0.5 * torch.sum((self.output - predict_output) ** 2) / self.output_noise ** 2 - 0.5 * n * torch.log(torch.tensor(2 * torch.pi * self.output_noise ** 2))
#             log_prior = -0.5 * torch.sum((theta_sample / self.prior_sigma) **2)
#             log_q = -0.5 * torch.sum(((theta_sample - means) / (torch.exp(log_std))) **2 + 2 * log_std)

#             loss = loss + log_q - log_prior - log_likelihood

#         return loss / sample_number

# class VI_MCMC(nn.Module):

#     def __init__(self, module, input, output, output_noise = 1., prior_sigma = 1., proposal_step = 0.1):

#         super(VI_MCMC, self).__init__()

#         self.module = module
#         self.input = input
#         self.output = output
#         self.output_noise = output_noise
#         self.prior_sigma = prior_sigma
#         self.proposal_step = proposal_step

#         # self.means = nn.Parameter(torch.cat([torch.randn_like(parameter).detach().clone().view(-1) for parameter in module.parameters()]))
#         # self.log_std = nn.Parameter(torch.ones_like(self.means) * -3.)

#     def sample_parameter(self):

#         means = nn.Parameter(torch.cat([torch.randn_like(parameter).detach().clone().view(-1) for parameter in self.module.parameters()]))
#         log_std = nn.Parameter(torch.ones_like(means) * -3.)

#         epsilon = torch.randn_like(log_std)

#         theta_sample = means + torch.exp(log_std) * epsilon

#         return theta_sample
    
#     def set_theta(self, theta):

#         with torch.no_grad():

#             assert theta.numel() == sum(p.numel() for p in self.module.parameters())

#             theta_offset = 0

#             for parameter in self.module.parameters():

#                 theta_number = parameter.numel()                

#                 parameter.copy_(theta[theta_offset : theta_offset+theta_number].view_as(parameter))

#                 theta_offset = theta_offset + theta_number

#     def log_prior(self, theta):

#         return -0.5 * torch.sum((theta / self.prior_sigma) **2)

#     def log_likelihood(self):
        
#         predict_output = self.module(self.input)

#         n = self.output.numel()

#         return -0.5 * torch.sum((self.output - predict_output) ** 2) / self.output_noise ** 2 - 0.5 * n * torch.log(torch.tensor(2 * torch.pi * self.output_noise ** 2))
    
#     def log_posterior(self, theta):

#         self.set_theta(theta)

#         return self.log_prior(theta) + self.log_likelihood()

#     def evidence_lower_bound(self, sample_number = 5):

#         loss = 0

#         for _ in range(sample_number):
            
#             theta_sample = self.sample_parameter()

#             means = nn.Parameter(torch.cat([torch.randn_like(parameter).detach().clone().view(-1) for parameter in self.module.parameters()]))
#             log_std = nn.Parameter(torch.ones_like(means) * -3.)

#             log_likelihood = self.log_likelihood()
#             log_prior = self.log_prior(theta_sample)
#             log_posterior = -0.5 * torch.sum(((theta_sample - means) / (torch.exp(log_std))) **2 + 2 * log_std)

#             loss = loss + log_posterior - log_prior - log_likelihood

#         return loss / sample_number
    
#     def metropolis_hasting(self, initial_theta, sample_number = 1000):

#         theta_samples = []
#         accept_count = 0

#         current_theta = initial_theta
#         current_log_posterior = self.log_posterior(current_theta)

#         for n in range(sample_number):

#             proposal_theta = current_theta + self.proposal_step * torch.randn_like(current_theta)

#             proposal_log_posterior = self.log_posterior(proposal_theta)

#             accept_ratio = torch.exp(proposal_log_posterior - current_log_posterior)
#             accept_ratio = torch.clamp(accept_ratio, max = 1.0)

#             if torch.rand(1) < accept_ratio:

#                 current_theta = proposal_theta
#                 current_log_posterior = proposal_log_posterior

#                 accept_count = accept_count + 1

#             theta_samples.append(current_theta.clone())

#             if (n % 100 == 0):

#                 print(f"Sample {n}, Acceptance Rate: {accept_count / (n+1):.3f}")

#         return theta_samples
    
#     def predict(self, input_test, theta_samples):

#         predictions = []

#         for parameter in theta_samples:

#             self.set_theta(parameter)

#             with torch.no_grad():

#                 predictions.append(self.module(input_test).cpu().numpy())

#         return predictions