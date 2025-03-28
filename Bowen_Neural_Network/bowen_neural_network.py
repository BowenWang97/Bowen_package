import numpy as np
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

    def __init__(self, input_dimension, hidden_dimension, output_dimension, nonlinear_layer_name = ["relu", "sigmoid"]):

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
    
class two_layer_BNN_VI(nn.Module):

    def __init__(self, input_dimension, hidden_dimension, output_dimension, nonlinear_layer_name = ["sigmoid", "sigmoid"], prior_var = 1.):

        super(two_layer_BNN_VI, self).__init__()

        self.hidden_1 = Bayesian_Layer_VI(input_dimension, hidden_dimension[0], prior_var = prior_var)
        self.hidden_2 = Bayesian_Layer_VI(hidden_dimension[0], hidden_dimension[1], prior_var = prior_var)
        self.output = Bayesian_Layer_VI(hidden_dimension[1], output_dimension, prior_var = prior_var)

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
    
class MCMC(nn.Module):

    def __init__(self, module, input, output, output_noise = 1., prior_sigma = 1., proposal_step = 0.01):

        super(MCMC, self).__init__()

        self.module = module
        self.input = input
        self.output = output
        self.output_noise = output_noise
        self.prior_sigma = prior_sigma
        self.proposal_step = proposal_step
        self.initial_theta = torch.cat([parameter.detach().view(-1) for parameter in self.module.parameters()])
    
    def set_theta(self, theta):

        theta_offset = 0

        for parameter in self.module.parameters():

            theta_number = parameter.numel()

            with torch.no_grad():

                parameter.copy_(theta[theta_offset : theta_offset+theta_number].view(parameter.size()).clone())

            theta_offset = theta_offset + theta_number

    def log_prior(self, theta):

        return -0.5 * torch.sum((theta / self.prior_sigma) * (theta / self.prior_sigma))

    def log_likelihood(self):
        
        predict_output = self.module(self.input)

        return -0.5 * torch.sum((self.output - predict_output) * (self.output - predict_output)) / self.output_noise / self.output_noise
    
    def log_posterior(self, theta):
        
        self.set_theta(theta)

        return self.log_prior(theta) + self.log_likelihood()
    
    def potential_energy_gradient(self, theta):

        self.module.zero_grad()
        loss = -self.log_posterior(theta)
        loss.backward()
        gradient = []

        for parameter in self.module.parameters():

            gradient.append(parameter.grad.clone())

        return gradient
    
    def leapfrog(self, proposal_theta, proposal_momentum):

        gradient = torch.cat([grad.detach().view(-1) for grad in self.potential_energy_gradient(proposal_theta)])
        proposal_momentum = proposal_momentum - 0.5 * self.proposal_step * gradient

        proposal_theta = proposal_theta + self.proposal_step * proposal_momentum

        gradient = torch.cat([grad.detach().view(-1) for grad in self.potential_energy_gradient(proposal_theta)])
        proposal_momentum = proposal_momentum - 0.5 * self.proposal_step * gradient

        return proposal_theta, proposal_momentum, gradient

    def tree_building(self, theta, momentum, gradient, depth, hamilton_threshold, direction, max_delta = 1000.):

        if (depth == 0):

            proposal_theta, proposal_momentum, gradient = self.leapfrog(theta, momentum)

            hamilton = - self.log_posterior (proposal_theta) + 0.5 * torch.sum(proposal_momentum * proposal_momentum)

            valid = (hamilton_threshold <= torch.exp(-hamilton))

            return proposal_theta, proposal_momentum, gradient, proposal_theta, proposal_momentum, gradient, proposal_theta, valid, 1

        else:

            theta_minus, momentum_minus, gradient_minus, theta_plus, momentum_plus, gradient_plus, proposal_theta, valid_1, n1 = self.tree_building(theta, momentum, gradient, depth - 1, hamilton_threshold, direction)

            if (direction == -1):

                theta_minus, momentum_minus, gradient_minus, _, _, _, proposal_theta_1, valid_2, n2 = self.tree_building(theta_minus, momentum_minus, gradient_minus, depth - 1, hamilton_threshold, direction)

            else:

                _, _, _, theta_plus, momentum_plus, gradient_plus, proposal_theta_1, valid_2, n2 = self.tree_building(theta_plus, momentum_plus, gradient_plus, depth - 1, hamilton_threshold, direction)

            if ((n1 + n2) > 0):

                accept_ratio = n2 / (n1 + n2) 
                
            else:

                accept_ratio = 0

            if torch.randn(1) < accept_ratio:

                proposal_theta = proposal_theta_1

            valid = (valid_1 and valid_2 and not (torch.dot((theta_plus - theta_minus), momentum_minus) < 0 or torch.dot((theta_plus - theta_minus), momentum_plus) < 0))

            return theta_minus, momentum_minus, gradient_minus, theta_plus, momentum_plus, gradient_plus, proposal_theta, valid, n1+n2

    def metropolis_hasting(self, sample_number = 10000):

        self.samples = []
        accept_count = 0

        current_theta = self.initial_theta
        current_log_posterior = self.log_posterior(current_theta)

        for n in range(sample_number):

            proposal_theta = current_theta + self.proposal_step * torch.randn_like(current_theta)

            proposal_log_posterior = self.log_posterior(proposal_theta)

            accept_ratio = torch.exp(proposal_log_posterior - current_log_posterior)

            if torch.randn(1) < accept_ratio:

                current_theta = proposal_theta
                current_log_posterior = proposal_log_posterior

                accept_count = accept_count + 1

            self.samples.append(current_theta.clone())

            if (n % 100 == 0):

                print(f"Sample {n}, Acceptance Rate: {accept_count / (n+1):.3f}")

        return self.samples
    
    def hamiltonian_monte_carlo(self, sample_number = 10000, leapfrog_number = 10):

        self.samples = []
        accept_count = 0

        current_theta = self.initial_theta        

        for n in range(sample_number):

            proposal_theta = current_theta
            current_momentum = torch.randn_like(current_theta)

            gradient = torch.cat([grad.detach().view(-1) for grad in self.potential_energy_gradient(current_theta)])
            proposal_momentum = current_momentum - 0.5 * self.proposal_step * gradient

            for _ in range(leapfrog_number):

                proposal_theta = proposal_theta + self.proposal_step * proposal_momentum

                gradient = torch.cat([grad.detach().view(-1) for grad in self.potential_energy_gradient(proposal_theta)])
                proposal_momentum = proposal_momentum - self.proposal_step * gradient

            proposal_momentum = proposal_momentum -  0.5 * self.proposal_step * gradient
            proposal_momentum = - proposal_momentum

            current_potential_energy = - self.log_posterior(current_theta)
            current_kinetic_energy = 0.5* torch.sum(current_momentum * current_momentum)

            proposal_potential_energy = - self.log_posterior(proposal_theta)
            proposal_kinetic_energy = 0.5* torch.sum(proposal_momentum * proposal_momentum)

            accept_ratio = torch.exp(current_potential_energy + current_kinetic_energy - proposal_potential_energy - proposal_kinetic_energy)

            if torch.randn(1) < accept_ratio:

                current_theta = proposal_theta
                current_momentum = proposal_momentum

                accept_count = accept_count + 1

            self.samples.append(current_theta.clone())

            if (n % 100 == 0):

                print(f"Sample {n}, Acceptance Rate: {accept_count / (n+1):.3f}")

        return self.samples
    
    def no_u_turn_sampler(self, sample_number = 10000, max_depth = 5):

        self.samples = []

        current_theta = self.initial_theta

        gradient = torch.cat([grad.detach().view(-1) for grad in self.potential_energy_gradient(current_theta)])

        for n in range(sample_number):

            current_momentum = torch.randn_like(current_theta)

            hamilton = - self.log_posterior (current_theta) + 0.5 * torch.sum(current_momentum * current_momentum)

            hamilton_threshold = torch.torch.randn(1).log().item() - hamilton.item()

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

                direction = torch.randint([-1, 1]).item()

                if (direction == -1):

                    theta_minus, momentum_minus, gradient_minus, _, _, _, proposal_theta, valid, proposal_number = self.tree_building(theta_minus, momentum_minus, gradient_minus, depth - 1, hamilton_threshold, direction)

                else:

                    _, _, _, theta_plus, momentum_plus, gradient_plus, proposal_theta, valid, proposal_number = self.tree_building(theta_plus, momentum_plus, gradient_plus, depth - 1, hamilton_threshold, direction)

                current_number = current_number + proposal_number
                depth = depth +1
                current_valid = (valid and not (torch.dot((theta_plus - theta_minus), momentum_minus) < 0 or torch.dot((theta_plus - theta_minus), momentum_plus) < 0))

            current_theta = proposal_theta.clone()
            gradient = torch.cat([grad.detach().view(-1) for grad in self.potential_energy_gradient(current_theta)])

            self.samples.append(current_theta.clone())

            if (n % 100 == 0):

                print(f"Sample {n}")

            return self.samples
    
    def predict(self, input_test):

        predictions = []

        for parameter in self.samples[::10]:

            self.set_theta(parameter)

            with torch.no_grad():

                predictions.append(self.module(input_test).numpy())

        return predictions