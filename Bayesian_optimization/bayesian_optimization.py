import numpy as np
import torch
import torch.nn as nn
import scipy as sci

class data_scaler():

    def __init__(self, input_min, input_max):

        super(data_scaler, self).__init__()

        self.input_min = input_min
        self.input_max = input_max
    
    def scaler(self, input, output):

        scaler_input = (input - self.input_min) / (self.input_max - self.input_min)

        output_min = torch.min(output)
        output_max = torch.max(output)

        if (output_max == output_min):

            scaler_output = torch.ones_like(output)

        else:

            scaler_output = (output - output_min) / (output_max - output_min)

        return scaler_input, scaler_output
    
    def inverse_scaler(self, scaler_input):

        input = (scaler_input * (self.input_max - self.input_min)) + self.input_min

        return input

class kernel_function():

    def __init__(self, data, length_scale = None, kernel_name = "matern_nu_5", amplitude = torch.tensor(1.)):

        super(kernel_function, self).__init__()
    
        self.data = data
        self.kernel_name = kernel_name

        self.data_dimension = self.data.size()
        self.data_number = self.data_dimension[0]
        self.input_dimension = self.data_dimension[1] - 1

        self.amplitude = amplitude

        if (length_scale == None):
        
            self.length_scale = torch.ones(self.input_dimension)

        else:

            self.length_scale = length_scale

        self.input = data[:, :-1]
        self.output = data[:, self.input_dimension]
        
        self.all_kernel_function = {
            "matern_nu_1" : self.matern_nu_1,
            "matern_nu_3" : self.matern_nu_3,
            "matern_nu_5" : self.matern_nu_5,
            "squared_exponential" : self.squared_exponential
        }

    def output_amplitude(self):

        if (self.data_number == 1):

            self.amplitude = torch.tensor(1.)

        else:

            self.amplitude = torch.std(self.output)
    
    def matern_nu_1(self, relenvant_distance):

        k = self.amplitude **2 * torch.exp( -relenvant_distance)

        return k
    
    def matern_nu_3(self, relenvant_distance):

        k = self.amplitude **2 * (1 + torch.sqrt(torch.tensor(3.)) * relenvant_distance) * torch.exp( - torch.sqrt(torch.tensor(3.)) * relenvant_distance)

        return k
    
    def matern_nu_5(self, relenvant_distance):

        k = self.amplitude **2 * (1 + torch.sqrt(torch.tensor(5.)) * relenvant_distance + 5 * relenvant_distance **2 / torch.tensor(3.)) * torch.exp( - torch.sqrt(torch.tensor(5.)) * relenvant_distance)

        return k
    
    def squared_exponential(self, relenvant_distance):

        k = self.amplitude **2 * torch.exp( - relenvant_distance **2 / torch.tensor(2.))

        return k
    
    def kernel_data_sample(self):

        self.relenvant_distance_data = torch.zeros(self.data_number, self.data_number)

        for d in range(self.input_dimension):

            self.relenvant_distance_data[:, :] = self.relenvant_distance_data[:, :] + torch.abs(self.input[:, d].expand(self.data_number, self.data_number) - self.input[:, d].expand(self.data_number, self.data_number).transpose(0, 1)) / self.length_scale[d]

        self.kernel_data_matrix = self.all_kernel_function[self.kernel_name](self.relenvant_distance_data)
        self.kernel_sample_matrix = self.all_kernel_function[self.kernel_name](torch.tensor(0.))

    def kernel_matrix(self, sample):

        relenvant_distance_sample = torch.zeros(self.data_number)

        if (self.input_dimension == 1):

            relenvant_distance_sample = torch.abs(self.input[:, 0] - sample) / self.length_scale

        else:

            for d in range(self.input_dimension):

                relenvant_distance_sample = relenvant_distance_sample + torch.abs(self.input[:, d] - sample[d]) / self.length_scale[d]

        kernel_data_sample_matrix = self.all_kernel_function[self.kernel_name](relenvant_distance_sample)        
   
        return kernel_data_sample_matrix
    
    def return_self(self):

        return self
    
class gaussian_progress():

    def __init__(self, kernel_self, sample,  data_noise = None, acquisition_function_name = "expected_improvement", xi = torch.tensor(1.), kappa = torch.tensor(2.)):

        super(gaussian_progress, self).__init__()

        self.all_kernel_function = kernel_self.all_kernel_function
        self.data_number = kernel_self.data_number
        self.kernel_data_matrix = kernel_self.kernel_data_matrix
        self.kernel_sample_matrix = kernel_self.kernel_sample_matrix
        self.input = kernel_self.input
        self.input_dimension = kernel_self.input_dimension
        self.kernel_name = kernel_self.kernel_name
        self.length_scale = kernel_self.length_scale
        self.output = kernel_self.output

        self.sample = sample
        self.acquisition_function_name = acquisition_function_name

        if (data_noise == None):

            self.data_sigma = torch.zeros(self.data_number)

        else:

            self.data_sigma = data_noise

        self.sample_number = self.sample.size()[0]

        self.xi = xi
        self.kappa = kappa

        self.all_acquisition_function = {
            "expected_improvement" : self.expected_improvement,
            "probability_of_improvement" : self.probability_of_improvement,
            "upper_confidence_bound" : self.upper_confidence_bound
        }       

    def fit_function(self):

        self.mu = torch.zeros(self.sample_number)
        self.sigma = torch.zeros(self.sample_number)

        for n in range(self.sample_number):

            kernel_data_sample_matrix = kernel_function.kernel_matrix(self, self.sample[n])
            
            self.mu[n] = torch.linalg.multi_dot((kernel_data_sample_matrix, torch.linalg.inv(self.kernel_data_matrix + self.data_sigma * torch.eye(self.data_number)), self.output))
            self.sigma[n] = self.kernel_sample_matrix - torch.linalg.multi_dot((kernel_data_sample_matrix, torch.linalg.inv(self.kernel_data_matrix + self.data_sigma * torch.eye(self.data_number)), torch.unsqueeze(kernel_data_sample_matrix, 1)))

        self.sigma = torch.clamp(self.sigma, min = 0.)

        return self.mu, self.sigma
    
    def save_self(self):

        return self

    def fit_function_multi(self_save, index, mu, sigma):

        kernel_data_sample_matrix = kernel_function.kernel_matrix(self_save, self_save.sample[index])

        mu[index] = torch.linalg.multi_dot((kernel_data_sample_matrix, torch.linalg.inv(self_save.kernel_data_matrix + self_save.data_sigma * torch.eye(self_save.data_number)), self_save.output)).item()
        sigma[index] = self_save.kernel_sample_matrix - torch.linalg.multi_dot((kernel_data_sample_matrix, torch.linalg.inv(self_save.kernel_data_matrix + self_save.data_sigma * torch.eye(self_save.data_number)), torch.unsqueeze(kernel_data_sample_matrix, 1))).item()

    def save_ff_multi(self, mu, sigma):

        self.mu = torch.tensor(list(mu))
        self.sigma = torch.tensor(list(sigma))

        return self.mu, self.sigma
    
    def standard_normal_pdf(self, x):

        pdf = torch.exp( - x * x / 2) / torch.sqrt(torch.tensor(2.) * torch.pi)

        return pdf
    
    def standard_normal_cdf(self, x):

        cdf = (1 + torch.erf(x / torch.sqrt(torch.tensor(2.)))) / 2

        return cdf
    
    def standardized_improvement(self):

        z = (self.mu - torch.max(self.output) - self.xi) / self.sigma

        return z

    def expected_improvement(self):

        z = self.standardized_improvement()

        ei = (self.mu - torch.max(self.output) - self.xi) * torch.tensor(sci.stats.norm.cdf(z)) + self.sigma * torch.tensor(sci.stats.norm.pdf(z))

        return ei
    
    def probability_of_improvement(self):

        z = self.standardized_improvement()

        pi = torch.tensor(sci.stats.norm.cdf(z))

        return pi
    
    def upper_confidence_bound(self):

        ucb = self.mu - self.kappa * self.sigma

        return ucb
    
    def acquisition_function(self):

        af = self.all_acquisition_function[self.acquisition_function_name]()

        return af
    
    def return_self(self):

        return self
    
class sampling():

    def __init__(self, gaussian_progress_self):

        super(sampling, self).__init__()

        self.acquisition_function_name = gaussian_progress_self.acquisition_function_name
        self.all_acquisition_function = gaussian_progress_self.all_acquisition_function
        self.data_number = gaussian_progress_self.data_number
        self.input_dimension = gaussian_progress_self.input_dimension
        self.sample = gaussian_progress_self.sample
    
    def next_sample(self, next_sample_number, initial_input):

        initial_next_sample_number = next_sample_number

        af = gaussian_progress.acquisition_function(self)

        ns = torch.zeros(next_sample_number, self.input_dimension)

        loop_state = True
        loop_time = 0

        while(loop_state):

            _, indices = torch.topk(af, next_sample_number)

            ns_0 = torch.zeros(next_sample_number, self.input_dimension)

            index = []

            for n in range(next_sample_number):

                if (self.input_dimension == 1):

                    ns_0[n][0] = self.sample[indices[n]]

                else:

                    for d in range(self.input_dimension):

                        ns_0[n][d] = self.sample[indices[n]][d]

            if_state = [True] * indices.shape[0]

            for i in range(indices.shape[0]):

                eq = torch.eq(initial_input, ns_0[i])

                for n in range(self.data_number):

                    if_state[i] = True

                    for d in range(self.input_dimension):

                        if (if_state and eq[n][d]):

                            if_state[i] = True

                        else:

                            if_state[i] = False                            

                            break

                    if (if_state[i]):

                        next_sample_number = initial_next_sample_number + loop_time

                        break

                if not (all(if_state)):

                    loop_state = False

                    index.append(indices[i])

            loop_time = loop_time + 1

        for n in range(initial_next_sample_number):

            if (self.input_dimension == 1):

                    ns[n][0] = self.sample[int(index[n])]

            else:

                for d in range(self.input_dimension):

                    ns[n][d] = self.sample[int(index[n])][d]

        return ns
    
class GP_nn(nn.Module):

    def __init__(self, data, data_noise = torch.tensor([1e-4]), kernel_name = "matern_nu_5", acquisition_function_name = "upper_confidence_bound", xi = torch.tensor(1.), kappa = torch.tensor(2.), lengthscale = None, variance = None):

        super(GP_nn, self).__init__()

        self.data = data
        self.data_noise = data_noise
        self.kernel_name = kernel_name
        self.acquisition_function_name = acquisition_function_name
        self.xi = xi
        self.kappa = kappa

        self.data_dimension = self.data.size()
        self.data_number = self.data_dimension[0]
        self.input_dimension = self.data_dimension[1] - 1

        self.input = data[:, :-1]
        self.output = data[:, self.input_dimension]
        
        if (lengthscale == None):

            self.lengthscale = torch.nn.Parameter(torch.ones(1))

        else:

            self.lengthscale = lengthscale

        if (variance == None):
            
            self.variance = torch.nn.Parameter(torch.ones(1))

        else:

            self.variance = variance

        self.all_kernel_function = {
            "matern_nu_5" : self.kf_m5
        }

        self.all_acquisition_function = {
            "expected_improvement" : self.ac_ei,
            "upper_confidence_bound" : self.ac_ucb
        }    

    def kf_m5(self, x1, x2):

        if (x1.dim() == 1):

            x1 = x1.unsqueeze(0)

        if (x2.dim() == 1):

            x2 = x2.unsqueeze(0)

        relenvant_distance = torch.cdist(x1 / self.lengthscale, x2 / self.lengthscale, p=2)

        k = self.variance * (1 + torch.sqrt(torch.tensor(5.0)) * relenvant_distance + torch.tensor(5.0/3.0) * relenvant_distance**2) * torch.exp(- torch.sqrt(torch.tensor(5.0)) * relenvant_distance)

        return k
    
    def standardized_improvement(self, mu, sigma):

        z = (mu - torch.max(self.output) - self.xi) / (sigma + 1e-9)

        return z
    
    def norm_pdf(self, x):

        pdf = (1 / torch.sqrt(torch.tensor(2.) * torch.pi)) * torch.exp(- 0.5 * x **2)

        return pdf
    
    def norm_cdf(self, x):

        cdf = 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.))))

        return cdf
    
    def ac_ei(self, mu, sigma):

        z = self.standardized_improvement(mu, sigma)

        ei = (mu - torch.max(self.output) - self.xi) * self.norm_cdf(z) + sigma * self.norm_pdf(z)

        return ei
    
    def ac_ucb(self, mu, sigma):

        ucb = mu - self.kappa * sigma

        return ucb

    def forward(self, sample):

        kernel_sample_data_matrix = self.all_kernel_function[self.kernel_name](sample, self.input)
        kernel_data_matrix = self.all_kernel_function[self.kernel_name](self.input, self.input)
        kernel_sample_matrix = self.all_kernel_function[self.kernel_name](sample, sample)
        kernel_data_sample_matrix = self.all_kernel_function[self.kernel_name](self.input, sample)

        if self.data_noise.dim() == 0:

            noise_diag = self.data_noise * torch.eye(self.data_number)

        else:

            noise_diag = torch.diag(self.data_noise)

        K_inv = torch.linalg.inv(kernel_data_matrix + noise_diag)
        mu = kernel_sample_data_matrix @ K_inv @ self.output
        sigma = torch.diagonal(kernel_sample_matrix - kernel_sample_data_matrix @ K_inv @ kernel_data_sample_matrix, 0)

        return mu, sigma
    
    def acquisition_function(self, mu, sigma):

        af = self.all_acquisition_function[self.acquisition_function_name](mu, sigma)

        return af

class embedding():

    def __init__(self, input_dimension, low_dimension, input_start, input_stop):

        super(embedding, self).__init__()

        self.input_dimension = input_dimension
        self.low_dimension = low_dimension
        self.input_start = input_start
        self.input_stop = input_stop

    def initial_embedding_matrix(self):

        Q, _ = torch.linalg.qr(torch.randn((self.input_dimension, self.input_dimension)))
        self.embedding_matrix = torch.nn.Parameter(Q[:, :self.low_dimension])

        return self.embedding_matrix

    def calculate_barrier(self, low_dimension_input):

        feasible_region_matrix = torch.cat([self.embedding_matrix, - self.embedding_matrix], dim = 0)
        high_dimension_boundary = torch.cat([self.input_stop, - self.input_start], dim = 0)

        if torch.any(feasible_region_matrix @ low_dimension_input.transpose(0, 1) >= high_dimension_boundary):

            return None
        
        barrier = - torch.sum(torch.log(high_dimension_boundary - feasible_region_matrix @ low_dimension_input + 1e-8))

        return barrier
    
    def calculate_laplacian(self, low_dimension_input, l_sigma = 1.0):

        low_dimension_input_distance = torch.sum((low_dimension_input ** 2), dim = 1).view(-1, 1)
        pairwise_distance = torch.sqrt(torch.clamp((low_dimension_input_distance + low_dimension_input_distance.transpose(0, 1) - 2.0 * (low_dimension_input @ low_dimension_input.transpose(0, 1))), min = 1e-8))

        laplacian_weight = torch.exp(- pairwise_distance ** 2 / l_sigma ** 2)
        laplacian_weight = laplacian_weight.clone()
        laplacian_weight.fill_diagonal_(0.0)
        degree_matrix = torch.diag(torch.sum(laplacian_weight, dim = 1))
        laplacian = degree_matrix - laplacian_weight

        return laplacian

    def map_to_low_dimension(self, high_dimension_input, input_dimension, low_dimension_bias = None):

        if (low_dimension_bias == None):

            low_dimension_bias = torch.zeros(input_dimension).unsqueeze(0)

        # low_dimension_input = torch.inverse(self.embedding_matrix.T @ self.embedding_matrix) @ self.embedding_matrix.T @ torch.squeeze(high_dimension_input - low_dimension_bias)

        low_dimension_input = torch.squeeze(torch.cholesky_solve(self.embedding_matrix.clone().transpose(0, 1) @ torch.unsqueeze(torch.squeeze(high_dimension_input - low_dimension_bias), 1), torch.linalg.cholesky(self.embedding_matrix.clone().transpose(0, 1) @ self.embedding_matrix.clone())))

        return low_dimension_input.unsqueeze(0)

    def map_to_high_dimension(self, low_dimension_input, input_dimension, low_dimension_bias = None):

        if (low_dimension_bias == None):

            low_dimension_bias = torch.zeros(input_dimension)

        input = (self.embedding_matrix.clone() @ low_dimension_input.transpose(0, 1)).transpose(0, 1) + low_dimension_bias

        input = torch.clamp(input, self.input_start, self.input_stop)

        return input
    
    def negative_log_marginal_likelihood(self, gp_module, low_dimension_data, data_noise = torch.tensor([1e-4])):

        input_low_dimension = low_dimension_data[:, :-1]
        output = low_dimension_data[:, self.low_dimension]

        data_number = input_low_dimension.size()[0]

        input = (self.embedding_matrix @ input_low_dimension.transpose(0, 1)).transpose(0, 1)

        kernel_low_dimenstion_data_matrix = gp_module.all_kernel_function[gp_module.kernel_name](input, input)

        kernel_low_dimenstion_data_matrix = kernel_low_dimenstion_data_matrix + data_noise * torch.eye(data_number)

        L = torch.linalg.cholesky(kernel_low_dimenstion_data_matrix)

        nlml = 0.5 * output @ torch.squeeze(torch.cholesky_solve(output.unsqueeze(1), L),1) + torch.sum(torch.log(torch.diag(L))) + 0.5 * data_number * torch.log(torch.tensor(2.0 * torch.pi))

        return nlml
    
    def gp_train(self, gp_module, low_dimension_data, gp_epoch_time, low_dimension):

        gp_optimizer = torch.optim.Adam(gp_module.parameters(), lr=0.01)

        low_dimension_input = low_dimension_data[:, :-1]
        output = low_dimension_data[:, low_dimension]

        for ep in range(gp_epoch_time):

            prediction_mu, _ = gp_module(low_dimension_input)

            loss = ((prediction_mu - output) **2).mean()

            gp_optimizer.zero_grad()
            loss.backward()
            gp_optimizer.step()

            if (ep + 1) % 100 == 0:

                print(f'Epoch [{ep+1}/{gp_epoch_time}], Loss: {loss.item():.4f}')

    def em_train(self, gp_module, low_dimension_data, em_epoch_time):

        self.embedding_matrix.requires_grad_()

        em_optimizer = torch.optim.Adam([self.embedding_matrix, *gp_module.parameters()], lr=0.01)

        for ep in range(em_epoch_time):

            loss = self.negative_log_marginal_likelihood(gp_module = gp_module, low_dimension_data = low_dimension_data.detach())

            em_optimizer.zero_grad()
            loss.backward()
            em_optimizer.step()

            # if (ep + 1) % 10 == 0:

            #     print(f'Epoch [{ep+1}/{em_epoch_time}], Loss: {loss.item():.4f}')

    def return_embedding_matrix(self):

        return self.embedding_matrix

class gradient_descent_sampling():

    def __init__(self, gp_module, data, input_start, input_stop):
         
        super(gradient_descent_sampling, self).__init__()

        self.gp = gp_module
        self.data = data
        self.input_start = input_start
        self.input_stop = input_stop

        self.data_dimension = self.data.size()
        self.input_dimension = self.data_dimension[1] - 1

    def next_sample(self, ns_epoch_time):

        next_sample = self.input_start + (self.input_stop - self.input_start) * torch.rand(self.input_dimension)
        next_sample = next_sample.unsqueeze(0)

        next_sample.requires_grad_()

        ac_optimizer = torch.optim.Adam([next_sample], lr=0.0002)

        for ep in range(ns_epoch_time):

            prediction_mu, prediction_sigma = self.gp(next_sample)

            if ((next_sample == self.input_start).any() or (next_sample == self.input_stop).any()):

                loss = - self.gp.acquisition_function(prediction_mu, prediction_sigma) + torch.tensor(1e10)

            else:

                loss = - self.gp.acquisition_function(prediction_mu, prediction_sigma)

            ac_optimizer.zero_grad()
            loss.backward()
            ac_optimizer.step()

            # if (ep + 1) % 100 == 0:

            #     print(f'Epoch [{ep+1}/{ns_epoch_time}], Loss: {loss.item():.4f}')

            with torch.no_grad():

                next_sample.copy_(next_sample.clamp(self.input_start, self.input_stop))

        return next_sample.detach()
    
    def next_sample_with_embedding(self, em_module, ns_epoch_time, barrier_mu = 1., laplacian_mu = 1.):

        next_sample = self.input_start + (self.input_stop - self.input_start) * torch.rand(self.input_dimension)
        next_sample = next_sample.clone().detach()

        next_sample.requires_grad_()

        ac_optimizer = torch.optim.Adam([next_sample], lr=0.1)

        for ep in range(ns_epoch_time):

            prediction_mu, prediction_sigma = self.gp(next_sample)

            barrier = em_module.calculate_barrier(low_dimension_input = next_sample)
            laplacian = em_module.calculate_laplacian(low_dimension_input = next_sample)

            if (barrier is None):

                loss = - self.gp.acquisition_function(prediction_mu, prediction_sigma) + torch.tensor(1e10)

            else:

                loss = - self.gp.acquisition_function(prediction_mu, prediction_sigma) + barrier * barrier_mu + (prediction_mu.T @ laplacian @ prediction_mu) * laplacian_mu

            ac_optimizer.zero_grad()
            loss.backward()
            ac_optimizer.step()

            # if (ep + 1) % 100 == 0:

            #     print(f'Epoch [{ep+1}/{ns_epoch_time}], Loss: {loss.item():.4f}')

            with torch.no_grad():

                next_sample.copy_(next_sample.clamp(self.input_start, self.input_stop))

        return next_sample.detach()