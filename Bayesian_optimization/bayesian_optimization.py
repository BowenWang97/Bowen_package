import numpy as np
import torch
import torch.nn as nn
import scipy as sci

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

        ucb = self.mu + self.kappa * self.sigma

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

    def __init__(self, kernel_self,  data_noise = None, acquisition_function_name = "expected_improvement", xi = torch.tensor(1.), kappa = torch.tensor(2.)):

        super(gaussian_progress_nn, self).__init__()

        self.all_kernel_function = kernel_self.all_kernel_function
        self.data_number = kernel_self.data_number
        self.kernel_data_matrix = kernel_self.kernel_data_matrix
        self.kernel_sample_matrix = kernel_self.kernel_sample_matrix
        self.input = kernel_self.input
        self.input_dimension = kernel_self.input_dimension
        self.kernel_name = kernel_self.kernel_name
        self.length_scale = kernel_self.length_scale
        self.output = kernel_self.output

        self.acquisition_function_name = acquisition_function_name

        if (data_noise == None):

            self.data_sigma = torch.zeros(self.data_number)

        else:

            self.data_sigma = data_noise

        self.xi = xi
        self.kappa = kappa

        self.all_acquisition_function = {
            "expected_improvement" : self.expected_improvement,
            "probability_of_improvement" : self.probability_of_improvement,
            "upper_confidence_bound" : self.upper_confidence_bound
        }       

    def forward(self, sample):

        mu = torch.zeros(sample.shape[0])
        sigma = torch.zeros(sample.shape[0])

        for n in range(sample.shape[0]):

            kernel_data_sample_matrix = kernel_function.kernel_matrix(self, sample[n])
            
            mu[n] = torch.linalg.multi_dot((kernel_data_sample_matrix, torch.linalg.inv(self.kernel_data_matrix + self.data_sigma * torch.eye(self.data_number)), self.output))
            sigma[n] = self.kernel_sample_matrix - torch.linalg.multi_dot((kernel_data_sample_matrix, torch.linalg.inv(self.kernel_data_matrix + self.data_sigma * torch.eye(self.data_number)), torch.unsqueeze(kernel_data_sample_matrix, 1)))

        sigma = torch.clamp(self.sigma, min = 0.)

        return mu, sigma
    
    def standard_normal_pdf(self, x):

        pdf = torch.exp( - x * x / 2) / torch.sqrt(torch.tensor(2.) * torch.pi)

        return pdf
    
    def standard_normal_cdf(self, x):

        cdf = (1 + torch.erf(x / torch.sqrt(torch.tensor(2.)))) / 2

        return cdf
    
    def standardized_improvement(self, mu, sigma):

        z = (mu - torch.max(self.output) - self.xi) / sigma

        return z

    def expected_improvement(self, mu, sigma):

        z = self.standardized_improvement(mu, sigma)

        ei = (mu - torch.max(self.output) - self.xi) * torch.tensor(sci.stats.norm.cdf(z)) + sigma * torch.tensor(sci.stats.norm.pdf(z))

        return ei
    
    def probability_of_improvement(self, mu, sigma):

        z = self.standardized_improvement(mu, sigma)

        pi = torch.tensor(sci.stats.norm.cdf(z))

        return pi
    
    def upper_confidence_bound(self, mu, sigma):

        ucb = mu + self.kappa * sigma

        return ucb
    
    def acquisition_function(self, mu, sigma):

        af = self.all_acquisition_function[self.acquisition_function_name](mu, sigma)

        return af

class gradient_descent_sampling():

    def __init__(self, kernel_self, input_start, input_stop):
         
        super(gradient_descent_sampling, self).__init__()

        self.kernel_self = kernel_self
        self.input_dimension = kernel_self.input_dimension
        self.input_start = input_start
        self.input_stop = input_stop

    def next_sample(self, gf_epoch_time):

        gp = gaussian_progress_nn(self.kernel_self)

        next_sample = self.input_start + (self.input_stop - self.input_start) * torch.rand_like()

        gp_optimizer = torch.optim.Adam(next_sample, lr=0.01)

        for gf_ep in range(gf_epoch_time):

            prediction_mu, prediction_sigma = gp(next_sample)

            loss = - gp.acquisition_function(prediction_mu, prediction_sigma)

            gp_optimizer.zero_grad()
            loss.backward()
            gp_optimizer.step()

            with torch.no_grad():

                next_sample.clamp_(self.input_stop, self.input_start)

        return next_sample.detach()
    
class random_embedding():

    def __init__(self, kernel_self, low_dimension):

        super(random_embedding, self).__init__()

        self.data_number = kernel_self.data_number
        self.input_dimension = kernel_self.input_dimension
        self.input = kernel_self.input
        self.low_dimension = low_dimension

    def map_to_high_dimension(self, random_embedding_matrix, low_dim_input, low_dim_bias = None):

        if (low_dim_bias == None):

            low_dim_bias = torch.zeros(self.input_dimension)

        input = random_embedding_matrix @ low_dim_input + low_dim_bias

        return input
    
    def rande(self, re_epoch_time):

        random_embedding_matrix = torch.randn((self.data_number, self.low_dimension)) / self.low_dimension

        for re_ep in range(re_epoch_time):

            gp = gaussian_progress(self.kernel_self, self.input)

            prediction_mu, prediction_sigma = gp.fit_function()