import torch

class kernel_function():

    def __init__(self, data, length_scale = None, kernel_name = "matern_nu_5"):

        super().__init__()
    
        self.data = data        
        self.kernel_name = kernel_name

        self.data_dimension = self.data.size()
        self.data_number = self.data_dimension[0]
        self.input_dimension = self.data_dimension[1] - 1

        self.amplitude = torch.tensor(1.)

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

        k = self.amplitude * self.amplitude * torch.exp( -relenvant_distance)

        return k
    
    def matern_nu_3(self, relenvant_distance):

        k = self.amplitude * self.amplitude * (1 + torch.sqrt(torch.tensor(3.)) * relenvant_distance) * torch.exp( - torch.sqrt(torch.tensor(3.)) * relenvant_distance)

        return k
    
    def matern_nu_5(self, relenvant_distance):

        k = self.amplitude * self.amplitude * (1 + torch.sqrt(torch.tensor(5.)) * relenvant_distance + 5 * relenvant_distance * relenvant_distance / torch.tensor(3.)) * torch.exp( - torch.sqrt(torch.tensor(5.)) * relenvant_distance)

        return k
    
    def squared_exponential(self, relenvant_distance):

        k = self.amplitude * self.amplitude * torch.exp( - relenvant_distance * relenvant_distance / torch.tensor(2.))

        return k
    
    def kernel_data_sample(self):

        self.relenvant_distance_data = torch.zeros(self.data_number, self.data_number)

        for d in range(self.input_dimension):

            self.relenvant_distance_data[:, :] = self.relenvant_distance_data[:, :] + torch.abs(self.input[:, d].expand(self.data_number, self.data_number) - self.input[:, d].expand(self.data_number, self.data_number).transpose(0, 1)) / self.length_scale[d]
        
        self.kernel_data_matrix = self.all_kernel_function[self.kernel_name](self.relenvant_distance_data)
        self.kernel_sample_matrix = self.all_kernel_function[self.kernel_name](torch.tensor(0.))

    def kernel_matrix(self, sample):

        relenvant_distance_sample = torch.zeros(self.data_number)

        for d in range(self.input_dimension):

            relenvant_distance_sample = relenvant_distance_sample + torch.abs(self.input[:, d] - sample[d]) / self.length_scale[d]

        kernel_data_sample_matrix = self.all_kernel_function[self.kernel_name](relenvant_distance_sample)        
   
        return kernel_data_sample_matrix
    
    def return_self(self):

        return self
    
class gaussian_progress():

    def __init__(self, kernel_self, sample,  data_noise = None, acquisition_function_name = "expected_improvement"):

        super().__init__()

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

        self.xi = torch.tensor(1.)
        self.kappa = torch.tensor(2.)

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

        ei = (self.mu - torch.max(self.output) - self.xi) * self.standard_normal_cdf(z) + self.sigma * self.standard_normal_pdf(z)

        return ei
    
    def probability_of_improvement(self):

        z = self.standardized_improvement()

        pi = self.standard_normal_cdf(z)

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

        super().__init__()

        self.acquisition_function_name = gaussian_progress_self.acquisition_function_name
        self.all_acquisition_function = gaussian_progress_self.all_acquisition_function
        self.input_dimension = gaussian_progress_self.input_dimension
        self.sample = gaussian_progress_self.sample
    
    def next_sample(self, next_sample_number):

        af = gaussian_progress.acquisition_function(self)

        values, indices = torch.topk(af, next_sample_number)

        ns = torch.zeros(next_sample_number, self.input_dimension)

        for n in range(next_sample_number):

            for d in range(self.input_dimension):

                ns[n][d] = self.sample[indices[n]][d]

        # loop_state = True

        # while(loop_state):

        #     values, indices = torch.topk(af, next_sample_number)

        #     ns = torch.zeros(next_sample_number, self.input_dimension)

        #     for n in range(next_sample_number):

        #         for d in range(self.input_dimension):

        #             ns[n][d] = self.sample[indices[n]][d]

        #     eq = torch.eq(initial_input, ns)

        #     for n in range(next_sample_number):

        #         if_state = True

        #         for d in range(self.input_dimension):

        #             if (if_state and eq[n][d]):

        #                 if_state = True

        #             else:

        #                 if_state = False
        #                 loop_state = False

        #                 break

        #         if (if_state):

        #             next_sample_number = next_sample_number + 1

        return ns
    