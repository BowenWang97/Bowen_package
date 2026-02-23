import math
import torch

class yield_analysis():

    def __init__(self, input_min, input_max, condition_min, condition_max):

        super(yield_analysis, self).__init__()

        self.input_min = input_min
        self.input_max = input_max
        self.condition_min = condition_min
        self.condition_max = condition_max

        self.input_dimension = input_min.size()[0]
        self.condition_number = condition_min.size()[0]

    # def normal_probability_density_function(self, x, mu, sigma):

    #     try:

    #         sigma_inv = torch.linalg.inv(sigma)
    #         sigma_det = torch.linalg.det(sigma)

    #     except:

    #         sigma_inv = torch.linalg.pinv(sigma)
    #         _, s, _ = torch.svd(sigma)
    #         sigma_det = torch.prod(s[s > 1e-10])

    #     sigma_det = torch.clamp(sigma_det, min=1e-10)

    #     diff = x - mu
    #     quad = diff.unsqueeze(0) @ sigma_inv @ diff.unsqueeze(1)
    #     exponent = -0.5 * quad.squeeze()  

    #     normalization = 1.0 / torch.sqrt((2 * torch.pi) ** self.input_dimension * sigma_det)

    #     pdf = normalization * torch.exp(exponent)

    #     return pdf

    # def normal_yield(self, input, input_step, output, sigma):

    #     input_number = input.size()[0]
        
    #     yield_value = torch.zeros(input_number)

    #     for n_mu in range(input_number):

    #         total_yield = 0

    #         for n_x in range(input_number):

    #             condition = True

    #             total_yield = total_yield + self.normal_probability_density_function(x = input[n_x], mu = input[n_mu], sigma = sigma)

    #             for n_condition in range(self.condition_number):

    #                 if ((output[n_x][n_condition] < self.condition_min[n_condition]) or ((output[n_x][n_condition] > self.condition_max[n_condition]))):

    #                     condition = False

    #             if condition:

    #                 yield_value[n_mu] = yield_value[n_mu] + self.normal_probability_density_function(x = input[n_x], mu = input[n_mu], sigma = sigma)

    #         total_yield = total_yield * torch.prod(input_step)

    #     yield_value = yield_value * torch.prod(input_step)

    #     return yield_value

    def normal_probability_density_function(self, x, mu, sigma):

        try:

            sigma_inv = torch.linalg.inv(sigma)
            sigma_det = torch.linalg.det(sigma)

        except:

            sigma_inv = torch.linalg.pinv(sigma)
            _, s, _ = torch.svd(sigma)
            sigma_det = torch.prod(s[s > 1e-10])
        
        sigma_det = torch.clamp(sigma_det, min=1e-10)

        # x_Sinv = x @ sigma_inv
        # mu_Sinv = mu @ sigma_inv

        # A = (x_Sinv * x).sum(-1)
        # B = (mu_Sinv * mu).sum(-1)
        # C = x_Sinv @ mu.T

        # quad = A.unsqueeze(1) - 2*C + B.unsqueeze(0)

        diff = x.unsqueeze(1) - mu.unsqueeze(0)

        quad = torch.einsum('ijk,kl,ijl->ij', diff, sigma_inv, diff)
        exponent = -0.5 * quad
        
        normalization = 1.0 / torch.sqrt((2 * torch.pi) ** self.input_dimension * sigma_det)
        pdf_vector = normalization * torch.exp(exponent)

        del diff, quad, exponent, normalization, sigma_inv, sigma_det
        
        return pdf_vector

    def normal_yield(self, input, input_step, output, sigma):

        input_number = input.size()[0]

        yield_value = torch.zeros(input_number)

        for n in range(input_number):
        
            pdf_vector = self.normal_probability_density_function(x = input, mu = input[n], sigma = sigma)
            
            condition_mask = torch.ones(input_number, dtype=torch.bool, device=input.device)
            
            for n_condition in range(self.condition_number):

                condition_mask = condition_mask & (output[:, n_condition] >= self.condition_min[n_condition]) & (output[:, n_condition] <= self.condition_max[n_condition])
            
            yield_value[n] = torch.sum(pdf_vector[condition_mask], dim=0) * torch.prod(input_step)

            del pdf_vector
            
        return yield_value

    # def normal_yield(self, input, input_step, output, sigma):

    #     input_number = input.size()[0]
        
    #     pdf_matrix = self.normal_probability_density_function(x = input, mu = input, sigma = sigma)
        
    #     condition_mask = torch.ones(input_number, dtype=torch.bool, device=input.device)
        
    #     for n_condition in range(self.condition_number):

    #         condition_mask = condition_mask & (output[:, n_condition] >= self.condition_min[n_condition]) & (output[:, n_condition] <= self.condition_max[n_condition])
        
    #     yield_value = torch.sum(pdf_matrix[condition_mask], dim=0) * torch.prod(input_step)
            
    #     return yield_value