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

    def normal_probability_density_function(self, x, mu, sigma):

        try:

            sigma_inv = torch.linalg.inv(sigma)
            sigma_det = torch.linalg.det(sigma)

        except:

            sigma_inv = torch.linalg.pinv(sigma)
            _, s, _ = torch.svd(sigma)
            sigma_det = torch.prod(s[s > 1e-10])

        sigma_det = torch.clamp(sigma_det, min=1e-10)

        diff = x - mu
        quad = diff.unsqueeze(0) @ sigma_inv @ diff.unsqueeze(1)
        exponent = -0.5 * quad.squeeze()  

        normalization = 1.0 / torch.sqrt((2 * torch.pi) ** self.input_dimension * sigma_det)

        pdf = normalization * torch.exp(exponent)

        return pdf

    def normal_yield(self, input, input_step, output, sigma):

        input_number = input.size()[0]
        
        yield_value = torch.zeros(input_number)

        for n_mu in range(input_number):

            total_yield = 0

            for n_x in range(input_number):

                condition = True

                total_yield = total_yield + self.normal_probability_density_function(x = input[n_x], mu = input[n_mu], sigma = sigma)

                for n_condition in range(self.condition_number):

                    if ((output[n_x][n_condition] < self.condition_min[n_condition]) or ((output[n_x][n_condition] > self.condition_max[n_condition]))):

                        condition = False

                if condition:

                    yield_value[n_mu] = yield_value[n_mu] + self.normal_probability_density_function(x = input[n_x], mu = input[n_mu], sigma = sigma)

            total_yield = total_yield * torch.prod(input_step)

        yield_value = yield_value * torch.prod(input_step)

        return yield_value