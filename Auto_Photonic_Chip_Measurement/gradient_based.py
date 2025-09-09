import torch
import numpy as np

class data_scaler():

    def __init__(self, input, output, predicted_input = False):

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

        if (self.predicted_input is not False):

            scaler_predicted_input = ( self.predicted_input - self.input_mean) / self.input_std        

            return scaler_input, scaler_output, scaler_predicted_input
        
        else:

            return scaler_input, scaler_output

    def inverse_standardscaler(self, scaler_predicted_output):

        predicted_output = scaler_predicted_output * self.output_std + self.output_mean

        return predicted_output
    
    def minmaxscaler(self, input_min = False, input_max = False):

        if (input_min is False) and (input_max is False):

            self.input_min = torch.min(self.input)
            self.input_max = torch.max(self.input)

        else:

            self.input_min = input_min
            self.input_max = input_max

        self.output_min = torch.min(self.output)
        self.output_max = torch.max(self.output)

        scaler_input = (self.input - self.input_min) / (self.input_max - self.input_min)
        scaler_output = (self.output - self.output_min) / (self.output_max - self.output_min)

        if (self.predicted_input is not False):

            scaler_predicted_input = (self.predicted_input - self.input_min) / (self.input_max - self.input_min)

            return scaler_input, scaler_output, scaler_predicted_input
        
        else:

            return scaler_input, scaler_output
    
    def inverse_minmaxscaler(self, scaler_predicted_input = False, scaler_predicted_output = False):

        if (scaler_predicted_input is False):

            predicted_output = scaler_predicted_output * (self.output_max - self.output_min) + self.output_min

            return predicted_output
        
        elif(scaler_predicted_output is False):

            predicted_input = scaler_predicted_input * (self.input_max - self.input_min) + self.input_min

            return predicted_input
    
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
    
    def inverse_minmax_input_standard_output_scaler(self, scaler_predicted_input = False, scaler_predicted_output = False):

        if (scaler_predicted_input is False):

            predicted_output = scaler_predicted_output * self.output_std + self.output_mean

            return predicted_output
        
        elif(scaler_predicted_output is False):

            predicted_input = scaler_predicted_input * (self.input_max - self.input_min) + self.input_min

            return predicted_input
    
    def inverse_minmax_input_standard_output_scaler_theta(self, scaler_weight, scaler_bias):

        weight = scaler_weight * self.output_std / (self.input_max - self.input_min)
        bias = scaler_bias * self.output_std + self.output_mean - weight * self.input_min

        return weight, bias

class AdamOptimizer:

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):

        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.momentum = None
        self.velocity = None
        self.time = 0

    def next_input_max(self, input_0, input, output_0, output):

        if self.momentum is None:

            self.momentum = torch.zeros_like(input)

        if self.velocity is None:

            self.velocity = torch.zeros_like(input)

        self.time = self.time + 1

        gradient = (output - output_0) * (input - input_0) / torch.dot(input - input_0, input - input_0)

        self.momentum = self.beta1 * self.momentum + (1 - self.beta1) * gradient
        self.velocity = self.beta2 * self.velocity + (1 - self.beta2) * gradient**2
        momentum_hat = self.momentum / (1 - self.beta1**self.time)
        velocity_hat = self.velocity / (1 - self.beta2**self.time)
        update = self.learning_rate * momentum_hat / (torch.sqrt(velocity_hat) + self.epsilon)

        next_input = input + update

        return next_input
    
    def next_input_min(self, input_0, input, output_0, output):

        if self.momentum is None:

            self.momentum = torch.zeros_like(input)

        if self.velocity is None:

            self.velocity = torch.zeros_like(input)

        self.time = self.time + 1

        gradient = (output - output_0) * (input - input_0) / torch.dot(input - input_0, input - input_0)

        self.momentum = self.beta1 * self.momentum + (1 - self.beta1) * gradient
        self.velocity = self.beta2 * self.velocity + (1 - self.beta2) * gradient**2
        momentum_hat = self.momentum / (1 - self.beta1**self.time)
        velocity_hat = self.velocity / (1 - self.beta2**self.time)
        update = self.learning_rate * momentum_hat / (torch.sqrt(velocity_hat) + self.epsilon)

        next_input = input - update

        return next_input
    
    def random_direction(self, input_0, epsilon = 0.001):

        input_size = input_0.size()

        direction = torch.randn(input_size)

        input_sample = input_0 + direction * epsilon

        return input_sample, direction
       
    def one_sample_next_input_max(self, input_0, input_sample, direction, output_0, output_sample, epsilon = 0.001):

        if self.momentum is None:

            self.momentum = torch.zeros_like(input_sample)

        if self.velocity is None:

            self.velocity = torch.zeros_like(input_sample)

        self.time = self.time + 1

        gradient = (output_sample - output_0) * direction / epsilon

        self.momentum = self.beta1 * self.momentum + (1 - self.beta1) * gradient
        self.velocity = self.beta2 * self.velocity + (1 - self.beta2) * gradient**2
        momentum_hat = self.momentum / (1 - self.beta1**self.time)
        velocity_hat = self.velocity / (1 - self.beta2**self.time)
        update = self.learning_rate * momentum_hat / (torch.sqrt(velocity_hat) + self.epsilon)

        next_input = input_0 + update

        return next_input