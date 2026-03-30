import bayesian_optimization as BO
import csv
import gradient_based as GP
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import time
import torch

class polarization():

    def __init__(self, powermeter, paddle_control, polarization_threshold_dB = -30):

        super(polarization, self).__init__()

        self.powermeter = powermeter

        self.paddle_control = paddle_control
        self.polarization_threshold_dB = polarization_threshold_dB

    def initial_optimization(self, start = 10, stop = 170, step = 40):

        range_angle = torch.arange(start, stop + step, step)

        paddle_1_angle, paddle_2_angle, paddle_3_angle = torch.meshgrid(range_angle, range_angle, range_angle, indexing='ij')

        paddle_1_angle = paddle_1_angle.flatten().unsqueeze(-1)
        paddle_2_angle = paddle_2_angle.flatten().unsqueeze(-1)
        paddle_3_angle = paddle_3_angle.flatten().unsqueeze(-1)

        paddle_angle = torch.cat((paddle_1_angle, paddle_2_angle, paddle_3_angle), dim = 1).tolist()

        power = []

        for n in range(len(paddle_angle)):

            for paddle_number in range(3):
                
                self.paddle_control.move_to(paddle_number = (paddle_number + 1), position = paddle_angle[n][paddle_number])
                
            current_power = self.powermeter.measure()

            power.append(current_power)

        power = torch.tensor(power)

        max_position = paddle_angle[torch.argmax(power).item()]

        print(max_position)

        for paddle_number in range(3):
        
            self.paddle_control.move_to(paddle_number=(paddle_number + 1), position = max_position[paddle_number])

        return max_position

    def restrict_angle(self, angle):

        angle_mod = angle % 360.0
        angle_plus_180 = (angle_mod + 180.0) % 360.0

        in_range_original = (angle_mod >= 0) & (angle_mod <= 170)
        in_range_shift = (angle_plus_180 >= 0) & (angle_plus_180 <= 170)

        angle = torch.where(in_range_original, angle_mod, torch.where(in_range_shift, angle_plus_180, angle_plus_180))
        angle = torch.where((in_range_original | in_range_shift), angle, torch.where(angle < 0, 0.0, 170.0))

        angle = torch.round(angle * 10) / 10        

        return angle
    
    def scan_optimization(self, step = 2):

        current_position = self.paddle_control.read_current_position()

        range_angle_1 = torch.arange(current_position[0] - 2 * step, current_position[0] + 3 * step, step)
        range_angle_2 = torch.arange(current_position[1] - 2 * step, current_position[1] + 3 * step, step)
        range_angle_3 = torch.arange(current_position[2] - 2 * step, current_position[2] + 3 * step, step)

        range_angle_1 = self.restrict_angle(angle = range_angle_1)
        range_angle_2 = self.restrict_angle(angle = range_angle_2)
        range_angle_3 = self.restrict_angle(angle = range_angle_3)

        range_angle_1 = torch.unique(range_angle_1)
        range_angle_2 = torch.unique(range_angle_2)
        range_angle_3 = torch.unique(range_angle_3)

        range_angle_1 = torch.Tensor([int(num) for num in range_angle_1.squeeze().tolist()])
        range_angle_2 = torch.Tensor([int(num) for num in range_angle_2.squeeze().tolist()])
        range_angle_3 = torch.Tensor([int(num) for num in range_angle_3.squeeze().tolist()])

        paddle_1_angle, paddle_2_angle, paddle_3_angle = torch.meshgrid(range_angle_1, range_angle_2, range_angle_3, indexing='ij')

        paddle_1_angle = paddle_1_angle.flatten().unsqueeze(-1)
        paddle_2_angle = paddle_2_angle.flatten().unsqueeze(-1)
        paddle_3_angle = paddle_3_angle.flatten().unsqueeze(-1)

        position = torch.cat((paddle_1_angle, paddle_2_angle, paddle_3_angle), dim = 1).tolist()

        power = []

        for n in range(len(position)):

            for paddle_number in range(3):
                
                self.paddle_control.move_to(paddle_number = (paddle_number + 1), position = position[n][paddle_number])

            time.sleep(1)
                
            current_power = self.powermeter.measure()

            power.append(current_power)

            print(f"\rProgressing: {int(((n + 1)/len(position)) * 100)}%, Max Power = {max(power)}dB", end = '', flush = True)

        print()

        power = torch.tensor(power)

        max_position = position[torch.argmax(power).item()]

        # print(max_position)

        for paddle_number in range(3):
        
            self.paddle_control.move_to(paddle_number=(paddle_number + 1), position = max_position[paddle_number])

        return position, power.tolist(), max_position, max(power)

    def bo_optimization(self, iteration_time = 40):

        current_position = self.paddle_control.read_current_position()

        current_power = self.powermeter.measure()

        input_min = torch.tensor([0., 0., 0.])
        input_max = torch.tensor([170., 170., 170.])

        initial_sample = BO.latin_hypercube_sampling(sample_min = input_min, sample_max = input_max, sample_number = 30)

        position = torch.cat((torch.tensor([current_position]), initial_sample), dim = 0)
        power =  torch.Tensor([current_power])

        position = torch.round(position)

        for it in range(1, iteration_time):

            for paddle_number in range(3):

                self.paddle_control.move_to(paddle_number = (paddle_number + 1), position = position[it][paddle_number].tolist())

            time.sleep(0.5)

            current_power = self.powermeter.measure()

            power = torch.cat((power, torch.Tensor([current_power])), dim = 0)    

            print(f"\rProgressing: {int(((it + 1)/iteration_time) * 100)}%, Max Power = {max(power)}dB", end = '', flush = True)

            if (it >= 30):

                data_scaler = BO.data_scaler(input = position, output = power)

                scaler_input, scaler_output  = data_scaler.minmaxscaler(input_min = input_min, input_max = input_max)

                scaler_data = torch.cat((scaler_input, scaler_output.unsqueeze(1)), dim = 1)

                gp_module = BO.GP_nn(scaler_data)

                gds = BO.gradient_descent_sampling(gp_module = gp_module, data = scaler_data)
                
                scaler_next_input = gds.next_sample()

                next_input = data_scaler.inverse_minmaxscaler(scaler_predicted_input = scaler_next_input)

                next_input = torch.round(next_input)

                position = torch.cat((position, next_input.unsqueeze(0)), dim = 0)

        print()

        max_position = position[torch.argmax(power).item()]

        for paddle_number in range(3):
        
            self.paddle_control.move_to(paddle_number = (paddle_number + 1), position = max_position[paddle_number].tolist())

        return position.tolist(), power.tolist(), max_position.tolist(), max(power)
    
    def bo_optimization_with_wavelength(self, laser, iteration_time = 80, scan_range = 20):

        current_position = self.paddle_control.read_current_position()
        current_wavelength = laser.get_laser_wavelength()

        current_power = self.powermeter.measure()

        current_input = torch.tensor([[current_position[0], current_position[1], current_position[2], current_wavelength]])

        input_min = torch.tensor([current_position[0] - scan_range, current_position[1] - scan_range, current_position[2] - scan_range, 1500])
        input_max = torch.tensor([current_position[0] + scan_range, current_position[1] + scan_range, current_position[2] + scan_range, 1600])

        initial_sample_number = 40
        initial_sample = BO.latin_hypercube_sampling(sample_min = input_min, sample_max = input_max, sample_number = initial_sample_number)

        input = torch.cat((current_input, initial_sample), dim = 0)
        power =  torch.Tensor([current_power])
        
        input = torch.round(input)
        input[:, 0] = self.restrict_angle(input[:, 0])
        input[:, 1] = self.restrict_angle(input[:, 1])
        input[:, 2] = self.restrict_angle(input[:, 2])

        for it in range(1, iteration_time):

            for paddle_number in range(3):

                self.paddle_control.move_to(paddle_number = (paddle_number + 1), position = input[it][paddle_number].tolist())

            laser.set_laser_wavelength(wavelength = input[it][3])

            time.sleep(0.5)

            current_power = self.powermeter.measure()

            power = torch.cat((power, torch.Tensor([current_power])), dim = 0)    

            print(f"\rProgressing: {int(((it + 1)/iteration_time) * 100)}%, Max Power = {max(power)}dB", end = '', flush = True)

            if (it >= initial_sample_number):

                data_scaler = BO.data_scaler(input = input, output = power)

                scaler_input, scaler_output  = data_scaler.minmaxscaler(input_min = input_min, input_max = input_max)

                scaler_data = torch.cat((scaler_input, scaler_output.unsqueeze(1)), dim = 1)

                gp_module = BO.GP_nn(scaler_data)

                gds = BO.gradient_descent_sampling(gp_module = gp_module, data = scaler_data)
                
                scaler_next_input = gds.next_sample()

                next_input = data_scaler.inverse_minmaxscaler(scaler_predicted_input = scaler_next_input)

                next_input = torch.round(next_input)
                next_input[0] = self.restrict_angle(next_input[0])
                next_input[1] = self.restrict_angle(next_input[1])
                next_input[2] = self.restrict_angle(next_input[2])

                input = torch.cat((input, next_input.unsqueeze(0)), dim = 0)            

        max_position = input[torch.argmax(power).item()]

        for paddle_number in range(3):
        
            self.paddle_control.move_to(paddle_number = (paddle_number + 1), position = max_position[paddle_number].tolist())

        laser.set_laser_wavelength(wavelength = max_position[3])

        return input.tolist(), power.tolist(), max_position.tolist(), max(power)
            
    def polarization_scan(self, file_processing, laser, powermeter, wavelength_start, wavelength_stop, wavelength_number, scan_step = 10, scan_range = 50):

        current_position = self.paddle_control.read_current_position()

        range_angle_1 = torch.arange(-scan_range, scan_range + scan_step, scan_step) + current_position[0]
        range_angle_2 = torch.arange(-scan_range, scan_range + scan_step, scan_step) + current_position[1]
        range_angle_3 = torch.arange(-scan_range, scan_range + scan_step, scan_step) + current_position[2]

        range_angle_1 = self.restrict_angle(angle = range_angle_1)
        range_angle_2 = self.restrict_angle(angle = range_angle_2)
        range_angle_3 = self.restrict_angle(angle = range_angle_3)

        range_angle_1 = torch.unique(range_angle_1)
        range_angle_2 = torch.unique(range_angle_2)
        range_angle_3 = torch.unique(range_angle_3)

        range_angle_1 = torch.Tensor([int(num) for num in range_angle_1.squeeze().tolist()])
        range_angle_2 = torch.Tensor([int(num) for num in range_angle_2.squeeze().tolist()])
        range_angle_3 = torch.Tensor([int(num) for num in range_angle_3.squeeze().tolist()])

        paddle_1_angle, paddle_2_angle, paddle_3_angle = torch.meshgrid(range_angle_1, range_angle_2, range_angle_3, indexing='ij')

        paddle_1_angle = paddle_1_angle.flatten().unsqueeze(-1)
        paddle_2_angle = paddle_2_angle.flatten().unsqueeze(-1)
        paddle_3_angle = paddle_3_angle.flatten().unsqueeze(-1)

        paddle_angle = torch.cat((paddle_1_angle, paddle_2_angle, paddle_3_angle), dim = 1).tolist()

        power = []

        for n in range(len(paddle_angle)):

            for paddle_number in range(3):

                self.paddle_control.move_to(paddle_number = (paddle_number + 1), position = paddle_angle[n][paddle_number])

            time.sleep(0.5)

            wavelength, power = wavelength_scan(laser = laser, powermeter = powermeter, wavelength_start = wavelength_start, wavelength_stop = wavelength_stop, wavelength_number = wavelength_number)

            file_processing.save_data_csv(wavelength = wavelength, power = power, id = n)

        return paddle_angle

class precise_position():

    def __init__(self, powermeter, qs, scan_threshold_dB = -33):

        super(precise_position, self).__init__()

        self.powermeter = powermeter

        self.qs = qs
        self.qs.response_timeout = 10

        self.qs.ustep[:] = 7

        self.scan_threshold_dB = scan_threshold_dB

    def scan_optimization(self, scan_range = 0.6, scan_step = 0.2, move = True):

        current_position_x = self.qs.x[0]
        current_position_y = self.qs.x[1]

        x_range = torch.arange(-scan_range, scan_range + scan_step, scan_step)
        y_range = torch.arange(scan_range, -scan_range - scan_step, -scan_step)

        x_position, y_posiotion = torch.meshgrid(x_range, y_range, indexing='ij')

        x_position = x_position.flatten().unsqueeze(-1)
        y_posiotion = y_posiotion.flatten().unsqueeze(-1)

        position = torch.cat((x_position, y_posiotion), dim = 1)

        power = []

        for n in range(len(position)):

            self.qs.x[0] = current_position_x + float(position[n][0])
            self.qs.wait_until_stopped()
            self.qs.x[1] = current_position_y + float(position[n][1])
            self.qs.wait_until_stopped()

            time.sleep(0.5)

            current_power = self.powermeter.measure()

            power.append(current_power)

            print(f"\rProgressing: {int(((n + 1)/len(position)) * 100)}%, Max Power = {max(power)}dB", end = '', flush = True)

        print()

        power = torch.tensor(power)

        max_index = torch.argmax(power).item()

        self.qs.x[0] = current_position_x
        self.qs.x[1] = current_position_y

        if (move):

            self.qs.x[0] = current_position_x + float(position[max_index][0])
            self.qs.x[1] = current_position_y + float(position[max_index][1])

            return position.tolist(), power, position[max_index].tolist()
        
        else:

            return position.tolist(), power, position[max_index].tolist()
        
    def bo_optimization(self, iteration_time = 40, scan_range = 3):

        current_position_x = self.qs.x[0]
        current_position_y = self.qs.x[1]

        current_power = self.powermeter.measure()

        input = torch.Tensor([[current_position_x, current_position_y]])
        power =  torch.Tensor([current_power])

        input_min = torch.tensor([current_position_x - scan_range, current_position_y - scan_range])
        input_max = torch.tensor([current_position_x + scan_range, current_position_y + scan_range])

        initial_sample_number = 20
        initial_sample = BO.latin_hypercube_sampling(sample_min = input_min, sample_max = input_max, sample_number = initial_sample_number)

        input = torch.cat((input, initial_sample), dim = 0)

        for it in range(1, iteration_time):

            self.qs.x[0] = float(input[it][0])
            self.qs.wait_until_stopped()
            self.qs.x[1] = float(input[it][1])
            self.qs.wait_until_stopped()

            time.sleep(0.5)

            current_power = self.powermeter.measure()

            power = torch.cat((power, torch.Tensor([current_power])), dim = 0)

            print(f"\rProgressing: {int(((it + 1)/iteration_time) * 100)}%, Max Power = {max(power)}dB", end = '', flush = True)

            if (it >= initial_sample_number):

                data_scaler = BO.data_scaler(input = input, output = power)

                scaler_input, scaler_output  = data_scaler.minmaxscaler(input_min = input_min, input_max = input_max)

                scaler_data = torch.cat((scaler_input, scaler_output.unsqueeze(1)), dim = 1)

                gp_module = BO.GP_nn(scaler_data)

                gds = BO.gradient_descent_sampling(gp_module = gp_module, data = scaler_data)
                
                scaler_next_input = gds.next_sample()

                next_input = data_scaler.inverse_minmaxscaler(scaler_predicted_input = scaler_next_input)

                next_input = torch.round(next_input * 1000) / 1000

                next_input = next_input.unsqueeze(0)
                
                input = torch.cat((input, next_input), dim = 0)

        max_input = input[torch.argmax(power).item()]

        self.qs.x[0] = float(max_input[0])
        self.qs.wait_until_stopped()
        self.qs.x[1] = float(max_input[1])
        self.qs.wait_until_stopped()

        return input.tolist(), power.tolist(), max_input.tolist(), max(power)
    
    def bo_optimization_with_wavelength(self, laser, iteration_time = 60, scan_range = 0.5):

        current_position_x = self.qs.x[0]
        current_position_y = self.qs.x[1]
        current_wavelength = laser.get_laser_wavelength()

        current_power = self.powermeter.measure()

        input = torch.Tensor([[current_position_x, current_position_y, current_wavelength]])
        power =  torch.Tensor([current_power])

        input_min = torch.tensor([current_position_x - scan_range, current_position_y - scan_range, 1500])
        input_max = torch.tensor([current_position_x + scan_range, current_position_y + scan_range, 1600])

        initial_sample_number = 30
        initial_sample = BO.latin_hypercube_sampling(sample_min = input_min, sample_max = input_max, sample_number = initial_sample_number)

        input = torch.cat((input, initial_sample), dim = 0)

        input[:, 2] = torch.round(input[:, 2])

        for it in range(1, iteration_time):

            self.qs.x[0] = float(input[it][0])
            self.qs.wait_until_stopped()
            self.qs.x[1] = float(input[it][1])
            self.qs.wait_until_stopped()
            laser.set_laser_wavelength(wavelength = input[it][2])

            time.sleep(0.5)

            current_power = self.powermeter.measure()

            power = torch.cat((power, torch.Tensor([current_power])), dim = 0)

            print(f"\rProgressing: {int(((it + 1)/iteration_time) * 100)}%, Max Power = {max(power)}dB", end = '', flush = True)

            if (it >= initial_sample_number):

                data_scaler = BO.data_scaler(input = input, output = power)

                scaler_input, scaler_output  = data_scaler.minmaxscaler(input_min = input_min, input_max = input_max)

                scaler_data = torch.cat((scaler_input, scaler_output.unsqueeze(1)), dim = 1)

                gp_module = BO.GP_nn(scaler_data)

                gds = BO.gradient_descent_sampling(gp_module = gp_module, data = scaler_data)
                
                scaler_next_input = gds.next_sample()

                next_input = data_scaler.inverse_minmaxscaler(scaler_predicted_input = scaler_next_input)

                next_input = torch.round(next_input * 1000) / 1000
                next_input[2] = torch.round(next_input[2])

                next_input = next_input.unsqueeze(0)
                
                input = torch.cat((input, next_input), dim = 0)

        max_input = input[torch.argmax(power).item()]

        self.qs.x[0] = float(max_input[0])
        self.qs.wait_until_stopped()
        self.qs.x[1] = float(max_input[1])
        self.qs.wait_until_stopped()

        laser.set_laser_wavelength(wavelength = max_input[2])

        return input.tolist(), power.tolist(), max_input.tolist(), max(power)

    def gp_optimization(self, iteration_time = 50, scan_range = 1):

        current_position_x = self.qs.x[0]
        current_position_y = self.qs.x[1]

        current_power = self.powermeter.measure()

        position = torch.Tensor([[current_position_x, current_position_y]])
        power =  torch.Tensor([[current_power]])

        next_position = 0.002 * torch.rand(2) - 0.004

        next_position[0] = current_position_x + next_position[0]
        next_position[1] = current_position_y + next_position[1]

        next_position = next_position.unsqueeze(0)

        self.qs.x[0] = float(next_position[0][0])
        self.qs.wait_until_stopped()
        self.qs.x[1] = float(next_position[0][1])
        self.qs.wait_until_stopped()

        time.sleep(0.5)

        current_power = self.powermeter.measure()

        position = torch.cat((position, next_position), dim = 0)
        power = torch.cat((power, torch.Tensor([[current_power]])), dim = 0)

        optimizer = GP.AdamOptimizer(lr = 0.05)

        for it in range(iteration_time):

            data_scaler = GP.data_scaler(input = position, output = power)

            scaler_input, scaler_output  = data_scaler.minmaxscaler(input_min = torch.tensor([current_position_x - scan_range, current_position_y - scan_range]), input_max = torch.tensor([current_position_x + scan_range, current_position_y + scan_range]))

            scaler_next_input = optimizer.next_input_max(input_0 = scaler_input[it], input = scaler_input[it + 1], output_0 = scaler_output[it], output = scaler_output[it + 1])

            next_position = data_scaler.inverse_minmaxscaler(scaler_predicted_input = scaler_next_input)
            
            next_position = next_position.unsqueeze(0)

            # if (it > 0 and torch.equal(next_angle_0, next_position)):

            #     break

            # next_angle_0 = next_position

            position = torch.cat((position, next_position), dim = 0)

            self.qs.x[0] = float(next_position[0][0])
            self.qs.wait_until_stopped()
            self.qs.x[1] = float(next_position[0][1])
            self.qs.wait_until_stopped()

            time.sleep(0.5)

            current_power = self.powermeter.measure()

            power = torch.cat((power, torch.Tensor([[current_power]])), dim = 0)

            print(it, next_position[0].tolist(), 1000*(position[it + 1] - position[it]), current_power)

        if (current_power < self.scan_threshold_dB):

            self.qs.x[0] = current_position_x
            self.qs.x[1] = current_position_y

        return position.tolist(), power.tolist()
    
    def gp_one_point_optimization(self, iteration_time = 50, scan_range = 5):

        self.qs.set_value(0, 'USTEP', 7)
        self.qs.set_value(1, 'USTEP', 7)

        current_position_x = self.qs.x[0] + 0.0001
        current_position_y = self.qs.x[1] + 0.0001

        self.qs.x[0] = float(current_position_x)
        self.qs.x[1] = float(current_position_y)

        current_power = self.powermeter.measure()

        position = torch.Tensor([[current_position_x, current_position_y]])
        power =  torch.Tensor([[current_power]])

        optimizer = GP.AdamOptimizer(lr = 0.1)

        for it in range(iteration_time):

            position_sample, direction = optimizer.random_direction(input_0 = position[it], epsilon = 0.0001)

            self.qs.x[0] = float(position_sample[0])
            self.qs.x[1] = float(position_sample[1])

            power_sample = self.powermeter.measure()

            time.sleep(0.5)

            all_input = torch.cat((position, position_sample.unsqueeze(0)))
            all_output = torch.cat((power, torch.Tensor([[power_sample]])), dim = 0)

            data_scaler = GP.data_scaler(input = all_input, output = all_output)

            scaler_input, scaler_output  = data_scaler.minmaxscaler(input_min = torch.tensor([current_position_x - scan_range, current_position_y - scan_range]), input_max = torch.tensor([current_position_x + scan_range, current_position_y + scan_range]))

            scaler_next_input = optimizer.one_sample_next_input_max(input_0 = scaler_input[it], input_sample = scaler_input[it + 1], direction = direction, output_0 = scaler_output[it], output_sample = scaler_output[it + 1], epsilon = 0.0001)

            next_position = data_scaler.inverse_minmaxscaler(scaler_predicted_input = scaler_next_input)

            position = torch.cat((position, next_position.unsqueeze(0)), dim = 0)

            self.qs.x[0] = float(next_position[0])
            self.qs.x[1] = float(next_position[1])

            time.sleep(0.5)

            current_power = self.powermeter.measure()

            power = torch.cat((power, torch.Tensor([[current_power]])), dim = 0)

            print(it, next_position.tolist(), current_power)

        if (current_power < self.scan_threshold_dB):

            self.qs.x[0] = current_position_x
            self.qs.x[1] = current_position_y

        return position.tolist(), power.tolist()
    
    def hill_climbing_layer(self, paddle_control, polarization, layer = 3, wavelength_optimization = False, laser = False):

        save_power = []

        for l in range(layer):

            layer_power = []

            for n in range(2):

                loop_condition = 0
                direction = 1
                initial_current_power = self.powermeter.measure()

                while True:

                    current_position = self.qs.x[n]
                    current_power = self.powermeter.measure()

                    self.qs.x[n] = current_position + direction * 1/2**(l + 1)

                    self.qs.wait_until_stopped()
                    time.sleep(1)

                    power = self.powermeter.measure()

                    if (current_power > power):

                        self.qs.x[n] = current_position
                        direction = direction * -1
                        loop_condition = loop_condition + 1

                    if (loop_condition >= 6 or power > initial_current_power):

                        break

            current_power = self.powermeter.measure()
            layer_power.append(current_power)
            print(f"Layer: {l}, Position Optimization, Power = {current_power}dB")

            # _, _, _, max_power = polarization.scan_optimization(step = 2 * (layer - l))
            # layer_power.append(max_power)
            # print(f"Layer: {l}, Polarization Optimization, Power = {max_power}dB")

            for n in range(3):

                loop_condition = 0
                direction = 1
                initial_current_power = self.powermeter.measure()

                while True:

                    current_position = paddle_control.read_current_position()
                    current_power = self.powermeter.measure()

                    paddle_control.move_to(paddle_number = n + 1, position = int(polarization.restrict_angle(torch.tensor([current_position[n] + direction * 2 * (layer - l)]))))

                    time.sleep(1)

                    power = self.powermeter.measure()

                    if (current_power > power):

                        paddle_control.move_to(paddle_number = n + 1, position = int(current_position[n]))
                        direction = direction * -1
                        loop_condition = loop_condition + 1

                    if (loop_condition >= 6 or power > initial_current_power):

                        break

            current_power = self.powermeter.measure()
            layer_power.append(current_power)
            print(f"Layer: {l}, Polarization Optimization, Power = {current_power}dB")

            if wavelength_optimization:

                loop_condition = 0
                direction = 1
                initial_current_power = self.powermeter.measure()

                while True:

                    current_wavelength = laser.get_laser_wavelength()
                    current_power = self.powermeter.measure()

                    laser.set_laser_wavelength(wavelength = current_wavelength + direction * (layer - l))

                    time.sleep(1)

                    power = self.powermeter.measure()

                    if (current_power > power):

                        laser.set_laser_wavelength(wavelength = current_wavelength)
                        direction = direction * -1
                        loop_condition = loop_condition + 1

                    if (loop_condition >= 6 or power > initial_current_power):

                        break

                current_power = self.powermeter.measure()
                layer_power.append(current_power)
                print(f"Layer: {l}, Wavelegnth Optimization, Power = {current_power}dB")

            save_power.append(layer_power)

        return save_power

    def wavelength_range_scan(self, file_processing, laser, powermeter, wavelength_start, wavelength_stop, wavelength_step, scan_range = 50, scan_step = 3):

        current_position_x = self.qs.x[0]
        current_position_y = self.qs.x[1]

        x_range = torch.arange(-scan_range, scan_range + scan_step, scan_step)
        y_range = torch.arange(scan_range, -scan_range - scan_step, -scan_step)

        x_position, y_posiotion = torch.meshgrid(x_range, y_range, indexing='ij')

        x_position = x_position.flatten().unsqueeze(-1)
        y_posiotion = y_posiotion.flatten().unsqueeze(-1)

        position = torch.cat((x_position, y_posiotion), dim = 1)

        # scan_x = []
        # scan_y = []

        for n in range(len(position)):

            self.qs.x[0] = current_position_x + float(position[n][0])
            self.qs.wait_until_stopped()
            self.qs.x[1] = current_position_y + float(position[n][1])
            self.qs.wait_until_stopped()

            # scan_x.append(self.qs.x[0])
            # scan_y.append(self.qs.x[1])

            wavelength, power = wavelength_scan(laser = laser, powermeter = powermeter, wavelength_start = wavelength_start, wavelength_stop = wavelength_stop, wavelength_step = wavelength_step)

            file_processing.save_data_csv(wavelength = wavelength, power = power, id = n)

        # return scan_x, scan_y

class rough_position():

    def __init__(self, powermeter, linear_actuator_x, linear_actuator_y, scan_threshold_dB = -37):

        super(rough_position, self).__init__()

        self.powermeter = powermeter

        self.linear_actuator_x = linear_actuator_x
        self.linear_actuator_y = linear_actuator_y

        self.inital_position_x = self.linear_actuator_x.read_current_position()
        self.inital_position_y = self.linear_actuator_y.read_current_position()

        self.scan_threshold_dB = scan_threshold_dB

    def range_scan(self, scan_range = 40, scan_step = 20, move = True):

        current_position_x = self.linear_actuator_x.read_current_position()
        current_position_y = self.linear_actuator_y.read_current_position()

        x_range = torch.arange(-scan_range, scan_range + scan_step, scan_step)
        y_range = torch.arange(-scan_range, scan_range + scan_step, scan_step)

        x_position, y_posiotion = torch.meshgrid(x_range, y_range, indexing='ij')

        x_position = x_position.flatten().unsqueeze(-1)
        y_posiotion = y_posiotion.flatten().unsqueeze(-1)

        position = torch.cat((x_position, y_posiotion), dim = 1)

        move = position[1:] - position[:-1]

        move = torch.cat((position[0].unsqueeze(0), move), dim = 0).tolist()

        power = []

        for n in range(len(position)):

            self.linear_actuator_x.move_by(distance_um = move[n][0])
            self.linear_actuator_y.move_by(distance_um = move[n][1])

            time.sleep(1)

            current_power = self.powermeter.measure()

            power.append(current_power)

            print(f"\rProgressing: {int(((n + 1)/len(position)) * 100)}%, Max Power = {max(power)}dB", end = '', flush = True)

        print()

        power = torch.tensor(power)
            
        max_power = power.max()

        max_index = torch.argmax(power).item()

        self.linear_actuator_x.move_to(distance_um = current_position_x)
        self.linear_actuator_y.move_to(distance_um = current_position_y)

        if (move and  max_power >= self.scan_threshold_dB):

            self.linear_actuator_x.move_by(distance_um = float(position[max_index][0]))
            self.linear_actuator_y.move_by(distance_um = float(position[max_index][1]))

            return position.tolist(), power, position[max_index].tolist()
        
        else:

            return position.tolist(), power, position[max_index].tolist()
        
    def bo_optimization(self, iteration_time = 30, scan_range = 40):

        current_position_x = self.linear_actuator_x.read_current_position()
        current_position_y = self.linear_actuator_y.read_current_position()

        current_power = self.powermeter.measure()

        position = torch.Tensor([[current_position_x, current_position_y]])
        power =  torch.Tensor([current_power])

        input_min = torch.tensor([current_position_x - scan_range, current_position_y - scan_range])
        input_max = torch.tensor([current_position_x + scan_range, current_position_y + scan_range])

        initial_sample = BO.latin_hypercube_sampling(sample_min = input_min, sample_max = input_max, sample_number = 20)

        position = torch.cat((position, initial_sample), dim = 0)
        power =  torch.Tensor([current_power])

        for it in range(1, iteration_time):

            self.linear_actuator_x.move_to(distance_um = float(position[it][0]))
            self.linear_actuator_y.move_to(distance_um = float(position[it][1]))

            time.sleep(0.5)

            current_power = self.powermeter.measure()

            power = torch.cat((power, torch.Tensor([current_power])), dim = 0)    

            print(f"\rProgressing: {int(((it + 1)/iteration_time) * 100)}%, Max Power = {max(power)}dB", end = '', flush = True)

            if (it >= 20):

                data_scaler = BO.data_scaler(input = position, output = power)

                scaler_input, scaler_output  = data_scaler.minmaxscaler(input_min = input_min, input_max = input_max)

                scaler_data = torch.cat((scaler_input, scaler_output.unsqueeze(1)), dim = 1)

                gp_module = BO.GP_nn(scaler_data)

                gds = BO.gradient_descent_sampling(gp_module = gp_module, data = scaler_data)
                
                scaler_next_input = gds.next_sample()

                next_position = data_scaler.inverse_minmaxscaler(scaler_predicted_input = scaler_next_input)

                next_position = torch.round(next_position, decimals = 1).unsqueeze(0)

                position = torch.cat((position, next_position), dim = 0)

            # self.linear_actuator_x.move_to(distance_um = float(next_position[0][0]))
            # self.linear_actuator_y.move_to(distance_um = float(next_position[0][1]))

            # time.sleep(1)

            # current_power = self.powermeter.measure()

            # position = torch.cat((position, next_position), dim = 0)
            # power = torch.cat((power, torch.Tensor([current_power])), dim = 0)

            # print(f"\rProgressing: {int(((it + 1)/iteration_time) * 100)}%, Max Power = {max(power)}dB", end = '', flush = True)

        print()

        max_power = power.max()

        max_index = torch.argmax(power).item()

        if (max_power >= self.scan_threshold_dB):

            self.linear_actuator_x.move_to(distance_um = float(position[max_index][0]))
            self.linear_actuator_y.move_to(distance_um = float(position[max_index][1]))

            return position.tolist(), power, (position[max_index] - torch.tensor([current_position_x, current_position_y])).tolist()
        
        else:

            self.linear_actuator_x.move_to(distance_um = current_position_x)
            self.linear_actuator_y.move_to(distance_um = current_position_y)

            return position.tolist(), power, (position[max_index] - torch.tensor([current_position_x, current_position_y])).tolist()
        
    def gp_optimization(self, iteration_time = 200, scan_range = 20):

        current_position_x = self.linear_actuator_x.read_current_position()
        current_position_y = self.linear_actuator_y.read_current_position()

        current_power = self.powermeter.measure()

        position = torch.Tensor([[current_position_x, current_position_y]])
        power =  torch.Tensor([current_power])

        next_position = position + torch.randint(low = -5, high = 6, size = (2,))

        position = torch.cat((position, next_position), dim = 0)

        self.linear_actuator_x.move_to(distance_um = float(next_position[0][0]))
        self.linear_actuator_y.move_to(distance_um = float(next_position[0][1]))

        time.sleep(0.5)

        current_power = self.powermeter.measure()

        power = torch.cat((power, torch.Tensor([current_power])), dim = 0)

        optimizer = GP.AdamOptimizer(lr = 0.001)

        for it in range(iteration_time):

            data_scaler = GP.data_scaler(input = position, output = power)

            scaler_input, scaler_output  = data_scaler.minmax_input_standard_output_scaler(input_min = torch.tensor([current_position_x - scan_range, current_position_y - scan_range]), input_max = torch.tensor([current_position_x + scan_range, current_position_y + scan_range]))

            scaler_next_input = optimizer.next_input_max(input_0 = scaler_input[it], input = scaler_input[it + 1], output_0 = scaler_output[it], output = scaler_output[it + 1])

            next_position = data_scaler.inverse_minmax_input_standard_output_scaler(scaler_predicted_input = scaler_next_input)
            
            next_position = next_position.unsqueeze(0)

            # if (it > 0 and torch.equal(next_angle_0, next_position)):

            #     break

            # next_angle_0 = next_position

            position = torch.cat((position, next_position), dim = 0)

            next_position.squeeze().tolist()

            self.linear_actuator_x.move_to(distance_um = float(next_position[0][0]))
            self.linear_actuator_y.move_to(distance_um = float(next_position[0][1]))

            time.sleep(0.5)

            current_power = self.powermeter.measure()

            power = torch.cat((power, torch.Tensor([current_power])), dim = 0)

            print(it, next_position[0].tolist(), current_power)

        if (current_power < self.scan_threshold_dB):

            self.linear_actuator_x.move_to(distance_um = current_position_x)
            self.linear_actuator_y.move_to(distance_um = current_position_y)

        return position.tolist(), power.tolist()

class file_processing():

    def __init__(self, path):

        super(file_processing, self).__init__()

        self.path = path

        self.current_time = time.strftime("%Y_%m_%d_%H_%M")

        os.makedirs(self.path + "\\" + self.current_time, exist_ok=True)

    def read_csv(self, csv_name):

        shutil.copy(self.path + "\\" + csv_name + ".csv", self.path + "\\" + self.current_time)

        with open(self.path + "\\" + csv_name + ".csv", 'r', encoding='utf-8') as file:

            reader = csv.DictReader(file)

            data = list(reader)

            id, device_name, port_x_position_um, port_y_position_um, number_of_channels, optimisation_channel, wavelength_start_nm, wavelength_stop_nm, steps, polarization_wavelength = [], [], [], [], [], [], [], [], [], []

            for _, row in enumerate(data):

                id.append(int(row['id']))
                device_name.append(int(row['device_name']))
                port_x_position_um.append(int(row['port_x_position_um']))
                port_y_position_um.append(int(row['port_y_position_um']))
                number_of_channels.append(int(row['number_of_channels']))
                optimisation_channel.append(int(row['optimisation_channel']))
                wavelength_start_nm.append(int(row['wavelength_start_nm']))
                wavelength_stop_nm.append(int(row['wavelength_stop_nm']))
                steps.append(int(row['steps']))
                polarization_wavelength.append(int(row['polarization_wavelength']))

        return id, device_name, port_x_position_um, port_y_position_um, number_of_channels, optimisation_channel, wavelength_start_nm, wavelength_stop_nm, steps, polarization_wavelength

    def save_data_csv(self, wavelength, power, id):

        labels = [['Wavelength (nm)', 'Power (dBm)']]
        data = np.column_stack((wavelength, power))

        csv_content = np.vstack((labels, data))

        with open(self.path + "\\" + self.current_time + '\\id_' + str(id) + '.csv', "w", newline = "") as file:

            writer = csv.writer(file)
            writer.writerows(csv_content)

    def plot_raw_data(self, x, y, id):

        plt.plot(x, y)
        plt.title(id)
        plt.xlabel("Wavelength")
        plt.ylabel("Power")

        # plt.show()

        plt.savefig(self.path + '\\id_' + id + '.png')
    
def wavelength_scan(laser, powermeter, wavelength_start = 1500, wavelength_stop = 1600, wavelength_number = 101):

    wavelength = np.linspace(wavelength_start, wavelength_stop, wavelength_number)
    laser.set_laser_wavelength(wavelength[0])
    time.sleep(10)

    power = []

    for n in range(len(wavelength)):

        laser.set_laser_wavelength(wavelength[n])
        time.sleep(0.5)

        current_power = powermeter.measure()

        power.append(current_power)

        print(f"\rProgressing: {int(((n + 1)/len(wavelength)) * 100)}%, Wavelegnth = {wavelength[n]}nm, Power = {power[n]}dB", end = '', flush = True)

    print()

    return wavelength, power

def wavelength_optimization(laser, powermeter, step = 5):

    current_wavelength = laser.get_laser_wavelength()

    wavelength = torch.arange(current_wavelength - 3 * step, current_wavelength + 4 * step, step)

    power = []

    for n in range(len(wavelength)):
                
        laser.set_laser_wavelength(wavelength = wavelength(n))

        time.sleep(1)
                
        current_power = powermeter.measure()

        power.append(current_power)

        print(f"\rProgressing: {int(((n + 1)/len(wavelength)) * 100)}%, Max Power = {max(power)}dB", end = '', flush = True)

    print()

    power = torch.tensor(power)

    max_position = wavelength[torch.argmax(power).item()]
    
    laser.set_laser_wavelength(wavelength = max_position)

    return max_position