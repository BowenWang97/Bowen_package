import csv
import gradient_based
import matplotlib.pyplot as plt
import numpy as np
import time
import torch

class polarization():

    def __init__(self, powermeter, paddle_control, polarization_threshold_dB = -27):

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
    
    def scan_optimization(self, max_position, step = 10):

        range_angle_1 = torch.arange(max_position[0] - 2 * step, max_position[0] + 3 * step, step)
        range_angle_2 = torch.arange(max_position[1] - 2 * step, max_position[1] + 3 * step, step)
        range_angle_3 = torch.arange(max_position[2] - 2 * step, max_position[2] + 3 * step, step)

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
                
            current_power = self.powermeter.measure()

            power.append(current_power)

        power = torch.tensor(power)

        max_position = paddle_angle[torch.argmax(power).item()]

        print(max_position)

        for paddle_number in range(3):
        
            self.paddle_control.move_to(paddle_number=(paddle_number + 1), position = max_position[paddle_number])

        return max_position

    def optimization(self, iteration_time = 100, epsilon = 15):

        current_angle = self.paddle_control.read_current_position()

        current_power = self.powermeter.measure()

        position = torch.Tensor([current_angle])
        power =  torch.Tensor([current_power])

        next_angle = position - epsilon + 2 * epsilon * torch.rand(3)

        next_angle = self.restrict_angle(angle = next_angle)

        for paddle_number in range(3):

            self.paddle_control.move_to(paddle_number = (paddle_number + 1), position = next_angle[paddle_number])

        time.sleep(2)

        current_power = self.powermeter.measure()

        position = torch.cat((position, torch.Tensor([next_angle])), dim = 0)
        power = torch.cat((power, torch.Tensor([current_power])), dim = 0)

        optimizer = gradient_based.AdamOptimizer(lr = 0.05)

        # convert_time = 0

        for it in range(iteration_time):

            data_scaler = gradient_based.data_scaler(input = position, output = power)

            scaler_input, scaler_output  = data_scaler.minmax_input_standard_output_scaler(input_min = torch.tensor([0., 0., 0.]), input_max = torch.tensor([170., 170., 170.]))

            scaler_next_input = optimizer.next_input_max(input_0 = scaler_input[it], input = scaler_input[it + 1], output_0 = scaler_output[it], output = scaler_output[it + 1])

            next_angle = data_scaler.inverse_minmax_input_standard_output_scaler(scaler_predicted_input = scaler_next_input)

            next_angle = self.restrict_angle(angle = next_angle)

            if (it > 0 and next_angle_0 == next_angle):

                # print(next_angle_0, next_angle)

                # convert_time = convert_time + 1

                # for d in range(3):

                #     if (next_angle_1[d] >= next_angle_0[d]):

                #         next_angle[d] = int(next_angle[d] + np.random.randint(1, 3, 1))

                #     else:

                #         next_angle[d] = int(next_angle[d] - np.random.randint(1, 3, 1))

                # if (convert_time >= 5):

                break

            # if (it > 0):

            #     next_angle_1 = next_angle_0

            next_angle_0 = next_angle

            for paddle_number in range(3):

                self.paddle_control.move_to(paddle_number = (paddle_number + 1), position = next_angle[paddle_number])

            time.sleep(2)

            current_power = self.powermeter.measure()

            position = torch.cat((position, torch.Tensor([next_angle])), dim = 0)
            power = torch.cat((power, torch.Tensor([current_power])), dim = 0)

            print(it, next_angle, position[it + 1] - position[it], current_power)

        return position.tolist(), power.tolist()
            
    def shift_polarization(self):

        current_angle = self.paddle_control.read_current_position()

        current_angle = torch.tensor(current_angle)

        range_angle = torch.arange(-45, 67.5, 22.5)

        paddle_1_angle, paddle_2_angle, paddle_3_angle = torch.meshgrid(range_angle, range_angle, range_angle, indexing='ij')

        paddle_1_angle = paddle_1_angle.flatten().unsqueeze(-1) + current_angle[0]
        paddle_2_angle = paddle_2_angle.flatten().unsqueeze(-1) + current_angle[1]
        paddle_3_angle = paddle_3_angle.flatten().unsqueeze(-1) + current_angle[2]

        paddle_angle = torch.cat((paddle_1_angle, paddle_2_angle, paddle_3_angle), dim = 1).tolist()

        power = []

        for n in range(125):

            for paddle_number in range(3):

                self.paddle_control.move_to(paddle_number = (paddle_number + 1), position = paddle_angle[n][paddle_number])

            current_power = self.powermeter.measure()

            power.append(current_power)

        return paddle_angle, power
    
class precise_position():

    def __init__(self, powermeter, qs, scan_threshold_dB = -33):

        super(precise_position, self).__init__()

        self.powermeter = powermeter

        self.qs = qs
        self.qs.response_timeout = 10

        self.scan_threshold_dB = scan_threshold_dB

    def range_scan(self, scan_range = 20, scan_step = 1, move = True):

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
            self.qs.x[1] = current_position_y + float(position[n][1])

            current_power = self.powermeter.measure()

            power.append(current_power)

        power = torch.tensor(power)
            
        max_power = power.max()

        max_index = torch.where(power == max_power)[0].item()

        self.qs.x[0] = current_position_x
        self.qs.x[1] = current_position_y

        if (move):

            self.qs.x[0] = current_position_x + float(position[max_index][0])
            self.qs.x[1] = current_position_y + float(position[max_index][1])

            return position.tolist(), power, position[max_index].tolist()
        
        else:

            return position.tolist(), power, position[max_index].tolist()

    def optimization(self, iteration_time = 100, scan_range = 0.01):

        self.qs.set_value(0, 'USTEP', 7)
        self.qs.set_value(1, 'USTEP', 7)

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
        self.qs.x[1] = float(next_position[0][1])

        time.sleep(2)

        current_power = self.powermeter.measure()

        position = torch.cat((position, next_position), dim = 0)
        power = torch.cat((power, torch.Tensor([[current_power]])), dim = 0)

        optimizer = gradient_based.AdamOptimizer(lr = 0.05)

        for it in range(iteration_time):

            data_scaler = gradient_based.data_scaler(input = position, output = power)

            scaler_input, scaler_output  = data_scaler.minmaxscaler(input_min = torch.tensor([current_position_x - scan_range, current_position_y - scan_range]), input_max = torch.tensor([current_position_x + scan_range, current_position_y + scan_range]))

            scaler_next_input = optimizer.next_input_max(input_0 = scaler_input[it], input = scaler_input[it + 1], output_0 = scaler_output[it], output = scaler_output[it + 1])

            next_position = data_scaler.inverse_minmaxscaler(scaler_predicted_input = scaler_next_input)
            
            next_position = next_position.unsqueeze(0)

            # if (it > 0 and torch.equal(next_angle_0, next_position)):

            #     break

            # next_angle_0 = next_position

            position = torch.cat((position, next_position), dim = 0)

            self.qs.x[0] = float(next_position[0][0])
            self.qs.x[1] = float(next_position[0][1])

            time.sleep(2)

            current_power = self.powermeter.measure()

            power = torch.cat((power, torch.Tensor([[current_power]])), dim = 0)

            print(it, next_position[0].tolist(), position[it + 1] - position[it], current_power)

        if (current_power < self.scan_threshold_dB):

            self.qs.x[0] = current_position_x
            self.qs.x[1] = current_position_y

        return position.tolist(), power.tolist()
    
    def one_point_optimization(self, iteration_time = 100, scan_range = 5):

        self.qs.set_value(0, 'USTEP', 7)
        self.qs.set_value(1, 'USTEP', 7)

        current_position_x = self.qs.x[0]
        current_position_y = self.qs.x[1]

        current_power = self.powermeter.measure()

        position = torch.Tensor([[current_position_x, current_position_y]])
        power =  torch.Tensor([[current_power]])

        optimizer = gradient_based.AdamOptimizer(lr = 0.01)

        for it in range(iteration_time):

            position_sample, direction = optimizer.random_direction(input_0 = position[it], epsilon = 0.0001)

            print(position_sample)

            self.qs.x[0] = float(position_sample[0])
            self.qs.x[1] = float(position_sample[1])

            power_sample = self.powermeter.measure()

            time.sleep(0.1)

            all_input = torch.cat((position, position_sample.unsqueeze(0)))
            all_output = torch.cat((power, torch.Tensor([[power_sample]])), dim = 0)

            data_scaler = gradient_based.data_scaler(input = all_input, output = all_output)

            scaler_input, scaler_output  = data_scaler.minmaxscaler(input_min = torch.tensor([current_position_x - scan_range, current_position_y - scan_range]), input_max = torch.tensor([current_position_x + scan_range, current_position_y + scan_range]))

            scaler_next_input = optimizer.one_sample_next_input_max(input_0 = scaler_input[it], input_sample = scaler_input[it + 1], direction = direction, output_0 = scaler_output[it], output_sample = scaler_output[it + 1], epsilon = 0.0001)

            next_position = data_scaler.inverse_minmaxscaler(scaler_predicted_input = scaler_next_input)

            position = torch.cat((position, next_position.unsqueeze(0)), dim = 0)

            self.qs.x[0] = float(next_position[0])
            self.qs.x[1] = float(next_position[1])

            time.sleep(0.1)

            current_power = self.powermeter.measure()

            power = torch.cat((power, torch.Tensor([[current_power]])), dim = 0)

            print(next_position.tolist(), current_power)

        if (current_power < self.scan_threshold_dB):

            self.qs.x[0] = current_position_x
            self.qs.x[1] = current_position_y

        return position.tolist(), power.tolist()

class rough_position():

    def __init__(self, powermeter, linear_actuator_x, linear_actuator_y, scan_threshold_dB = -33):

        super(rough_position, self).__init__()

        self.powermeter = powermeter

        self.linear_actuator_x = linear_actuator_x
        self.linear_actuator_y = linear_actuator_y

        self.inital_position_x = self.linear_actuator_x.read_current_position()
        self.inital_position_y = self.linear_actuator_y.read_current_position()

        self.scan_threshold_dB = scan_threshold_dB

    def range_scan(self, scan_range = 50, scan_step = 10, move = True):

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

            time.sleep(2)

            current_power = self.powermeter.measure()

            power.append(current_power)

        power = torch.tensor(power)
            
        max_power = power.max()

        max_index = torch.where(power == max_power)[0].item()

        self.linear_actuator_x.move_to(distance_um = current_position_x)
        self.linear_actuator_y.move_to(distance_um = current_position_y)

        if (move and  max_power >= self.scan_threshold_dB):

            self.linear_actuator_x.move_by(distance_um = float(position[max_index][0]))
            self.linear_actuator_y.move_by(distance_um = float(position[max_index][1]))

            return position.tolist(), power, position[max_index].tolist()
        
        else:

            return position.tolist(), power, position[max_index].tolist()
        
    def optimization(self, iteration_time = 200, scan_range = 20):

        current_position_x = self.linear_actuator_x.read_current_position()
        current_position_y = self.linear_actuator_y.read_current_position()

        current_power = self.powermeter.measure()

        position = torch.Tensor([[current_position_x, current_position_y]])
        power =  torch.Tensor([current_power])

        next_position = position + torch.randint(low = -5, high = 6, size = (2,))

        position = torch.cat((position, next_position), dim = 0)

        self.linear_actuator_x.move_to(distance_um = float(next_position[0][0]))
        self.linear_actuator_y.move_to(distance_um = float(next_position[0][1]))

        time.sleep(2)

        current_power = self.powermeter.measure()

        power = torch.cat((power, torch.Tensor([current_power])), dim = 0)

        optimizer = gradient_based.AdamOptimizer(lr = 0.001)

        for it in range(iteration_time):

            data_scaler = gradient_based.data_scaler(input = position, output = power)

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

            time.sleep(2)

            current_power = self.powermeter.measure()

            power = torch.cat((power, torch.Tensor([current_power])), dim = 0)

            print(it, next_position[0].tolist(), current_power)

        if (current_power < self.scan_threshold_dB):

            self.linear_actuator_x.move_to(distance_um = current_position_x)
            self.linear_actuator_y.move_to(distance_um = current_position_y)

        return position.tolist(), power.tolist()
    
def wavelength_scan(laser, powermeter, wavelegnth_start = 1500, wavelegnth_stop = 1600, wavelegnth_step = 101):

    wavelength = np.linspace(wavelegnth_start, wavelegnth_stop, wavelegnth_step)
    laser.set_laser_wavelength(wavelength[0])
    time.sleep(10)

    power = []

    for w in wavelength:

        laser.set_laser_wavelength(w)
        time.sleep(1)

        current_power = powermeter.measure()

        power.append(current_power)

    return wavelength, power

def save_data_csv(path, wavelength, power, file_name):

    labels = [['Wavelength (nm)', 'Power (dBm)']]
    data = np.column_stack((wavelength, power))

    csv_content = np.vstack((labels, data))

    with open(path + '/' + file_name + '.csv', "w", newline = "") as file:

        writer = csv.writer(file)
        writer.writerows(csv_content)

def plot_raw_data(path, x, y, figure_name):

    plt.plot(x, y)
    plt.title(figure_name)
    plt.xlabel("Wavelength")
    plt.ylabel("Power")

    # plt.show()

    plt.savefig(path + '/' + figure_name + '.png')