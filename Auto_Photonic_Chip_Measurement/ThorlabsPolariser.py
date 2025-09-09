# Author: James Blatcher
# Date: August 2024
# Based on code found in: https://github.com/Thorlabs/Motion_Control_Examples/tree/main/Python/Integrated%20Stages/Polarization%20Controller

# required python packages: System, pythonnet, (clr? if it doesn't work, remove clear and just use pythonnet)

import __init__
from Powermeter import *

import clr
import time

clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.DeviceManagerCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.GenericMotorCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\ThorLabs.MotionControl.PolarizerCLI.dll")

from Thorlabs.MotionControl.DeviceManagerCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.PolarizerCLI import *
from System import Decimal, Convert

class PolarisationController(object):

	def __init__(self, serial_number:int, move_sleep_time:int = 0, home_paddle = True):

		self.serial_number = str(serial_number)

		self.init_timeout = 10000
		self.query_timeout = 10000
		self.home_timeout = 60000
		self.move_timeout = 60000
		self.move_sleep_time = move_sleep_time
		self.home_sleep_time = 2

		try:

			# Build device list
			DeviceManagerCLI.BuildDeviceList()
			# print(DeviceManagerCLI.GetDeviceList())

			# Define the device
			self.device = Polarizer.CreatePolarizer(self.serial_number)

			# Connect to the device
			self.device.Connect(self.serial_number)

			# Ensure that the device settings have been initialised
			if not self.device.IsSettingsInitialized():
				# Wait for initialisation, with a 10 second timeout
				self.device.WaitForSettingsInitialized(self.init_timeout)
				assert self.device.IsSettingsInitialized() is True

			# Start polling loop and enable device.
			self.device.StartPolling(250)  #250ms polling rate.
			time.sleep(5)
			self.device.EnableDevice()
			time.sleep(0.25)  # Wait for device to enable.

			# Get Device Information and display description.
			device_info = self.device.GetDeviceInfo()
			print(f'Connected to Thorlabs {device_info.Description} with serial no. {self.serial_number}')
			# print(f'\ndevice_info = {device_info}')
			
			# Call device methods
			
			self.paddle1 = PolarizerPaddles.Paddle1
			self.paddle2 = PolarizerPaddles.Paddle2
			self.paddle3 = PolarizerPaddles.Paddle3

			# Home the polarisation paddles with a 60 second timeout

			if (home_paddle == True):
				
				print("Homing polarisation paddles...")
				self.device.Home(self.paddle1, self.home_timeout)
				self.device.Home(self.paddle2, self.home_timeout)
				self.device.Home(self.paddle3, self.home_timeout)
				time.sleep(self.home_sleep_time)

				print("Paddle homing complete!\n")

		except Exception as e:
			print(e)

	def read_current_position(self):

		# Get the polarisation paddle positions, returns a System.Decimal object
		paddle_1_position_decimal = self.device.Position(self.paddle1)
		paddle_2_position_decimal = self.device.Position(self.paddle2)
		paddle_3_position_decimal = self.device.Position(self.paddle3)

		# Convert position values to floats, to be used elsewhere
		paddle_1_position_float = Convert.ToDouble(paddle_1_position_decimal)
		paddle_2_position_float = Convert.ToDouble(paddle_2_position_decimal)
		paddle_3_position_float = Convert.ToDouble(paddle_3_position_decimal)

		current_position = [paddle_1_position_float, paddle_2_position_float, paddle_3_position_float]
		# print(f'current_position = {current_position}')
		return current_position

	def move_to(self, paddle_number:int, position):
		new_position = Decimal(position)

		if (0 <= position <= 170):
			if paddle_number == 1:
				# print(f'Moving paddle {paddle_number} to {new_position}...')
				self.device.MoveTo(new_position, self.paddle1, self.move_timeout)
				time.sleep(self.move_sleep_time)
			elif paddle_number == 2:
				# print(f'Moving paddle {paddle_number} to {new_position}...')
				self.device.MoveTo(new_position, self.paddle2, self.move_timeout)
				time.sleep(self.move_sleep_time)
			elif paddle_number == 3:
				# print(f'Moving paddle {paddle_number} to {new_position}...')
				self.device.MoveTo(new_position, self.paddle3, self.move_timeout)
				time.sleep(self.move_sleep_time)
			else:
				print(f'Error: incorrect paddle number {paddle_number}. Available paddle numbers [1, 2, 3]')
		else:
			print(f'Error: invalid angle positional angle for polarisation paddle [{new_position}].\nAvailable angles: 0 - 170')
	
	# def __del__(self):
	#     # Stop polling loop and disconnect device before program finishes. 
	#     self.device.StopPolling()
	#     self.device.Disconnect()

if __name__ == '__main__':



	powermeter1_port = 'USB0::0x1313::0x8078::P0010441::INSTR'

	power_meter = ThorlabsPowermeterWavelengthSweep(port=powermeter1_port)
	power_meter.open_power_meter()


	pol_paddles = PolarisationController(serial_number=38229054, move_sleep_time=0)

	### Example on moving paddles and moving to invalid position
	# pol_paddles.move_to(paddle_number=1, position=-10)
	# pol_paddles.move_to(paddle_number=1, position=10)
	# pol_paddles.move_to(paddle_number=2, position=20)
	# pol_paddles.move_to(paddle_number=3, position=30)

	### Example for optimisation ###

	def polarisation_optimisation():
		for paddle_num in range(3):
			powers = []

			for degree in range(171):
				pol_paddles.move_to(paddle_number=(paddle_num+1), position=degree)
				### Read PM and log value & degree (paddle position) - could use the index of the list as the degree
				current_power = power_meter.get_power()
				print(f'current power = {current_power}')
				powers.append(current_power)

			max_power_position = powers.index(max(powers))
			pol_paddles.move_to(paddle_number=(paddle_num+1), position=max_power_position)

			### Move to degree that gives highest throughput power - could use the index of the highest PM value in list
			### Contrinue to sweep next paddle
			print(f'\nCompleted optimisation for paddle {(paddle_num+1)}')

		print(f'\n\nCompleted optimisation for all paddles!\nPaddle positions: {pol_paddles.read_current_position()}\nMax throughput power:  {power_meter.get_power()} dBm')

	polarisation_optimisation()

	# # pol_paddles.move_to(1, float(80.2))
	# pol_paddles.move_to(1, float(0))
	# pol_paddles.move_to(2, float(0))
	# pol_paddles.move_to(3, float(0))
	# cur_position = pol_paddles.read_current_position()
	# print(cur_position)
