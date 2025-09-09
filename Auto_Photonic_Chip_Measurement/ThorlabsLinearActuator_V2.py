# Author: James Blatcher
# Date: August 2024
# Based on code found in: https://github.com/Thorlabs/Motion_Control_Examples/blob/main/C%2B%2B/KCube/KDC101/KDC101_Example.cpp

import clr
import time

clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.DeviceManagerCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.GenericMotorCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.TCube.DCServoCLI.dll")

from Thorlabs.MotionControl.DeviceManagerCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.TCube.DCServoCLI import *
from System import Decimal, Convert, Double

class LinearActuator(object):

	def __init__(self, motor_serial_number:int, move_sleep_time:int=2, disable_home:bool = False):
		
		self.serial_number = str(motor_serial_number)

		self.init_timeout = 10000	# 10 s timeout
		self.query_timeout = 10000
		self.home_timeout = 60000
		self.move_timeout = 20000
		self.move_sleep_time = move_sleep_time
		self.home_sleep_time = 2

		self.min_position = -99000
		self.max_position = 99000

		try:

			# Build device list
			DeviceManagerCLI.BuildDeviceList()

			# Define the device
			self.device = TCubeDCServo.CreateTCubeDCServo(self.serial_number)
			# Print a list of available devices
			# print(DeviceManagerCLI.GetDeviceList())

			# Connect to the device
			self.device.Connect(self.serial_number)
			time.sleep(0.25)

			# Start polling loop
			self.device.StartPolling(250)  # 250 ms polling rate.
			time.sleep(0.25)

			# Enable the device
			self.device.EnableDevice()
			time.sleep(0.25)  # Wait for device to enable.

			# Ensure that the device settings have been initialised
			if not self.device.IsSettingsInitialized():
				# Wait for initialisation, with a 10 second timeout
				self.device.WaitForSettingsInitialized(self.init_timeout)
				assert self.device.IsSettingsInitialized() is True

			# Get Device Information and display description.
			device_info = self.device.GetDeviceInfo()
			print(f'Connected to Thorlabs {device_info.Description} with serial no. {self.serial_number}')

			# Before homing or moving device, ensure the motor's configuration is loaded
			m_config = self.device.LoadMotorConfiguration(self.serial_number, DeviceConfiguration.DeviceSettingsUseOptionType.UseFileSettings)

			# m_config.DeviceSettingsName = "PRMTZ8"
			m_config.DeviceSettingsName = "TDC001"

			m_config.UpdateCurrentConfiguration()

			self.device.SetSettings(self.device.MotorDeviceSettings, True, False)
			
			# Home the linear actuator
			if disable_home == False:
				self.home()

		except Exception as e:
			print(e)
			
	def move_to(self, distance_um):
		if (self.min_position <= distance_um <= self.max_position):
			distance_mm = Decimal(distance_um/1000)
			# distance_mm = Double(distance_um/1000)
			# distance = Decimal(distance_um)
			self.device.MoveTo(distance_mm, self.move_timeout)
			# print(f'Moving linear actuator {self.serial_number} to {distance_mm}')
		else:
			print(f'Error: invalid position for linear actuator [{distance_um} um]\nAvailable angles: {self.min_position} um - {self.max_position} um')

	def move_by(self, distance_um):
		# print('DISTANCE TO BE MOVED BY ========= {}um'.format(distance_um))
		if distance_um == 0:
			pass
		else:
		# 	distance_mm = Decimal(distance_um/1000)
		# 	self.device.MoveRelative(distance_mm)#, self.move_timeout)
			### HACK: can't figure out how many arguments or which Type to use for argument(s) - so use the move_to function
			current_position = self.read_current_position()
			new_position = current_position + distance_um

			distance_mm = Decimal(new_position/1000)
			self.device.MoveTo(distance_mm, self.move_timeout)
			
	def read_current_position(self):

		# Get the linear actuator position, returns a System.Decimal object
		lin_act_position_decimal = self.device.get_Position()

		# Convert position values to floats, to be used elsewhere
		lin_act_position_float = Convert.ToDouble(lin_act_position_decimal)

		current_position_um = lin_act_position_float * 1000
		return current_position_um

	def home(self):
		# settings = self.device.GetSettings()
		# print(f'settings = {settings}')

		# Default homing velocity values:
		# Homing velocity = 171.355 for 83852988
		# Homing velocity = 1 for 83845817
		# homing_velocity = self.device.GetHomingVelocity()
		# print(f'homing_velocity = {homing_velocity}')
		# Set the homing velocity to 1

		# if self.serial_number == '83852988':
		# 	homing_velocity = Decimal(171.355)
		# 	self.device.SetHomingVelocity(homing_velocity)
		# 	print(f'Homiung velocity for {83852988} set to {homing_velocity}')
		# elif self.serial_number == '83845817':
		# 	homing_velocity = Decimal(1)
		# 	self.device.SetHomingVelocity(homing_velocity)
		# 	print(f'Homiung velocity for {83845817} set to {homing_velocity}')

		# current_homing_velocity = self.device.GetHomingVelocity()
		# print(f'homing_velocity = {current_homing_velocity}')

		# Home the linear actuator with a 60 second timeout
		print(f'Homing linear actuator with serial no.: {self.serial_number}\nPlease wait...')
		self.device.Home(self.home_timeout)
		print("Linear actuator homing complete!\n")


if __name__ == '__main__':

	chip_y_lin_act_serial_num = 83852988 
	chip_x_lin_act_serial_num = 83845817
	vga_x_lin_act_serial_num = 83860410

	lin_act_y = LinearActuator(motor_serial_number=chip_y_lin_act_serial_num)
	lin_act_x = LinearActuator(motor_serial_number=chip_x_lin_act_serial_num)
	

	distance = +1000

	# Initial position
	# x_pos = lin_act_x.read_current_position()
	# print(f'current X position = {x_pos}')

	y_pos = lin_act_y.read_current_position()
	print(f'current Y position = {y_pos}')

	# Move X
	print(f'\nmoving X by {distance}')
	lin_act_x.move_by(distance_um=distance)
	time.sleep(2)

	# new_x_pos = lin_act_x.read_current_position()
	# x_moved = new_x_pos - x_pos
	# x_error = x_moved - distance
	# print(f'new X position = {new_x_pos}\nX position error = {x_error}')

	# y_pos = lin_act_y.read_current_position()
	# print(f'new Y position = {y_pos}')

	# Move Y
	print(f'\nmoving Y by {distance}')
	lin_act_y.move_by(distance_um=distance)
	time.sleep(2)

	# x_pos = lin_act_x.read_current_position()
	# print(f'current X position = {x_pos}')

	new_y_pos = lin_act_y.read_current_position()
	y_moved = new_y_pos - y_pos
	y_error = y_moved - distance
	print(f'new Y position = {new_y_pos}\nY position error = {y_error}')

	# REVERSE
	distance = -distance

	# Move X
	# print(f'\nmoving X by {distance}')
	# lin_act_x.move_by(distance_um=distance)
	# time.sleep(2)

	# new_x_pos = lin_act_x.read_current_position()
	# x_moved = new_x_pos - x_pos
	# x_error = x_moved - distance
	# print(f'new X position = {new_x_pos}\nX position error = {x_error}')

	# y_pos = lin_act_y.read_current_position()
	# print(f'new Y position = {y_pos}')

	# # Move Y
	# print(f'\nmoving Y by {distance}')
	# lin_act_y.move_by(distance_um=distance)
	# time.sleep(2)

	# x_pos = lin_act_x.read_current_position()
	# print(f'current X position = {x_pos}')

	# new_y_pos = lin_act_y.read_current_position()
	# y_moved = new_y_pos - y_pos
	# y_error = y_moved - distance
	# print(f'new Y position = {new_y_pos}\nY position error = {y_error}')

	