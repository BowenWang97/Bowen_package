# Author: James Blatcher 2024
# This is a script to generate a GUI for moving linear acutators using the .net python wrapper 'ThorlabsLinearActuatpr_V2.py


# -*- coding: utf-8 -*-
from __future__ import print_function
import __init__
import ctypes
import sys
from PyQt5.QtWidgets import (QApplication, QLabel, QWidget, QPushButton, QVBoxLayout, QSlider, \
							QComboBox, QDoubleSpinBox, QMainWindow, QHBoxLayout, QGroupBox, QGridLayout, QTextEdit)
from PyQt5.QtCore import QTimer, QRect
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor

from Powermeter import *
from OsicsMainframe import *
from ThorlabsLinearActuator_V2 import *
from ThorlabsPolariser import *


rm = visa.ResourceManager()

laser_com_port = 'COM5'
laser_channel = "CH1"

powermeter1_port = 'USB0::0x1313::0x8078::P0010441::INSTR'
powermeter2_port = 'USB0::0x1313::0x8078::P0005846::INSTR'

powermeter1_model = 'PM100D'
powermeter1_serial = 'P0010441'
powermeter1_unit = 'dBm'
powermeter1_averages = 100

powermeter2_model = 'PM100D'
powermeter2_serial = 'P0005846'
powermeter2_unit = 'dBm'
powermeter2_averages = 100

linear_motor_1_serial = 83845817
linear_motor_2_serial = 83852988

polarisation_controller_serial_number = 38229054

qm_sleep_time = 1

# Hack to make GUI icon appear correctly in Windows
myappid = '_'
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

class Window(QMainWindow):
	'''
	This is a PyQT5 GUI interface for connecting to Qontrol M2 motor drivers
	'''
	# Timeout in ms
	TIMEOUT = 100
	TIMEOUT_LASER = 10000

	def __init__(self, adopt_power_meter_lin_actuator=False, pm1=None, pm2=None, lin_act_x=None, lin_act_y=None, disable_lin_act_hom=False, laser=None, pol_paddles=None):
		super(Window, self).__init__()

		# Instantiate tech
		if adopt_power_meter_lin_actuator == True:
			self.pm1 = pm1
			self.pm2 = pm2
			self.laser = laser
			self.lin_motor1 = lin_act_x
			self.lin_motor2 = lin_act_y
			self.pol_paddles = pol_paddles
		else:
			try:
				self.pm1 = Powermeter(model=powermeter1_model, port=powermeter1_port, unit=powermeter1_unit, averages=powermeter1_averages)
			except:
				print(f'Warning: No powermeter found on {powermeter1_port}')
				self.pm1 = None

			try:
				self.pm2 = Powermeter(model=powermeter2_model, port=powermeter2_port, unit=powermeter2_unit, averages=powermeter2_averages)
			except:
				print(f'Warning: No powermeter found on {powermeter1_port}')
				self.pm2 = None

			try:
				self.pol_paddles = PolarisationController(serial_number=polarisation_controller_serial_number, move_sleep_time=0)
			except:
				print(f'Warning: No polarisation paddles found with serial no. {polarisation_controller_serial_number}')
				self.pol_paddles = None
			
			self.laser = OsicsMainframe(serial_port_name=laser_com_port, channel=laser_channel)
			self.lin_motor1 = LinearActuator(motor_serial_number=linear_motor_1_serial, disable_home=disable_lin_act_hom)
			self.lin_motor2 = LinearActuator(motor_serial_number=linear_motor_2_serial, disable_home=disable_lin_act_hom)
		
		# Instantiate max power value
		self.mp1 = -80
		self.mp2 = -80
		self.initUI()

	def initUI(self):
		'''Set GUI interface parameters'''
		self.setGeometry(50, 50, 1000, 350)  # (50, 50, 400, 250)
		self.setWindowTitle("MotorGUI")
		self.setWindowIcon(QIcon('./Icons/QC.png'))
		# self.setWindowIcon(QIcon('Icons/QC.png'))
		#self.setWindowFlag(Qt.FramelessWindowHint)
		self.layout()
		
		# Initialise the app
		windowLayout = QVBoxLayout()
		windowLayout.addWidget(self.group_box) 
		self.setCentralWidget(QWidget(self))
		self.centralWidget().setLayout(windowLayout)

	def connect_state(self):
		'''Apply the connection to powermeter if button is clicked'''
		if self.connect.isChecked():
			print('Connect pressed!')
		else:
			if self.pm1 is None or self.pm2 is None:
				# print(self.ports)
				try:
					self.port1 = self.ports1.currentText()
					print('Connecting PM1 to:', self.port1)

					self.port2 = self.ports2.currentText()
					print('Connecting PM2 to:', self.port2)

					self.pm1 = Powermeter(model=powermeter1_model, port=powermeter1_port, unit=powermeter1_unit, averages=powermeter1_averages)#'USB0::0x1313::0x8078::P0005846::INSTR', unit='dBm', averages=100)
					self.pm2 = Powermeter(model=powermeter2_model, port=powermeter2_port, unit=powermeter2_unit, averages=powermeter2_averages)#Powermeter2(model='PM100D', port='USB0::0x1313::0x8070::P0000901::INSTR', unit='dBm', averages=100)

					self._is_connected = True
					self.update_power()
				except:
					print('Connection failed')
	
	# def disconnect_state(self):
	# 	'''Close the connection to powermeter if button is clicked'''
	# 	if self.disconnect.isChecked():
	# 		print('Disconnect pressed!')
	# 	else:
	# 		print('Disconnecting from:', self.port)
	# 		self._is_connected = False
	#
	# 		if self.pm1 is not None:
	# 			self.pm1.__del__()
	# 			self.pm1 = None
	# 			self.port = None
	# 			self.power_reading1.setText('-')
	# 			print('OK')
	#
	# 		elif self.pm2 is not None:
	# 			self.pm2.__del__()
	# 			self.pm2 = None
	# 			self.port = None
	# 			self.power_reading1.setText('-')

	### Laser methods ###

	def setting_laser_power(self, laser_power):
		if self.laser is not None:
			self.laser.set_laser_power(float(laser_power))  # ('{:5.2f}'.format(laser_power))
		else:
			pass

	def update_laser_wavelength(self, laser_wavelength):

		if self.laser is not None:
			self.laser.set_laser_wavelength('{:7.2f}'.format(laser_wavelength))
		else:
			pass

	def update_laser_power(self):
		laser_state = self.laser.get_laser_state()
		if laser_state == "ENABLED":
			laser_power_read_dB = self.laser.get_laser_power()
			self.laser_power_read.setText('{:5.2f}'.format(laser_power_read_dB))
			laser_power_read_mW = 10**(laser_power_read_dB / 10)
			self.laser_power_read_mW.setText('{:5.3f}'.format(laser_power_read_mW))
		else:
			self.laser_power_read.setText("     --     ")
			self.laser_power_read_mW.setText("     --     ")
		####
		# laser_power_read_dB = self.laser.get_laser_power()
		# self.laser_power_read.setText('{:5.2f} dBm'.format(laser_power_read_dB))
		# laser_power_read_mW = 10**(laser_power_read_dB/10)
		# self.laser_power_read_mW.setText('{:6.4f} mW'.format(laser_power_read_mW))

	### PM methods ###

	def update_power(self):
		'''Refresh the reading of the powermeter'''

		if self.pm1 is not None:
			# self.mp1 = -80
			power_dB1 = self.pm1.measure()
			if power_dB1 > self.mp1:
				self.mp1 = power_dB1
			self.max_power1.setText('{:7.3f} dBm'.format(self.mp1))
			self.power_reading1.setText('{:7.3f} dBm'.format(power_dB1))
		else:
			pass

		if self.pm2 is not None:
			# self.mp2 = -80
			power_dB2 = self.pm2.measure()
			if power_dB2 > self.mp2:
				self.mp2 = power_dB2
			self.max_power2.setText('{:7.3f} dBm'.format(self.mp2))
			self.power_reading2.setText('{:7.3f} dBm'.format(power_dB2))
		else:
			pass

	def reset_mp1(self):
		self.mp1 = -80

	def reset_mp2(self):
		self.mp2 = -80

	### Polarisation paddle methods

	def set_pol_paddle_angle(self, paddle_num, angle):
		"""
		paddle_num = int [1 - 3]
		angle = int or float [0 - 170]
		"""
		if self.pol_paddles != None:
			# Move the pol paddle to the new angle
			self.pol_paddles.move_to(paddle_number=paddle_num, position=angle)
			# Update the GUI with the new angle value
			# self.update_pol_paddle_angles()
		else:
			print('Error: No polarisation paddles connected')

	def update_pol_paddle_angles(self):
		if self.pol_paddles != None:
			angles = self.pol_paddles.read_current_position()
			self.pol_paddle1_angle.setText('{:5.2f} °'.format(angles[0]))
			self.pol_paddle2_angle.setText('{:5.2f} °'.format(angles[1]))
			self.pol_paddle3_angle.setText('{:5.2f} °'.format(angles[2]))

	### Linear actuator methods ###

	def move_x_left(self, step_size):
		# self.qm._move(XYZ_um=(step_size, 0, 0))		############### TO FIX!!! ###############################################
		self.lin_motor1.move_by(distance_um=step_size)
		time.sleep(qm_sleep_time)

	def move_x_right(self, step_size):
		# self.qm._move(XYZ_um=(-step_size, 0, 0))		############### TO FIX!!! ###############################################
		self.lin_motor1.move_by(distance_um=-step_size)
		time.sleep(qm_sleep_time)

	def move_y_up(self, step_size):
		self.lin_motor2.move_by(distance_um=step_size)
		time.sleep(qm_sleep_time)

	def move_y_down(self, step_size):
		self.lin_motor2.move_by(distance_um=-step_size)
		time.sleep(qm_sleep_time)

	# def move_y_up(self, step_size):
	# 	self.qm._move(XYZ_um=(0, -step_size, 0))		############### TO FIX!!! ###############################################
	# 	time.sleep(qm_sleep_time)

	# def move_y_down(self, step_size):
	# 	self.qm._move(XYZ_um=(0, step_size, 0))		############### TO FIX!!! ###############################################
	# 	time.sleep(qm_sleep_time)

	def move_z_raise(self, step_size):
		# self.qm._move(XYZ_um=(0, 0, step_size))		############### TO FIX!!! ###############################################
		time.sleep(qm_sleep_time)

	def move_z_lower(self, step_size):
		# self.qm._move(XYZ_um=(0, 0, -step_size))		############### TO FIX!!! ###############################################
		time.sleep(qm_sleep_time)

	def locally_optimise(self, local_optimisation_range):
		self.qm._local_optimisation(preferred_pm=self.pm1, scan_range_um=local_optimisation_range)		############### TO FIX!!!

	### Window layout ###

	def layout(self):
		'''Create all of the different widgets for the layout'''
		# 		self.hbox = QHBoxLayout()
		self.group_box = QGroupBox()

		grid = QGridLayout()
		grid.setColumnStretch(1, 4)
		grid.setColumnStretch(2, 4)

		# Laser labels and buttons
		laser_title = QLabel('Laser')
		laser_title.setStyleSheet(" font-size: 25px; font-family: Helvetica; text-align:center ; color: #B346F6;")

		wavelength_title = QLabel('Set laser wavelength')
		wavelength_title.setStyleSheet(" font-size: 20px; font-family: Helvetica; text-align:center ; color: #B346F6;")

		laser_wavelength_value = QDoubleSpinBox(self)
		laser_wavelength_value.setGeometry(100, 100, 100, 100)
		laser_wavelength_value.setRange(1500, 1600)
		laser_wavelength_value.setSuffix(" nm")
		laser_wavelength_value.setStyleSheet(" font-size: 20px; font-family: Helvetica; text-align:center ; color: cyan;")

		self.laser_wavelength_value_set = QPushButton("Set Laser Wavelength")
		self.laser_wavelength_value_set.setStyleSheet(" font-size: 15px; font-family: Helvetica; text-align:center ; color: cyan;")
		self.laser_wavelength_value_set.setDefault(False)
		self.laser_wavelength_value_set.clicked.connect(lambda: self.update_laser_wavelength(laser_wavelength_value.value()))

		wavelength_min_max_title = QLabel('(min = 1500nm, max = 1600nm)')
		wavelength_min_max_title.setStyleSheet(" font-size: 10px; font-family: Helvetica;text-align:center ; color: #B346F6;")

		power_title = QLabel('Set laser power')
		power_title.setStyleSheet(" font-size: 20px; font-family: Helvetica; text-align:center ; color: #B346F6;")

		laser_power_value = QDoubleSpinBox(self)
		laser_power_value.setGeometry(100, 100, 100, 100)
		laser_power_value.setRange(-6, 7)
		laser_power_value.setSuffix(" dBm")
		laser_power_value.setStyleSheet(" font-size: 20px; font-family: Helvetica; text-align:center ; color: cyan;")

		self.laser_power_value_set = QPushButton("Set Laser Power")
		self.laser_power_value_set.setStyleSheet(" font-size: 15px; font-family: Helvetica; text-align:center ; color: cyan;")
		self.laser_power_value_set.setDefault(False)
		self.laser_power_value_set.clicked.connect(lambda: self.setting_laser_power(laser_power_value.value()))

		power_min_max_title = QLabel('(min = -6dBm, max = +7dBm)')
		power_min_max_title.setStyleSheet(" font-size: 10px; font-family: Helvetica; text-align:center ; color: #B346F6;")

		laser_power_title = QLabel('Laser power (dBm)')
		laser_power_title.setStyleSheet(" font-size: 20px; font-family: Helvetica; text-align:center ; color: #B346F6;")

		self.laser_enable = QPushButton("Laser ENABLE")
		self.laser_enable.setStyleSheet(" font-size: 15px; font-family: Helvetica; text-align:center ; color: cyan;")
		self.laser_enable.setDefault(False)
		self.laser_enable.clicked.connect(lambda: self.laser.switch_on())

		self.laser_disable = QPushButton("Laser DISABLE")
		self.laser_disable.setStyleSheet(" font-size: 15px; font-family: Helvetica; text-align:center ; color: cyan;")
		self.laser_disable.setDefault(False)
		self.laser_disable.clicked.connect(lambda: self.laser.switch_off())

		self.laser_power_read = QLabel('     --     ')
		self.laser_power_read.setStyleSheet(" font-size: 30px; font-family: Helvetica; text-align:center ; color: cyan;")

		laser_power_mw_title = QLabel('Laser power (mw)')
		laser_power_mw_title.setStyleSheet(" font-size: 20px; font-family: Helvetica; text-align:center ; color: #B346F6;")

		self.laser_power_read_mW = QLabel('     --     ')
		self.laser_power_read_mW.setStyleSheet(" font-size: 30px; font-family: Helvetica; text-align:center ; color: cyan;")


		# Power meter labels and buttons
		powermeter_title = QLabel('Powermeters')
		powermeter_title.setStyleSheet(" font-size: 25px; font-family: Helvetica; text-align:center; color: #B346F6;")

		powermeter1_title = QLabel('Powermeter 1')
		powermeter1_title.setStyleSheet(" font-size: 20px; font-family: Helvetica; text-align:center; color: #B346F6;")

		powermeter2_title = QLabel('Powermeter 2')
		powermeter2_title.setStyleSheet(" font-size: 20px; font-family: Helvetica; text-align:center; color: #B346F6;")

		# self.connect = QPushButton("Connect to powermeter(s)")
		# self.connect.setDefault(False)
		# self.connect.clicked.connect(self.connect_state)

		# self.disconnect = QPushButton("Disconnect from powermeter(s)")
		# self.disconnect.setDefault(False)
		# self.disconnect.clicked.connect(self.disconnect_state)

		self.power_reading1 = QLabel('     --     ')
		self.power_reading1.setStyleSheet(" font-size: 30px; font-family: Helvetica; text-align:center; color: cyan;")
		#self.power_reading1.setGeometry(100, 100, 200, 100)

		max_power1_title = QLabel('Powermeter 1 Max')
		max_power1_title.setStyleSheet(" font-size: 20px; font-family: Helvetica; text-align:center; color: #B346F6;")

		self.max_power1 = QLabel('     --     ')
		self.max_power1.setStyleSheet(" font-size: 30px; font-family: Helvetica; text-align:center; color: cyan;")

		self.max_power1_reset = QPushButton("RESET Powermeter1 Max")
		self.max_power1_reset.setStyleSheet(" font-size: 15px; font-family: Helvetica; text-align:center; color: cyan;")
		self.max_power1_reset.setDefault(False)
		self.max_power1_reset.clicked.connect(self.reset_mp1)

		self.power_reading2 = QLabel('     --     ')
		self.power_reading2.setStyleSheet(" font-size: 30px; font-family: Helvetica; text-align:center; color: cyan;")

		max_power2_title = QLabel('Powermeter 2 Max')
		max_power2_title.setStyleSheet(" font-size: 20px; font-family: Helvetica; text-align:center; color: #B346F6;")

		self.max_power2 = QLabel('     --     ')
		self.max_power2.setStyleSheet(" font-size: 30px; font-family: Helvetica; text-align:center; color: cyan;")

		self.max_power2_reset = QPushButton("RESET Powermeter2 Max")
		self.max_power2_reset.setStyleSheet(" font-size: 15px; font-family: Helvetica; text-align:center; color: cyan;")
		self.max_power2_reset.setDefault(False)
		self.max_power2_reset.clicked.connect(self.reset_mp2)


		# Motor labels and buttons
		motors_title = QLabel('Motors')
		motors_title.setStyleSheet(" font-size: 25px; font-family: Helvetica; text-align:center; color: #B346F6;")

		motor_step_size_title = QLabel('Motor step size')
		motor_step_size_title.setStyleSheet(" font-size: 20px; font-family: Helvetica; text-align:center; color: #B346F6;")

		motor_step_size_value = QDoubleSpinBox(self)
		motor_step_size_value.setGeometry(100, 100, 100, 100)
		motor_step_size_value.setRange(0.01, 1000)#(0.01, 100)
		motor_step_size_value.setSuffix(" um")
		motor_step_size_value.setStyleSheet(" font-size: 20px; font-family: Helvetica; text-align:center; color: cyan;")

		move_x_title = QLabel('Move X')
		move_x_title.setStyleSheet(" font-size: 20px; font-family: Helvetica; text-align:center; color: #B346F6;")

		self.move_x_left_button = QPushButton()
		self.move_x_left_button.setIcon(QIcon('./Icons/left_triangle_cyan.png'))
		self.move_x_left_button.setDefault(False)
		self.move_x_left_button.clicked.connect(lambda: self.move_x_left(motor_step_size_value.value()))

		self.move_x_right_button = QPushButton()
		self.move_x_right_button.setIcon(QIcon('./Icons/right_triangle_cyan.png'))
		self.move_x_right_button.setDefault(False)
		self.move_x_right_button.clicked.connect(lambda: self.move_x_right(motor_step_size_value.value()))

		move_y_title = QLabel('Move Y')
		move_y_title.setStyleSheet(" font-size: 20px; font-family: Helvetica; text-align:center; color: #B346F6;")

		self.move_y_up_button = QPushButton()
		self.move_y_up_button.setIcon(QIcon('./Icons/up_triangle_cyan.png'))
		self.move_y_up_button.setDefault(False)
		self.move_y_up_button.clicked.connect(lambda: self.move_y_up(motor_step_size_value.value()))

		self.move_y_down_button = QPushButton()
		self.move_y_down_button.setIcon(QIcon('./Icons/down_triangle_cyan.png'))
		self.move_y_down_button.setDefault(False)
		self.move_y_down_button.clicked.connect(lambda: self.move_y_down(motor_step_size_value.value()))

		move_z_title = QLabel('Move Z')
		move_z_title.setStyleSheet(" font-size: 20px; font-family: Helvetica; text-align:center; color: #B346F6;")

		self.move_z_raise_button = QPushButton()
		self.move_z_raise_button.setIcon(QIcon('./Icons/raise_cyan.png'))
		self.move_z_raise_button.setDefault(False)
		self.move_z_raise_button.clicked.connect(lambda: self.move_z_raise(motor_step_size_value.value()))		############### TO FIX!!!

		self.move_z_lower_button = QPushButton()
		self.move_z_lower_button.setIcon(QIcon('./Icons/lower_cyan.png'))
		self.move_z_lower_button.setDefault(False)
		self.move_z_lower_button.clicked.connect(lambda: self.move_z_lower(motor_step_size_value.value()))		############### TO FIX!!!


		# Local optimisation labels and buttons
		local_optimisation_title = QLabel('Local Optimisation')
		local_optimisation_title.setStyleSheet(" font-size: 25px; font-family: Helvetica; text-align:center; color: #B346F6;")

		local_optimisation_range_title = QLabel('Local optimisation range')
		local_optimisation_range_title.setStyleSheet(" font-size: 20px; font-family: Helvetica; text-align:center; color: #B346F6;")


		local_optimisation_range_value = QDoubleSpinBox(self)
		local_optimisation_range_value.setGeometry(100, 100, 100, 100)
		local_optimisation_range_value.setRange(1, 20)
		local_optimisation_range_value.setSuffix(" um")
		local_optimisation_range_value.setStyleSheet(" font-size: 20px; font-family: Helvetica; text-align:center; color: cyan;")

		self.locally_optimise_button = QPushButton("Optimise coupling!")
		self.locally_optimise_button.setStyleSheet(" font-size: 20px; font-family: Helvetica; text-align:center; color: cyan;")
		self.locally_optimise_button.setDefault(False)
		# self.locally_optimise_button.clicked.connect(lambda: self.locally_optimise(local_optimisation_range_value.value()))		############### TO FIX!!!


		# Polarisation paddles labels and buttons
		
		pol_paddles_title = QLabel('Polarisation paddles')
		pol_paddles_title.setStyleSheet(" font-size: 25px; font-family: Helvetica; text-align:center; color: #B346F6;")

		pol_paddle1_title = QLabel('Polarisation paddle 1')
		pol_paddle1_title.setStyleSheet(" font-size: 20px; font-family: Helvetica; text-align:center; color: #B346F6;")

		self.pol_paddle1_angle = QLabel('     --     ')
		self.pol_paddle1_angle.setStyleSheet(" font-size: 20px; font-family: Helvetica; text-align:center; color: cyan;")

		self.pol_paddle1_slider = QSlider(Qt.Horizontal)
		self.pol_paddle1_slider.setMinimum(0)
		self.pol_paddle1_slider.setMaximum(170)
		self.pol_paddle1_slider.setSingleStep(1)
		self.pol_paddle1_slider.valueChanged.connect(lambda: self.set_pol_paddle_angle(paddle_num=1, angle=self.pol_paddle1_slider.value()))

		pol_paddle2_title = QLabel('Polarisation paddle 2')
		pol_paddle2_title.setStyleSheet(" font-size: 20px; font-family: Helvetica; text-align:center; color: #B346F6;")

		self.pol_paddle2_angle = QLabel('     --     ')
		self.pol_paddle2_angle.setStyleSheet(" font-size: 20px; font-family: Helvetica; text-align:center; color: cyan;")

		self.pol_paddle2_slider = QSlider(Qt.Horizontal)
		self.pol_paddle2_slider.setMinimum(0)
		self.pol_paddle2_slider.setMaximum(170)
		self.pol_paddle2_slider.setSingleStep(1)
		self.pol_paddle2_slider.valueChanged.connect(lambda: self.set_pol_paddle_angle(paddle_num=2, angle=self.pol_paddle2_slider.value()))

		pol_paddle3_title = QLabel('Polarisation paddle 3')
		pol_paddle3_title.setStyleSheet(" font-size: 20px; font-family: Helvetica; text-align:center; color: #B346F6;")

		self.pol_paddle3_angle = QLabel('     --     ')
		self.pol_paddle3_angle.setStyleSheet(" font-size: 20px; font-family: Helvetica; text-align:center; color: cyan;")

		self.pol_paddle3_slider = QSlider(Qt.Horizontal)
		self.pol_paddle3_slider.setMinimum(0)
		self.pol_paddle3_slider.setMaximum(170)
		self.pol_paddle3_slider.setSingleStep(1)
		self.pol_paddle3_slider.valueChanged.connect(lambda: self.set_pol_paddle_angle(paddle_num=3, angle=self.pol_paddle3_slider.value()))


		# Timer settings for continuous measurement
		self._timer = QTimer(self)
		self._timer.start(self.TIMEOUT)
		self._timer.timeout.connect(self.update_power)
		
		self._timer_pol_paddles = QTimer(self)
		self._timer_pol_paddles.start(self.TIMEOUT)
		self._timer_pol_paddles.timeout.connect(self.update_pol_paddle_angles)

		self._timer_laser = QTimer(self)
		self._timer_laser.start(self.TIMEOUT_LASER)
		### FIX!!!
		self._timer_laser.timeout.connect(self.update_laser_power)
		###

		# Add laser widgets
		grid.addWidget(laser_title, 0, 0)
		grid.addWidget(wavelength_title, 1, 0)
		grid.addWidget(laser_wavelength_value, 1, 1)
		grid.addWidget(wavelength_min_max_title, 2, 0)
		grid.addWidget(self.laser_wavelength_value_set, 2, 1)
		grid.addWidget(power_title, 3, 0)
		grid.addWidget(laser_power_value, 3, 1)
		grid.addWidget(power_min_max_title, 4, 0)
		grid.addWidget(self.laser_power_value_set, 4, 1)
		grid.addWidget(self.laser_enable, 5, 0)
		grid.addWidget(self.laser_disable, 5, 1)
		grid.addWidget(laser_power_title, 6, 0)
		grid.addWidget(self.laser_power_read, 6, 1)
		grid.addWidget(laser_power_mw_title, 7, 0)
		grid.addWidget(self.laser_power_read_mW, 7, 1)

		# Add powermeter widgets
		grid.addWidget(powermeter_title, 0, 2)
		grid.addWidget(powermeter1_title, 1, 2)
		grid.addWidget(self.power_reading1, 1, 3)
		grid.addWidget(max_power1_title, 2, 2)
		grid.addWidget(self.max_power1, 2, 3)
		grid.addWidget(powermeter2_title, 3, 2)
		grid.addWidget(self.power_reading2, 3, 3)
		grid.addWidget(max_power2_title, 4, 2)
		grid.addWidget(self.max_power2, 4, 3)
		grid.addWidget(self.max_power1_reset, 5, 2)
		grid.addWidget(self.max_power2_reset, 5, 3)

		# Add motor widgets
		grid.addWidget(motors_title, 0, 4)
		grid.addWidget(motor_step_size_title, 1, 4)
		grid.addWidget(motor_step_size_value, 1, 5)
		grid.addWidget(move_x_title, 2, 4)
		grid.addWidget(self.move_x_left_button, 2, 5)
		grid.addWidget(self.move_x_right_button, 2, 6)
		grid.addWidget(move_y_title, 3, 4)
		grid.addWidget(self.move_y_up_button, 3, 5)
		grid.addWidget(self.move_y_down_button, 3, 6)
		grid.addWidget(move_z_title, 4, 4)
		grid.addWidget(self.move_z_raise_button, 4, 5)
		grid.addWidget(self.move_z_lower_button, 4, 6)

		# Add local optimisation widgets
		grid.addWidget(local_optimisation_title, 5, 4)
		grid.addWidget(local_optimisation_range_title, 6, 4)
		grid.addWidget(local_optimisation_range_value, 6, 5)
		grid.addWidget(self.locally_optimise_button, 7, 4)

		# Add polarisation paddle widgets
		grid.addWidget(pol_paddles_title, 0, 7)
		grid.addWidget(pol_paddle1_title, 1, 7)
		grid.addWidget(self.pol_paddle1_angle, 1, 8)
		grid.addWidget(self.pol_paddle1_slider, 2, 7, 1, 2)
		grid.addWidget(pol_paddle2_title, 3, 7)
		grid.addWidget(self.pol_paddle2_angle, 3, 8)
		grid.addWidget(self.pol_paddle2_slider, 4, 7, 1, 2)
		grid.addWidget(pol_paddle3_title, 5, 7)
		grid.addWidget(self.pol_paddle3_angle, 5, 8)
		grid.addWidget(self.pol_paddle3_slider, 6, 7, 1, 2)

		# grid.addWidget(self.connect, 3, 5)
		# grid.addWidget(self.disconnect, 3, 5)

		self.group_box.setLayout(grid)


def run(adopt_power_meter_lin_actuator=False,
		pm1=None,
		pm2=None,
		laser=None,
		lin_act_x=None,
		lin_act_y=None,
		disable_lin_act_homing=False,
		pol_paddles=None):
	
	app = QApplication([])
	# Force the style to be the same on all OSs:
	app.setStyle("Fusion")
	#app.setApplicationDisplayName("MotorGUI")
	#app.setWindowIcon(QIcon('Icons/QC.png'))

	# Now use a palette to switch to dark colors:
	palette = QPalette()
	palette.setColor(QPalette.Window, QColor(53, 53, 53))
	palette.setColor(QPalette.WindowText, Qt.white)
	palette.setColor(QPalette.Base, QColor(25, 25, 25))
	palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
	palette.setColor(QPalette.ToolTipBase, Qt.black)
	palette.setColor(QPalette.ToolTipText, Qt.white)
	palette.setColor(QPalette.Text, Qt.white)
	palette.setColor(QPalette.Button, QColor(53, 53, 53))
	palette.setColor(QPalette.ButtonText, Qt.white)
	palette.setColor(QPalette.BrightText, Qt.red)
	palette.setColor(QPalette.Link, QColor(42, 130, 218))
	palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
	palette.setColor(QPalette.HighlightedText, Qt.black)
	app.setPalette(palette)

	win = Window(adopt_power_meter_lin_actuator=adopt_power_meter_lin_actuator,
			  pm1 = pm1,
			  pm2 = pm2,
			  laser = laser,
			  lin_act_x = lin_act_x,
			  lin_act_y = lin_act_y,
			  disable_lin_act_hom=disable_lin_act_homing,
			  pol_paddles = pol_paddles)
	win.show()

	# Exit the python script if __name__ == '__main__' , otherwise (if this GUI is being callef from elsewhere), close just the GUI window	
	if __name__ == '__main__':
		sys.exit(app.exec_())
	else:
		app.exec_()

if __name__ == '__main__':
	run(disable_lin_act_homing=False)#=True)

