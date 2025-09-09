from __future__ import print_function
import __init__
import time
import qontrol
import numpy as np
import threading

class QontrolMotors(object):
	"""
	Class which handles tech setup, user interfacing, and data processing.

	 pd_models = None            Photodiode device IDs
	 pd_serials = None           Photodiode serial port objects (eg 'COM1', '/dev/tty1')
	 units = None 				 Photodiode units NOTE this does not set the units on the PMs, it is for reading only
	 m2_device_id = None         M2 device IDs
	 m2_serial_port = None       M2 serial port objects
	 m2_serial_port_name = None  Name of M2 port, (eg 'COM1', '/dev/tty1')
	 laser_port_name = None      Name of Oscis mainframe port, (eg 'COM1', '/dev/tty1')
	 laser_channel = None		 Name of Oscis mainframe channel e.g., 'CH4'

	 log = fifo(maxlen = 256)    Log FIFO of communications
	 log_handler = None          Function which catches log dictionaries
	 log_to_stdout = True        Copy new log entries to stdout
	 error_desc_dict             Error code descriptions
	 device_dict = {}            Dictionary of devices to sweep, see port_dict.csv for example layout
	 pms = [None]                List of photodiode devices if already connected
	 laser = None				 Mainframe laser device if already connected

	"""

	def __init__(self, *args, **kwargs):
		"""
		Initialiser.
		"""

		# Defaults
		self.m2_device_id = None  # Qontrol M2 device ID (i.e. [device type]-[device number])
		self.m2_serial_port = None  # Qontrol M2 serial port object
		self.m2_serial_port_name = None  # Qontrol M2 name of serial port, eg if WINDOWS then 'COM1' or MAC then '/dev/tty1'

		# Set a time benchmark
		self.init_time = time.time()

		# Define XYZ motor channel defaults
		self.dims = {'X': 0, 'Y': 1, 'Z': 2}

		# Microstep distance for ustep = 0 to 8
		self.dudx = [3.175 / 2 ** n for n in [0, 1, 2, 3, 4, 5, 6, 7, 8]]

		self.backlash_um = 20
		self.ustep_backlash_um = 10

		# Get arguments from init
		# Populate parameters, if provided
		for para in ['pd_models',
					 'pd_serials',
					 'units',
					 'm2_device_id',
					 'm2_serial_port_name',
					 'laser_port_name',
					 'laser_channel',
					 'log_to_stdout',
					 'device_dict',
					 'pms',
					 'dims',
					 'laser']:
			try:
				self.__setattr__(para, kwargs[para])
			except KeyError:
				continue

		# Setup Qontrol tech
		self.qs = qontrol.MXMotor(serial_port_name=self.m2_serial_port_name, response_timout=10)
		self.qs.response_timeout = 10

		# # # Set motor speed to slow - more accurate translations
		self.qs.set_value(0, 'VMAX', 1)
		self.qs.set_value(1, 'VMAX', 1)
		self.qs.set_value(2, 'VMAX', 1)
		self.qs.set_value(0, 'USTEP', 3)
		self.qs.set_value(1, 'USTEP', 3)
		self.qs.set_value(2, 'USTEP', 3)
		time.sleep(1)

		print("'{:}' initialised with firmware {:} and {:} channels".format(self.qs.device_id, self.qs.firmware,
																			self.qs.n_chs))

		# Create a lock for current position
		self.lock = threading.Lock()

		return

	def _local_optimisation(self, preferred_pm, scan_range_um=10):
		"""
		Run a local optimisation over X and Y
		For each ustep, create grid and find maximum.
		"""

		# Calculate number of inital grid poitns -> scans grid_N x grid_N, spaced 3.175um apart, centered on the inital position
		grid_N = int(np.ceil(scan_range_um / self.dudx[0]))

		# Hack - if grid_N is odd, then add one to make it even
		if (grid_N % 2 == 1):
			grid_N += 1

		# Check min scan range
		if scan_range_um < 3.175:
			print('Error: scan range must be > 3.175um')
			# self.log_append(type='err', id=115)

		# Step through resolutions, down to a resolution of 3.175um / 2^[usteps[-1]]
		usteps = [0, 1]
		grid_N = [grid_N, 1]

		# X0 = self.qs.x[self.dims['X']]
		# self.qs.x[self.dims['X']] = round(X0)
		# self.qs.wait_until_stopped(t_poll = 0.5)
		# Y0 = self.qs.x[self.dims['Y']]
		# self.qs.x[self.dims['Y']] = round(Y0)
		# self.qs.wait_until_stopped(t_poll = 0.5)

		for u, N in zip(usteps, grid_N):

			# create grid
			stops = [(-N, -N)]
			measurements = []

			for i in range(0, 2 * N + 1):
				i = i % 2
				stops.extend([(((-1) ** i), 0)] * (2 * N))
				stops.append((0, 1))
			stops.pop()

			for stop in stops:
				# Move X
				X0 = self.qs.x[self.dims['X']]
				self.qs.x[self.dims['X']] = X0 + float(stop[0] * (self.dudx[u] / self.dudx[0]))
				self.qs.wait_until_stopped(t_poll=0.5)
				time.sleep(np.abs(float(stop[0] * (self.dudx[u] / self.dudx[0])) * 0.5))
				X1 = self.qs.x[self.dims['X']]

				# # FOR DEBUGGING WITHOUT M2 PLUGGED IN
				# X0 = self.current_position[0]
				# X1 = self.current_position[0] + stop[0]*(self.dudx[u]/self.dudx[0])
				# if X1 - X0 != float(stop[0]*(self.dudx[u]/self.dudx[0])):
				# 	self.log_append(type='info', id='-1', params='Local optimisation X scan step failed: target move = {:}, X0 = {:}, X1 = {:}'.format(float(stop[0]*(self.dudx[u]/self.dudx[0])), X0, X1))

				Y0 = self.qs.x[self.dims['Y']]
				self.qs.x[self.dims['Y']] = Y0 + float(stop[1] * (self.dudx[u] / self.dudx[0]))
				self.qs.wait_until_stopped(t_poll=0.5)
				time.sleep(np.abs(float(stop[1] * (self.dudx[u] / self.dudx[0])) * 0.5))
				Y1 = self.qs.x[self.dims['Y']]

				# # FOR DEBUGGING WITHOUT M2 PLUGGED IN
				# Y0 = self.current_position[1]
				# Y1 = self.current_position[1] + stop[1]*(self.dudx[u]/self.dudx[0])
				# if Y1 - Y0 != float(stop[1]*(self.dudx[u]/self.dudx[0])):
				# 	self.log_append(type='info', id='-1', params='Local optimisation Y scan step failed: target move = {:}, Y0 = {:}, Y1 = {:}'.format(float(stop[0]*(self.dudx[u]/self.dudx[0])), X0, X1))

				# Update current position with actual distance moved
				# self.current_position = np.array([self.current_position[0] + self.dudx[0] * (X0 - X1),
				# 								  self.current_position[1] + self.dudx[0] * (Y1 - Y0)])

				# # Measure
				_ = self.pms[preferred_pm - 1].measure()
				measurements.append(_)
			# measurements.append(randrange(100))

			# Move back to optimal spot and switch to next ustep
			p_opt = len(measurements) - np.argmax(measurements) - 1

			backwards_stops = [tuple(-x for x in tup) for tup in stops]
			backwards_stops.reverse()

			if u == 0 and p_opt == 0:
				print('Optimal value found at edge of optimisation range - consider increasing scan range')

			move_to = tuple(sum(values) for values in zip(*backwards_stops[0:p_opt]))

			if p_opt == 0:
				pass
			else:

				# Move X
				X0 = self.qs.x[self.dims['X']]
				self.qs.x[self.dims['X']] = X0 + float(move_to[0] * (self.dudx[u] / self.dudx[0]))
				self.qs.wait_until_stopped(t_poll=0.5)
				time.sleep(np.abs(float(move_to[0] * (self.dudx[u] / self.dudx[0])) * 0.5))
				X1 = self.qs.x[self.dims['X']]

				# # FOR DEBUGGING WITHOUT M2 PLUGGED IN
				# X0 = self.current_position[0]
				# X1 = self.current_position[0] + move_to[0]*(self.dudx[u]/self.dudx[0])
				# if X1 - X0 != float(move_to[0]*(self.dudx[u]/self.dudx[0])):
				# 	self.log_append(type='info', id='-1', params='Local optimisation X position step failed: target move = {:}, X0 = {:}, X1 = {:}'.format(float(move_to[0]*(self.dudx[u]/self.dudx[0])), X0, X1))

				Y0 = self.qs.x[self.dims['Y']]
				self.qs.x[self.dims['Y']] = Y0 + float(move_to[1] * (self.dudx[u] / self.dudx[0]))
				self.qs.wait_until_stopped(t_poll=0.5)
				time.sleep(np.abs(float(move_to[1] * (self.dudx[u] / self.dudx[0])) * 0.5))
				Y1 = self.qs.x[self.dims['Y']]

				# # FOR DEBUGGING WITHOUT M2 PLUGGED IN
				# Y0 = self.current_position[1]
				# Y1 = self.current_position[1] + move_to[1]*(self.dudx[u]/self.dudx[0])
				# if Y1 - Y0 != float(move_to[1]*(self.dudx[u]/self.dudx[0])):
				# 	self.log_append(type='info', id='-1', params='Local optimisation Y position step failed: target move = {:}, Y0 = {:}, Y1 = {:}'.format(float(move_to[1]*(self.dudx[u]/self.dudx[0])), X0, X1))

				# Update current position with actual distance moved
				# self.current_position = np.array([self.current_position[0] + self.dudx[0] * (X0 - X1),
				# 								  self.current_position[1] + self.dudx[0] * (Y1 - Y0)])

			# Save move incase of failure
			if u == 0:
				if p_opt == 0:
					save_move = ((int(np.ceil(scan_range_um / self.dudx[0]))) * self.dudx[0],
								 (int(np.ceil(scan_range_um / self.dudx[0]))) * self.dudx[0])
				else:
					save_move = ((int(np.ceil(scan_range_um / self.dudx[0])) + move_to[0]) * self.dudx[0],
								 (int(np.ceil(scan_range_um / self.dudx[0])) + move_to[1]) * self.dudx[0])

		return save_move

	def _move(self, XYZ_um = (0,0,0)):
		"""
		For translating distance (x,y,z)um
		"""

		# Axis direction corrections
		XYZ_um = (-XYZ_um[0], XYZ_um[1], XYZ_um[2])

		# Move distance in X
		X0 = self.qs.x[self.dims['X']]
		# If moving in a positive x direction according to coordinate csv (= negative x motor movement), account for backlash
		if (X0 + float(XYZ_um[0]/self.dudx[0])) < X0:	### may need to swap < with >  -- the signs in this code are confusing JEB
			# Move to new x position - backlash
			self.qs.x[self.dims['X']] = X0 + float(XYZ_um[0]/self.dudx[0]) - float(self.backlash_um/self.dudx[0])
			self.qs.wait_until_stopped(t_poll = 0.5)
			time.sleep(np.abs(float((XYZ_um[0] - self.backlash_um)*(self.dudx[3]/self.dudx[0]))*0.2))
			# Then move to new x position
			self.qs.x[self.dims['X']] = X0 + float(XYZ_um[0]/self.dudx[0])
			self.qs.wait_until_stopped(t_poll = 0.5)
			time.sleep(np.abs(float(XYZ_um[0]*(self.dudx[3]/self.dudx[0]))*0.2))
		# If moving in a negative x direction according to coordinate csv (= positive x motor movement), perform a standard motor translation
		else:
			self.qs.x[self.dims['X']] = X0 + float(XYZ_um[0]/self.dudx[0])
			self.qs.wait_until_stopped(t_poll = 0.5)
			time.sleep(np.abs(float(XYZ_um[0]*(self.dudx[3]/self.dudx[0]))*0.2))

		X1 = self.qs.x[self.dims['X']]

		if X1 - X0 != float(XYZ_um[0]/self.dudx[0]):
			print('Device stepping X position failed: target move = {:} steps = {:}um, actual move = {:}um, X0 = {:}, X1 = {:}'.format(float(XYZ_um[0]/self.dudx[0]), XYZ_um[0], (X0 - X1), X0, X1))

		# Move distance in Y
		Y0 = self.qs.x[self.dims['Y']]
		# If moving in a positive y direction according to coordinate csv (= negative y motor movement), account for backlash
		if (Y0 + float(XYZ_um[1]/self.dudx[0])) < Y0:	### may need to swap < with >  -- the signs in this code are confusing JEB
			# Move to new y position - backlash
			self.qs.x[self.dims['Y']] = Y0 + float(XYZ_um[1]/self.dudx[0]) - float(self.backlash_um/self.dudx[0])
			self.qs.wait_until_stopped(t_poll = 0.5)
			time.sleep(np.abs(float((XYZ_um[1] - self.backlash_um)/self.dudx[0]))*0.2)
			# Then move to new y position
			self.qs.x[self.dims['Y']] = Y0 + float(XYZ_um[1]/self.dudx[0])
			self.qs.wait_until_stopped(t_poll = 0.5)
			time.sleep(np.abs(float(XYZ_um[1]/self.dudx[0]))*0.2)
		# If moving in a negative y direction according to coordinate csv (= positive y motor movement), perform a standard motor translation
		else:
			self.qs.x[self.dims['Y']] = Y0 + float(XYZ_um[1]/self.dudx[0])
			self.qs.wait_until_stopped(t_poll = 0.5)
			time.sleep(np.abs(float(XYZ_um[1]/self.dudx[0]))*0.2)
		
		Y1 = self.qs.x[self.dims['Y']]

		if Y1 - Y0 != float(XYZ_um[1]/self.dudx[0]):
			print('Device stepping Y position failed: target move = {:}, Y0 = {:}, Y1 = {:}'.format(float(XYZ_um[1]/self.dudx[0]), Y0, Y1))       
		
				
		return

