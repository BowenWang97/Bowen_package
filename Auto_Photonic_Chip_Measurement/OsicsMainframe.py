from __future__ import print_function
import __init__
import serial, re, time

class OsicsMainframe(object):
	"""
	Super class which handles serial communication, device identification, and logging with an Osics mainframe laser.

	serial_port_name = None             Name of serial port, eg 'COM1' or '/dev/tty1'
	channel = 1                         Mainframe module selection 1-3

	"""

	baudrate = 9600  # Serial port baud rate (signalling frequency, Hz)
	response_timeout = 0.200  # Timeout for RESPONSE_OK or error to set commands
	max_channel = 3

	def __init__(self, serial_port_name, channel=None):

		# constructor
		self.serial_port_name = serial_port_name
		self.channel = 1

		try:
			self.serial_port = serial.Serial(self.serial_port_name, self.baudrate, timeout=self.response_timeout)
		except:
			raise AttributeError("No device found on port {}".format(self.serial_port_name))

		self.init_time = time.time()
		print('\nConnected to Tunics laser on serial port {0}\n'.format(self.serial_port_name))

		if str(channel)[0:2] == 'CH' and (int(channel[-1]) >= 0 and int(channel[-1]) <= self.max_channel) and len(channel) == 3:
			self.channel = channel + ':'
			print('Controlling laser on ' + channel)
		else:
			raise NameError("Wrong name for channel: must be 'CH<# of channel>', or channel specified isn't installed")

	def close(self):
		"""
		Destructor.
		"""
		# Close serial port
		if self.serial_port is not None and self.serial_port.is_open:
			self.serial_port.close()

	def transmit(self, command_string):
		"""
		Low-level transmit data method.
		"""
		# Ensure serial port is open
		if not self.serial_port.is_open:
			self.serial_port.open()

		# Write to port
		self.serial_port.write(command_string.encode('ascii'))

	def switch_channel(self, channel):
		if channel[0:2] == 'CH' and (int(channel[-1]) >= 0 and int(channel[-1]) <= self.max_channel) and len(channel) == 3:
			self.channel = channel + ':'
			print('Controlling laser on ' + channel)

		else:
			raise NameError("Wrong name for channel: must be 'CH<# of channel>', or channel specified isn't installed")

		return None

	def set_echo(self, echo=1):
		echo_dict = ['OFF', 'ON']
		self.transmit(self.channel + 'ECHO' + echo_dict[echo] + '\r')

	def mw_or_dbm(self, val='MW'):
		self.transmit(self.channel + val + '\r')

	def get_laser_wavelength(self):
		self.transmit(self.channel + 'L?\r')
		response = []
		while 'L=' not in response:
			response = str(self.serial_port.readline().decode())
		return [float(lam) for lam in re.findall('\d+\.\d+', response)][0]

	def set_laser_wavelength(self, wavelength):
		response = []
		self.transmit(self.channel + 'L={0: 4.4f}\r'.format(float(wavelength)))
		while 'OK' not in response:
			response = str(self.serial_port.readline().decode())
		return None

	def switch_on(self):
		cmd = self.channel + 'ENABLE\r'
		self.transmit(cmd)
		time.sleep(1)
		return None

	def switch_off(self):
		cmd = self.channel + 'DISABLE\r'
		self.transmit(cmd)
		return None

	def get_laser_power(self):
		# wont work unless laser is on
		self.transmit(self.channel + 'P?\r')
		response = []
		while 'P=' not in response:
			response = str(self.serial_port.readline().decode())
		return [float(p) for p in re.findall('\d+\.\d+', response)][0]#[float(p) for p in re.findall('\d+\.\d+', response)][0]

	def set_laser_power(self, power):
		self.transmit(self.channel + 'P={0: 4.4f}\r'.format(power))
		return None

	def get_laser_state(self):
		self.transmit(self.channel + 'ENABLE?\r')
		response = []
		while 'DISABLED' or 'ENABLED' not in response:
			response = str(self.serial_port.readline())
			# print(response)
			# print("waiting...")
			if 'ENABLED' in response:
				#print("ENABLED")
				return "ENABLED"
			elif 'DISABLED' in response:
				#print("DISABLED")
				return "DISABLED"
			else:
				pass
