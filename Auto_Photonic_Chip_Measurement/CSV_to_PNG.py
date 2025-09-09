import __init__
import os
import pandas as pd
from matplotlib import pyplot as plt

class csv_to_png(object):

	def __init__(self, measurement_filepath = None, plot_folder_path = None, new_plot_folder_name=None):
		
		# Clean up the file name and file paths 
		self.measurement_filepath = self.file_name_cleanup(type='filepath', name=measurement_filepath)
		self.plot_folder_path = self.file_name_cleanup(type='filepath', name=plot_folder_path)

		# Instantiate global parameters
		self.new_plot_folder_name = new_plot_folder_name
		self.coordinate_map_file = None

		# Extract the name of the folder that contains the measurements and add it to the end of the newly created plot folder name
		self.meas_run_folder = self.measurement_filepath.split('/')[-1]
		self.new_plot_folder = self.plot_folder_path + '/' + self.new_plot_folder_name + '_' + self.meas_run_folder

		# Check folder exsists, if not then make it
		try: 
			os.chdir(self.new_plot_folder)
		except:
			os.mkdir(self.new_plot_folder)

		# Extract the location of the coordinate map
		self.coordinate_map_file = self.find_coordinate_map_in_folder()

		# Read the coordinate map
		self.coordinate_map = pd.read_csv(self.coordinate_map_file)#, index_col='id')
		# Determine the number of devices listed in the coordinate map
		self.num_devices = self.coordinate_map.shape[0]

	def find_coordinate_map_in_folder(self):
		# Find the coordinate map - assuming its in the results folder
		files = os.listdir(self.measurement_filepath)
		for file in files:
			if file.lower().endswith('.csv') and not file.lower().startswith('id'):
				return os.path.join(self.measurement_filepath, file)
		
		if self.coordinate_map_file == None:
			print('ERROR - cannot find coordinate map .csv file')
			exit()

	def file_name_cleanup(self, type, name):
		if type == 'filename':
			name = name.replace('|', '')
			name = name.replace(' ', '_')
			name = name.replace('__', '_')
			# name = name.replace('.', '')
			return name
		elif type == 'filepath':
			name = name.replace('\\', '/')
			return name
		else:
			print('Error - invalid filetype: {}'.format(type))

	def file_name_shorten(self, name):
		name = name.replace('Apodised', 'Ap')
		name = name.replace('Periods', 'P')
		name = name.replace('periods', 'P')
		name = name.replace('Period', 'P')
		# name = name.replace('Cavity', 'Cav')
		# name = name.replace('Cavities', 'Cav')
		name = name.replace('Cavity', 'C')
		name = name.replace('Cavities', 'C')
		name = name.replace('Device', 'Dev')
		name = name.replace('Etch_Width', 'EW')
		name = name.replace('Series', 'Ser')
		name = name.replace('No.', 'N')
		return name

	def generate_png_plots_all_files_in_folder(self):
		# For each device in the coordinate map
		for i in range(self.num_devices):

			# Can use df.loc[index_name, column_name], where index_name = i bc the index column starts at 0, as does i in this for loop

			# Determine the device's id number
			id_num = self.coordinate_map.loc[i, 'id']
			
			# Determine the filepath of the measurement results for the current device
			measurement_filename = '\id_{}.csv'.format(id_num)
			measurement_file = self.measurement_filepath + measurement_filename

			# Extract the key parameters of the current device from the coordinate map for plot labelling
			device_name = self.coordinate_map.loc[i, 'device_name']
			device_type = self.coordinate_map.loc[i, 'device_type']
			device_name_plot = 'Device ' + str(device_name) + ' - ' + str(device_type)

			# Generate a list of device details from the coordinate map
			device_details_list = self.coordinate_map.loc[i, 'device_additional_info_1' : 'device_additional_info_6']

			# Create a string for the device details, by taking the first item in the device details list
			device_details = str(device_details_list[0])
			device_details_plot = str(device_details_list[0])

			# If there are additional device details, append them to the string in a plot name friendly and filename friendly format
			for j in range(5):
				if str(device_details_list[j + 1]) != 'nan':
					device_details = device_details + '_' + str(device_details_list[j + 1])
					device_details_plot = device_details_plot + ' | ' + str(device_details_list[j + 1])
			
			# Generate a full device name for the filename and apply the filename cleanup
			full_device_name = 'Device_' + str(device_name) + '_-_' + str(device_type) + '__' + str(device_details)
			full_device_name = self.file_name_cleanup(type='filename', name=full_device_name)

			# If the measurement file for the current device exists, open it and read the data
			if os.path.isfile(measurement_file):
				measurement_data = pd.read_csv(measurement_file)
			else:
				print('WARNING - Measurement data .csv file not found for:\n{}'.format(measurement_file))
				continue

			# Extract the X and Y axis values
			x = measurement_data['Wavelength (nm)'].values
			y = measurement_data['Channel 1 power (dBm)'].values
			
			# Generate the filename for the new plot
			plot_file = self.new_plot_folder + '/' + 'id_'+ str(id_num) + '_-_' + full_device_name + '.png'
			
			# Shorten the filename to avoid the filepath being > 255 characters
			plot_file = self.file_name_shorten(name=plot_file)

			# Generate the plot
			fig, ax = plt.subplots()
			ax.plot(x, y)

			ax.set(xlabel='Wavelength (nm)', ylabel='Received Power (dBm)')
			ax.grid()
			fig.suptitle(device_name_plot, fontsize=15)
			plt.title(device_details_plot, fontsize=10)

			# If the .png file in the plot folder does not exist yet, save the plot as a .png
			if not (os.path.isfile(plot_file)):
				fig.savefig(plot_file)
			#plt.show()
			plt.close()

			# print('Successfully generated plot for device with ID No. {}'.format(id_num))

	def generate_png_plots_single_file(self, id_num):
		
		# id_num = self.coordinate_map.iloc[i, 0]

		row_number = id_num - 1
		
		# Determine the filepath of the measurement results for the current device
		measurement_filename = '\id_{}.csv'.format(id_num)
		measurement_file = self.measurement_filepath + measurement_filename

		# Extract the key parameters of the current device from the coordinate map for plot labelling
		# device_name = self.coordinate_map.loc[id_num, ['device_name']].values[0]
		# device_type = self.coordinate_map.loc[id_num, ['device_type']].values[0]
		device_name = self.coordinate_map.loc[row_number, 'device_name']
		device_type = self.coordinate_map.loc[row_number, 'device_type']
		device_name_plot = 'Device ' + str(device_name) + ' - ' + str(device_type)

		# Generate a list of device details from the coordinate map
		device_details_list = self.coordinate_map.loc[row_number, 'device_additional_info_1' : 'device_additional_info_6']

		# Create a string for the device details, by taking the first item in the device details list
		device_details = str(device_details_list[0])
		device_details_plot = str(device_details_list[0])

		# If there are additional device details, append them to the string in a plot name friendly and filename friendly format
		for j in range(5):
			if str(device_details_list[j + 1]) != 'nan':
				device_details = device_details + '_' + str(device_details_list[j + 1])
				device_details_plot = device_details_plot + ' | ' + str(device_details_list[j + 1])
		
		# Generate a full device name for the filename and apply the filename cleanup
		full_device_name = 'Device_' + str(device_name) + '_-_' + str(device_type) + '__' + str(device_details)
		full_device_name = self.file_name_cleanup(type='filename', name=full_device_name)

		# If the measurement file for the current device exists, open it and read the data
		if os.path.isfile(measurement_file):
			measurement_data = pd.read_csv(measurement_file)
		else:
			print('WARNING - Measurement data .csv file not found for:\n{}'.format(measurement_file))

		# Extract the X and Y axis values
		x = measurement_data['Wavelength (nm)'].values
		# y = measurement_data['Channel 1 power (dBm)'].values
		y = measurement_data.iloc[:, -1].values	# Now selects the last column dynamically
		
		# Generate the filename for the new plot
		plot_file = self.new_plot_folder + '/' + 'id_'+ str(id_num) + '_-_' + full_device_name + '.png'
		
		# Shorten the filename to avoid the filepath being > 255 characters
		plot_file = self.file_name_shorten(name=plot_file)

		# Generate the plot
		fig, ax = plt.subplots()
		ax.plot(x, y)

		ax.set(xlabel='Wavelength (nm)', ylabel='Received Power (dBm)')
		ax.grid()
		fig.suptitle(device_name_plot, fontsize=15)
		plt.title(device_details_plot, fontsize=10)

		fig.savefig(plot_file)
		#plt.show()
		plt.close()

		# print('Successfully generated plot for device with ID No. {}'.format(id_num))

if __name__ == '__main__':
###############################################################

	chip_measurement_filepath = r'C:/Users/jb16078/OneDrive - University of Bristol/PhD/Lab/Measurements/AutoRig_Measurements/ANT_SiN_Spring_2024'
	measurement_run_folder = r'_2024-08-13 15-03-35'

	plot_folder_path = r'C:/Users/jb16078/OneDrive - University of Bristol/PhD/Lab/Measurements/AutoRig_Measurements/ANT_SiN_Spring_2024'

	new_plot_folder_name = 'Side_1_Plots'

	measurement_filepath = chip_measurement_filepath + '/' + measurement_run_folder

	plotter = csv_to_png(measurement_filepath=measurement_filepath,
					  plot_folder_path=plot_folder_path,
					  new_plot_folder_name=new_plot_folder_name)

	plotter.generate_png_plots_all_files_in_folder()
	
###############################################################



# import __init__
# import pandas as pd
# from matplotlib import pyplot as plt

# def csv_to_png(coordinate_map_file:str, measurement_filepath:str, plot_filepath:str):

#     coordinate_map = pd.read_csv(coordinate_map_file, index_col='id')
#     num_devices = coordinate_map.shape[0]

#     for i in range(num_devices):
#         id_num = coordinate_map.iloc[i, 0]
		
#         measurement_filename = '\id_{}.csv'.format(id_num)
#         measurement_file = measurement_filepath + measurement_filename

#         # device_name = coordinate_map.loc[[i + 1], ['device_type']].values[0][0]
#         device_name = coordinate_map.loc[id_num, ['device_type']].values[0]#[0]

#         # device_details_list = coordinate_map.loc[[i + 1], 'device_additional_info_1' : 'device_additional_info_6'].values[0]
#         device_details_list = coordinate_map.loc[id_num, 'device_additional_info_1' : 'device_additional_info_6']#.values[0]
#         device_details = str(device_details_list[0])
#         device_details_plot = str(device_details_list[0])
#         for j in range(5):
#             if str(device_details_list[j + 1]) != 'nan':
#                 device_details = device_details + '_' + str(device_details_list[j + 1])
#                 device_details_plot = device_details_plot + ' | ' + str(device_details_list[j + 1])
#         full_device_name = str(device_name) + '__' + str(device_details)

#         measurement_data = pd.read_csv(measurement_file)

#         x = measurement_data['Wavelength (nm)'].values
#         y = measurement_data['Channel 1 power (dBm)'].values

#         # id_num = i+1
#         # plot_file = plot_filepath + '/' + full_device_name + '.png'
#         plot_file = plot_filepath + '/' + 'id_'+ str(id_num) + '.png'

#         fig, ax = plt.subplots()
#         ax.plot(x, y)

#         ax.set(xlabel='Wavelength (nm)', ylabel='Received Power (dBm)')
#         ax.grid()
#         fig.suptitle(device_name, fontsize=15)
#         plt.title(device_details_plot, fontsize=10)

#         fig.savefig(plot_file)
# 		#plt.show()
# 		plt.close()

# if __name__ == '__main__':
# ###############################################################
# 	coordinate_map_filepath = r'C:\Users\jb16078\OneDrive - University of Bristol\PhD\Lab\Measurements\AutoRig_Measurements\ANT_SiN_Spring_2024'
# 	coordinate_map_filename = r'/ANT_SiN_Spring_2024_Chip_1_V2_4__Chip_Map_Side_1_LR_COORDSTESTTTTT.csv'

# 	chip_measurement_filepath = r'C:/Users/jb16078/OneDrive - University of Bristol/PhD/Lab/Measurements/AutoRig_Measurements/ANT_SiN_Spring_2024'
# 	measurement_run_folder = r'/_2024-06-07 18-27-45'
	

# 	#plot_filepath = 'C:/Users/jb16078/OneDrive - University of Bristol/PhD/Lab/Measurements/AutoRig_Measurements/Cornerstone_SiN_Winter_2022/Plots'
# 	plot_filepath = r'C:/Users/jb16078/OneDrive - University of Bristol/PhD/Lab/Measurements/AutoRig_Measurements/ANT_SiN_Spring_2024'
# 	plot_filepath = plot_filepath.replace('\\', '/')
# 	new_plot_folder_name = 'Side_1_Plots'
# 	new_plot_folder = + '/' + new_plot_folder_name + '__' + measurement_run_folder[1:]

# 	measurement_filepath = chip_measurement_filepath + measurement_run_folder
# 	measurement_filepath = measurement_filepath.replace('\\', '/')

# 	coordinate_map_file = coordinate_map_filepath + coordinate_map_filename
# 	coordinate_map_file = coordinate_map_file.replace('\\', '/')

# 	csv_to_png(coordinate_map_file=coordinate_map_file, measurement_filepath=measurement_filepath, plot_filepath=plot_filepath)
# ###############################################################
