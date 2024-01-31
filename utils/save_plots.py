"""
TODO: DO proper documenation
"""
# native modules
import os
import sys
import json

# 3rd party modules
import matplotlib.pyplot as plt
from matplotlib.pylab import plt

# First add project parent directory to sys.path
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.insert(0,PROJ_DIR)

class SavePlots:
	def __init__(self,
	      		dir = os.path.join(PROJ_DIR,"data","plots"),
				plot_name:str="train_loss.png", 
				plot_figure_size:tuple = (10,7),
				save_json_also:bool = True,
				overwrite:bool=True,
				number_of_plots:int=1,
				plot_title:str="Some Generic Title",
				x_axis_label:str="X axis label",
				y_axis_label:str="Y axis label"):
		"""
		Save plots during training.

		Args.:	<str>dir: path to where plots should be saved
				<str>plot_name: the name the plot should be saved as
				<tuple>plot_figure_size: the dimensin of the figure
				<bool>save_json_also: bool to save the corresponding JSON encoded details
				<bool>over_write: overwrite file or create new ones
		
		Return:	None
		"""
		self.dir = dir
		self.plot_name = plot_name
		self.plot_figure_size = plot_figure_size
		self.save_json_also = save_json_also
		self.overwrite = overwrite
		self.number_of_plots = number_of_plots
		self.plot_title = plot_title
		self.x_axis_label = x_axis_label
		self.y_axis_label = y_axis_label

		# data elements
		self.y_axis_data = {}
		self.legends = []

		# initailize lists
		for i in range(self.number_of_plots):
			self.y_axis_data[i] = []

		if not os.path.exists(self.dir):
			raise ValueError("Invalid destnation path. Path provided is `{}`".format(self.dir))
		
		if not isinstance(number_of_plots,int):
			raise ValueError("Please provide how many plots are going to be created. Provided number is {}.".format(number_of_plots))

	def __call__(self,y_axis:list,legend:list):
				
		if not isinstance(y_axis,list) or not isinstance(legend,list):
			raise ValueError("y_axis and legend must be of type list. Provided types are {} and {}".format(type(y_axis),type(legend)))
				
		if len(y_axis) != len(legend):
			raise ValueError("y_axis and legend must have the same length. Provided lengths are {} and {}".format(len(y_axis),len(legend)))


		plt.figure(figsize=self.plot_figure_size)
		plt.title(self.plot_title)
		plt.xlabel(self.x_axis_label)
		plt.ylabel(self.y_axis_label)

		for i in range(len(y_axis)):
			self.y_axis_data[i].append(y_axis[i])
			plt.plot(self.y_axis_data[i], label=legend[i])

		# plt.xticks(arange(0, 21, 2)) # Set the tick locations
		plt.legend(loc='best')
		plt.savefig(os.path.join(self.dir,self.plot_name))
		plt.close() # https://heitorpb.github.io/bla/2020/03/18/close-matplotlib-figures/

		if self.save_json_also:
			self._save_json_also(self.y_axis_data,legend)

	
	def _save_json_also(self,y_axis_data, legend):
		""" 
		Save raw plot values as json files
		"""
		# use legend list to create dictionary keys
		data = {}

		for index, key in enumerate(legend,start=0):
			data[key] = y_axis_data[index]
		
		file_name = self.plot_name.split(".")[0]
		file_name += ".json"

		if self.overwrite is False:
			file_name = file_name if self.over_write==True else "epoch_{}_{}".format(str(self.epochs[-1]+1),file_name)

		with open(os.path.join(self.dir,file_name), 'w') as f:
			json.dump(data, f) 