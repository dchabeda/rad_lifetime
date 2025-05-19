# Python script to plot anisotropy order parameters for xyz files or lammps trajectories
import numpy as np
import re
import sys
from glob import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator

if 'br' in sys.argv:
	ortho_ref = 0.03245
else:
	ortho_ref = 0.026142
	

plot_files = glob('lat_anis*allsizes_out*.dat')

for file in plot_files:
	
	match = re.search('lat_anis_(\d+)K', file)
	if match:
		temp_flag = match.group(1)
	
	match = re.search('lat_anis_\d+K_allsizes_out_([a-z]+)', file)
	if match:
		order_param = match.group(1)

	match = re.search('lat_anis_\d+K_allsizes_out_[a-z]+_(\d).', file)
	if match:
		avg_flag = '_'+ match.group(1)
	else:
		avg_flag = ''
	
	#print('Filename: ', file)
	#print('Tempflag: ', temp_flag)
	#print('Avg flag: ', avg_flag)
	order_params = np.loadtxt(file)
	
	font = {'weight' : 'light',
       'size' : 16}
	mpl.rcParams['font.family'] = 'Serif'
	mpl.rcParams["axes.prop_cycle"]
	mpl.rcParams['axes.linewidth'] = 0.5

	fig, ax = plt.subplots()
	fig.set_size_inches(12,9)    

	ax.yaxis.set_minor_locator(AutoMinorLocator(2))
	ax.xaxis.set_minor_locator(AutoMinorLocator(2))
	ax.locator_params(axis='both',nbins=5)
	ax.tick_params(axis='both',which='major', direction='in',length=10,width=1,color='black',pad=5,labelsize=20,labelcolor='black',
				labelrotation=0)
	ax.tick_params(axis='both',which='minor', right=True, top=True, direction='in',length=5,width=0.5,color='black',pad=15,labelsize=10,labelcolor='black',
				labelrotation=0)
	ax.tick_params(top=True, bottom=True, left=True, right=True)
	
	if temp_flag == '0' or avg_flag == '_0':
		ax.plot(order_params[:,0], order_params[:,1], color='k', marker= 'o', linewidth=3.5, label='relaxed')
	elif avg_flag == '_1':
		ax.errorbar(order_params[:,0], order_params[:,1], order_params[:,2], color='k', marker= 'o', capsize=5,linewidth=2.5, label='relaxed' )
	
	ax.axhline(0, 0, 1,color = 'green', linestyle='-.', linewidth=1.5, label='cubic ref.')
	ax.axhline(ortho_ref, 0, 1,color = 'purple', linestyle='--', linewidth=1.5, label='ortho ref.')

	ax.set_xlabel('Dot size (nm)', fontsize = 24)
	ax.set_ylabel('Lattice Anisotropy (Å)', fontsize=24)
	ax.legend(frameon=False, fontsize=18, loc='best')
	if avg_flag == '_0':
		ax.set_title(r'$\langle{\chi}\rangle = \chi(\langle r_i \rangle_i)$', fontsize=24)
	elif avg_flag == '_1':
		ax.set_title(r'$\langle{\chi}\rangle = \langle \chi(r_i) \rangle_i$', fontsize=24)
	plt.savefig('perov_lat_anis_{}K{}_{}_angle.png'.format(temp_flag, avg_flag, order_param),dpi=200)

	