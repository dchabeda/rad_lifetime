# Script to calculate lattice anisotropy of configurations from LAMMPS
import numpy as np
import re
import os
import sys
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# Function to write .xyz and .par output
def write_output(filename, coords, opt='xyz'):
	if opt == 'xyz':
		units = 1.0
		filename += '.xyz'
		fmt = '\n\n'
	if opt == 'par':
		units = 1/0.529177 
		filename += '.par'
		fmt = '\n'
	if opt == 'H':
		units = 1.0
		filename += '_H.xyz'
		fmt = '\n\n'
	
	file = open(filename, 'w')
	
	n_atoms = str(len(coords))
	file.write(n_atoms+fmt)
	
	id_to_symb = {1: "Cs", 2: "Pb", 3: "I", 4: "Cs"}
	for coord in coords:
		file.write('{name} {x} {y} {z}\n'.format(name=id_to_symb[coord[0]],x=coord[1]*units,y=coord[2]*units,z=coord[3]*units))
		
	file.close()

# Function to get coordinates and return (n_atom,4) np.array with first column the atom label
def get_coords(xyzflag, tempflag, avgflag):
	# If avgflag = 0, then the average coords over the trajectory will be returned
	print('\tGetting coords...')
	if xyzflag == 'xyz':
		coords = np.loadtxt('min.xyz', skiprows=2, dtype=np.float64)
		# Shift the COM to 0
		coords[:,1:] -= np.mean(coords, axis=0)[1:]
		
		return coords
	elif tempflag:
		with open('equil_1ps_{}K.lammpstrj'.format(tempflag)) as f:
			file = f.readlines()
		file = ''.join(file)
		
		search_string = 'ITEM: ATOMS id type x y z q vx vy vz\n((?:[\d\.e-]+\s)+)'
		match = re.findall(search_string, file)
		if match:
			coords = match
		#String processing
		
		coords = [line.split('\n')[:-1] for line in coords]
		coords = [[line.split() for line in obj] for obj in coords]
		
		equil_time = 0#1000 # every entry is separated by 1 ps, so this is a 1 ns equilibration time
		print(f"\tEquilibration time: {equil_time / 1e3} ns")
		coords = np.asarray(coords[equil_time:], dtype=np.float64)
		coords = coords[:,:,1:6]
		print("\tLength of trajectory: ", coords.shape[0]*1e-3, " ns")
		
		if avgflag == 0:
			print('averaging coords')
			coords = np.mean(coords, axis=0)
		#print("The averaged coords:\n", coords)
		
		# Shift the COM to 0
		coords[:,:,1:4] -= np.mean(coords, axis = 1, keepdims=True)[:,:,1:4]
		np.savetxt("avg_coords.xyz", avg_coords:= np.mean(coords, axis=0)[:,:4], fmt=['%d', '%.6f', '%.6f', '%.6f'], header=f"{avg_coords.shape[0]}\nAvg. coords from trajectory", comments='')
		
		#np.savetxt("2x2x2_equil_1fs.dat", coords.reshape(coords.shape[0],-1), header=f"original shape: {coords.shape[0]} {coords.shape[1]} {coords.shape[2]}")
		return coords
	else:
		with open('min.traj') as f:
			file = f.readlines()
		file = ''.join(file)
		
		search_string = 'ITEM: ATOMS type xu yu zu q fx fy fz\n((?:[\d\.e-]+\s)+)'
		match = re.search(search_string, file)
		if match:
			
			coords = match.group(1)
		coords = coords.split('\n')
		coords.pop(-1) #remove trailing newline character from Regex
		coords = np.array([line.split() for line in coords], dtype=np.float64)
		coords = coords[:,0:5]
		# Shift the COM to 0
		coords[:,2:] -= np.mean(coords, axis=0)[2:]
		
		return coords
		
# Function to compute all the angles between n-dimensional atom1 (n,3) and atom2 (n,3) arrays with the same vertex (1,3)
def compute_angle(vertex, atom1, atom2):
	# compute vectors from vertex to the atoms
	vector1 = atom1 - vertex
	vector2 = atom2 - vertex
	
	# Compute dot product between every pair of vectors
	dot_products = np.dot(vector1[:, np.newaxis, :], vector2.T[np.newaxis, :, :])
	dot_products = np.squeeze(dot_products)
	
	# Create a matrix of the magnitudes ||u||*||v||
	magnitude_product = np.outer(np.linalg.norm(vector1, axis=1), np.linalg.norm(vector2, axis=1))
	
	# Avoid division by zero
	if magnitude_product.any() == 0:
		return np.nan

	# **** COMPUTE THE ANGLES ***
	cosine_angle = dot_products / magnitude_product
	angle_in_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
	angle_in_degrees = np.degrees(angle_in_radians) # Convert angle to degrees

	#print("\nAngle in degrees\n",angle_in_degrees)
	
	'''if angle_in_degrees[2,2] > 120.0 or angle_in_degrees[3,3] > 120.0:
		angle_in_degrees = np.array([angle_in_degrees[0,0],angle_in_degrees[1,2],angle_in_degrees[2,1],angle_in_degrees[3,3]])
	else:
		angle_in_degrees = np.diagonal(angle_in_degrees)'''
	

	return angle_in_degrees


def get_angles(xyz, ncube, plothist=1):
	# We want to compute the halide X-X-X angles between octahedra. 
	# Select the halide atoms (ID 3)
	
	I_atoms = xyz[xyz[:, 0] == 3]
	
	# Calculate pairwise distances between all I atoms to construct a quasi-neighbor list
	distances = cdist(I_atoms, I_atoms)

	# Create an array to store indices of type 2 atoms within a distance of 4.9
	indices_within_distance = np.where(distances < 4.9, 1, 0)

	# Ensure that the diagonal (self-distances) is set to zero
	np.fill_diagonal(indices_within_distance, 0)
	
	# We want to pick out the "octahedral vertex" halide atoms with 8 neighboring iodides. We will find the angles
	# around these centers to compute the order parameter
	vertex_ind = np.where(np.sum(indices_within_distance, axis=1) == 8)[0]
	vertex_I_atoms = I_atoms[vertex_ind]
	
	# Get the 8 halide atoms around the "vertex" I atom
	I_atoms_within_distance = np.array([I_atoms[j_arr.astype(bool)] for j_arr in indices_within_distance[vertex_ind]])
	
	# We need to figure out along which direction to compute angles. If the vertex is between
	# two octahedra below and above, then we only want the angles between the lower and upper octahedral I atoms.
	# This means we should sort our array into segments of small and large z values. This same logic goes 
	# for the other two directions.
	
	# Calculate the maximum difference in x, y, and z values for each of the 8 atoms around each vertex
	# dimension of max_array will be nrows = number of centers, ncols = 4 (id_diff, max_diff in x, y, z)
	max_array = np.ptp(I_atoms_within_distance, axis=1)
	#print("max_array: \n", max_array)
	vertex_direction = np.argmax(max_array,axis=1) # If the group of neighboring atoms is long along the x axis, then the vertex is horizontal between two octahedra, etc.
	I_atoms_side1 = []
	I_atoms_side2 = []
	for i in range(vertex_I_atoms.shape[0]):
		if vertex_direction[i] == 1:
			# The group of neighboring I atoms is long along the x axis. 
			#print(f"\nCenter {i}: {vertex_I_atoms[i,1:]} is x polarized.")
			# Sort by x value and then cut in half to get the four I atoms to the left and the four to the right.
			ind = np.argsort(I_atoms_within_distance[i,:,1])
			I_atoms_within_distance[i,:,:] = I_atoms_within_distance[i,:,:][ind]
			side1 = I_atoms_within_distance[i,:4,:]; side2 = I_atoms_within_distance[i,4:,:]
			# Subsequently sort the array by z
			#ind = np.argsort(side1[:,3]); side1 = side1[ind]
			#ind = np.argsort(side2[:,3]); side2 = side2[ind]
			I_atoms_side1.append(side1); I_atoms_side2.append(side2)
		if vertex_direction[i] == 2:
			#print(f"\nCenter {i}: {vertex_I_atoms[i,1:]} is y polarized.")
			ind = np.argsort(I_atoms_within_distance[i,:,2])
			I_atoms_within_distance[i,:,:] = I_atoms_within_distance[i,:,:][ind]
			side1 = I_atoms_within_distance[i,:4,:]; side2 = I_atoms_within_distance[i,4:,:]

			# Subsequently sort the array by x
			#ind = np.argsort(side1[:,1]); side1 = side1[ind]
			#ind = np.argsort(side2[:,1]); side2 = side2[ind]
			I_atoms_side1.append(side1); I_atoms_side2.append(side2)
		if vertex_direction[i] == 3:
			#print(f"\nCenter {i}: {vertex_I_atoms[i,1:]} is z polarized.")
			ind = np.argsort(I_atoms_within_distance[i,:,3])
			I_atoms_within_distance[i,:,:] = I_atoms_within_distance[i,:,:][ind]
			side1 = I_atoms_within_distance[i,:4,:]; side2 = I_atoms_within_distance[i,4:,:]
			
			#ind = np.argsort(side1[:,2]); side1 = side1[ind]
			#ind = np.argsort(side2[:,2]); side2 = side2[ind]
			I_atoms_side1.append(side1); I_atoms_side2.append(side2)
	#print(I_atoms_within_distance)
	
	I_atoms_side1 = np.array(I_atoms_side1); I_atoms_side2 = np.array(I_atoms_side2)
	
	#print("side1:\n", I_atoms_side1[0]); print("\nside2:\n", I_atoms_side2[0])
	#compute_angles = compute_angle(vertex_I_atoms[0,1:],I_atoms_side1[0,:,1:], I_atoms_side2[0,:,1:])
	
	# Compute all X-X-X angles in the neighboring 8 atoms 
	angles = []
	for i in range(vertex_I_atoms.shape[0]):
		angles.append(compute_angle(vertex_I_atoms[i,1:],I_atoms_side1[i,:,1:], I_atoms_side2[i,:,1:]))
		
	#compute_angles = np.array([(res := compute_angle(vertex_I_atoms[i,1:],I_atoms_side1[i,:,1:], I_atoms_side2[i,:,1:])) \
		#for i in range(vertex_I_atoms.shape[0])])
	#print(compute_angles)
	angles = np.array(angles)
	# Make a histogram of the angles in the crystal
	angles = angles.flatten()

	# Create histogram
	min_ang = 50; max_ang = 180; binwid = 10.0
	hist, bins = np.histogram(angles, bins=np.arange(min_ang+binwid/2, max_ang+3*binwid/2, binwid))
	# Calculate bin centers
	bin_centers = (bins[:-1] + bins[1:]) / 2
	
	# Normalize frequencies to create probabilities
	probabilities = hist / np.sum(hist)
	
	outfile_name = f"{ncube}x{ncube}x{ncube}_cubic_hist_{min_ang}-{max_ang}deg_bin{binwid}"

	np.savetxt(f"{outfile_name}_angles.dat", angles)
	bins_and_probs = np.concatenate((bin_centers.reshape(-1,1),probabilities.reshape(-1,1)), axis=1)
	np.savetxt(f"{outfile_name}_probs.dat", bins_and_probs, fmt='%.4f')

	if plothist:
		fig, ax = plt.subplots(dpi=120)
		plt.title('Histogram of Angles')
		plt.xlabel('All X-X-X Angles between centers (degrees)')
		plt.ylabel('Probability')
		plt.bar(bin_centers, probabilities, width=np.diff(bins), edgecolor='black')
		plt.grid(True, alpha=0.5)
		plt.savefig(f"{outfile_name}.png", dpi=160)
	
	return bin_centers,probabilities


def calc_order_param(bins, sample_probs, ang1, ang2):
	# These are the probabilities of seeing an X-X-X angle +/- 5 degrees in a perfect
	# cubic or orthorhombic crystal.
	
	cubic_distr_ref = np.array([
		[60.0000, 0.1667],
		[70.0000, 0.0000],
		[80.0000, 0.0000],
		[90.0000, 0.2708],
		[100.0000, 0.0000],
		[110.0000, 0.0000],
		[120.0000, 0.3333],
		[130.0000, 0.0000],
		[140.0000, 0.0000],
		[150.0000, 0.0000],
		[160.0000, 0.0000],
		[170.0000, 0.0000],
		[180.0000, 0.2292]])
	ortho_distr_ref = np.array([
		[60.0000, 0.0000],
		[70.0000, 0.1250],
		[80.0000, 0.0000],
		[90.0000, 0.0417],
		[100.0000, 0.1458],
		[110.0000, 0.1875],
		[120.0000, 0.0833],
		[130.0000, 0.0833],
		[140.0000, 0.0417],
		[150.0000, 0.1250],
		[160.0000, 0.1667],
		[170.0000, 0.0000],
		[180.0000, 0.0000]])

	P_cubic_ang1 = cubic_distr_ref[np.where(cubic_distr_ref[:,0] == ang1)[0],1][0]
	P_cubic_ang2 = cubic_distr_ref[np.where(cubic_distr_ref[:,0] == ang2)[0],1][0]
	P_ortho_ang1 = ortho_distr_ref[np.where(cubic_distr_ref[:,0] == ang1)[0],1][0]
	P_ortho_ang2 = ortho_distr_ref[np.where(cubic_distr_ref[:,0] == ang2)[0],1][0]
	#print(f"P_cubic_ang1 = {P_cubic_ang1}")
	# ***** ***** ***** ***** ***** ***** *****
	# First calculate the Xi order parameter!
	# ***** ***** ***** ***** ***** ***** *****

	# We define reference values of the order parameter as the difference between the probabilities
	# for observing a certain angle in the perfect crystals.
	xi_ang1_ref = (P_ortho_ang1 - P_cubic_ang1)**2
	xi_ang2_ref = (P_ortho_ang2 - P_cubic_ang2)**2
	
	# The order parameter requires the probabilities from the real distribution of the simulated
	# nanocrystal
	prob_ang1 = sample_probs[np.where(bins == ang1)[0]][0]
	prob_ang2 = sample_probs[np.where(bins == ang2)[0]][0]
	
	# Calculate the order parameters, xi_90 and xi_160: (P(theta) - P_cub_ref(theta))/xi_ref(theta)
	xi_ang1 = (prob_ang1 - P_cubic_ang1)**2/xi_ang1_ref
	xi_ang2 = (prob_ang2 - P_cubic_ang2)**2/xi_ang2_ref
	xi = (xi_ang1 + xi_ang2)/2
	# ***** ***** ***** ***** ***** ***** *****
	# Second order parameter: KL divergence
	# ***** ***** ***** ***** ***** ***** *****

	# Calculate the KL divergence (relative entropy) https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
	# D_KL(P||Q) = sum_i P(i) log(P(i)/Q(i)) where P(i) is the reference and Q is the actual distribution
	nonzero_ind_c = np.where(cubic_distr_ref[:,1] != 0.0)[0]
	cubic_probs = cubic_distr_ref[nonzero_ind_c,1]
	nonzero_ind_o = np.where(ortho_distr_ref[:,1] != 0.0)[0]
	ortho_probs = ortho_distr_ref[nonzero_ind_o,1]
	# Compute the logarithms while ignoring the zero values
	# Use epsilon = 1e-5 so that zero values of the sample probability don't break the log function
	eps = 1e-5
	cubic_rel_log = np.log((sample_probs[nonzero_ind_c]+eps)/cubic_probs)
	ortho_rel_log = np.log((sample_probs[nonzero_ind_o]+eps)/ortho_probs)
	
	KL_cubic = np.sum(sample_probs[nonzero_ind_c] * cubic_rel_log )
	KL_ortho = np.sum(sample_probs[nonzero_ind_o] * ortho_rel_log)
	#print(f"xi_ang1 = {xi_ang1} xi_ang2 = {xi_ang2} KL_c = {KL_cubic} KL_o = {KL_ortho}")
	
	return xi_ang1, xi_ang2, xi, np.abs(KL_cubic), np.abs(KL_ortho)



def calc_anisotropy(xyz, tempflag, avgflag = 0, ang1 = 90, ang2 = 160):
	# If avgflag = 0, then the anisotropy is computed for the time averaged geometry
	# If avgflag = 1, then the anisotropy is calculated along the trajectory and then averaged
	print('\tCalculating anisotropy...')

	
	if tempflag != 0 and avgflag == 1:
		# Prepare containers for the various order parameters computed over the time series.
		size = []
		latcx = []
		latcy = []
		latcz = []
		lat_diff = []
		dipoles = []; dipoles_atom = []
		xi_ang1_list = []; xi_ang2_list = []; xi_list = []
		KL_cubic_list = []; KL_ortho_list = []

		# Prepare output files for running and block averages
		running_avg_file = open("xi_running_avg.dat", "w")
		dipole_avg_file = open("dipole_running_avg.dat", "w")

		block_size = 200
		block_avg_file = open(f"xi_block_avg_{block_size}.dat", "w")
		block_avg = []

		for i, xyz_step in enumerate(xyz):
			
			atype = xyz_step[:,0].astype(np.int8); xu = xyz_step[:,1]; yu = xyz_step[:,2]; zu = xyz_step[:,3]; q = xyz_step[:,4].reshape(-1,1)
			size.append(max(max(xu) - min(xu), max(yu) - min(yu), max(zu) - min(zu)))

			xpb = xu[atype == 2]
			ypb = yu[atype == 2]
			zpb = zu[atype == 2]
			npb = len(atype[atype == 2])
	
			ncube = int(np.power(npb+0.1, 1./3.))
			
			latpbx = (max(xpb)-min(xpb))/float(ncube-1.)
			latpby = (max(ypb)-min(ypb))/float(ncube-1.)
			latpbz = (max(zpb)-min(zpb))/float(ncube-1.)
			latcx.append(latpbx)
			latcy.append(latpby)
			latcz.append(latpbz)

			latc = np.sort(np.array([latpbx,latpby,latpbz]))
			minlc = latc[0]; maxlc = latc[2]

			lat_diff.append(maxlc-minlc) # The aspect ratio of the longest to shortest lattice parameter
			
			# Compute molecular dipoles
			dipole = np.sum(np.multiply(q, xyz_step[:,1:4]), axis = 0)
			dipole_mag = np.linalg.norm(dipole); dipole_mag_atom = dipole_mag/xyz_step.shape[0] # per atom dipole magnitude
			#print(f"Dipole moment: x = {dipole[0]} y = {dipole[1]} z = {dipole[2]}")
			dipoles.append(dipole_mag)
			dipoles_atom.append(dipole_mag_atom)
			dipole_avg_file.write(f"{i} {np.mean(dipoles)} {np.mean(dipoles_atom)}\n")

			# Compute angle distribution
			ang_bins, ang_probs = get_angles(xyz_step[:,:-1], ncube, plothist=0) 
			
			xi_ang1, xi_ang2, xi, KL_cubic, KL_ortho = calc_order_param(ang_bins, ang_probs, ang1, ang2)
			xi_ang1_list.append(xi_ang1); xi_ang2_list.append(xi_ang2); xi_list.append(xi)
			KL_cubic_list.append(KL_cubic); KL_ortho_list.append(KL_ortho)
			
			# Compute running and block averages
			running_avg = np.mean(np.array(xi_list), axis=0)
			running_avg_file.write(f"{i} {running_avg:.6f}\n")
			block_avg.append(xi)
			if i % block_size == 0:
				avg_xi = np.mean(np.array(block_avg), axis=0)
				block_avg_file.write(f"{i} {avg_xi:.6f}\n")
				block_avg = []
				
			#order_param.append(avg_max_diff)
		running_avg_file.close(); block_avg_file.close(); dipole_avg_file.close()

		size = np.mean(size)/10
		lat_diff = np.array(lat_diff); lat_diff_avg = np.mean(lat_diff, axis=0); lat_diff_std = np.std(lat_diff, axis=0)
		xi_ang1_list = np.array(xi_ang1_list); xi_ang1_avg = np.mean(xi_ang1_list, axis=0); xi_ang1_std = np.std(xi_ang1_list, axis=0)
		xi_ang2_list = np.array(xi_ang2_list); xi_ang2_avg = np.mean(xi_ang2_list, axis=0); xi_ang2_std = np.std(xi_ang2_list, axis=0)
		xi_list = np.array(xi_list); xi_avg = np.mean(xi_list, axis=0); xi_std = np.std(xi_list, axis=0)
		KL_cubic_list = np.array(KL_cubic_list); KL_cubic_avg = np.mean(KL_cubic_list, axis=0); KL_cubic_std = np.std(KL_cubic_list, axis=0)
		KL_ortho_list = np.array(KL_ortho_list); KL_ortho_avg = np.mean(KL_ortho_list, axis=0); KL_ortho_std = np.std(KL_ortho_list, axis=0)
		np.savetxt("xi_all.dat", xi_list)
		return size, (xi_ang1_avg, xi_ang1_std), (xi_ang2_avg, xi_ang2_std), (xi_avg, xi_std), (KL_cubic_avg, KL_cubic_std), (KL_ortho_avg, KL_ortho_std), (lat_diff_avg, lat_diff_std), np.mean(latcx), np.mean(latcy), np.mean(latcz)
	else:
		atype = xyz[:,0].astype(np.int8); xu = xyz[:,1]; yu = xyz[:,2]; zu = xyz[:,3]
		
		size = max(max(xu) - min(xu), max(yu) - min(yu), max(zu) - min(zu))/10 #A to nm

		xpb = xu[atype == 2]
		ypb = yu[atype == 2]
		zpb = zu[atype == 2]
		npb = len(atype[atype == 2])
		
		ncube = int(np.power(npb+0.1, 1./3.))
		latpbx = (max(xpb)-min(xpb))/float(ncube-1.)
		latpby = (max(ypb)-min(ypb))/float(ncube-1.)
		latpbz = (max(zpb)-min(zpb))/float(ncube-1.)
		
		latc = np.sort(np.array([latpbx,latpby,latpbz]))
		minlc = latc[0]; maxlc = latc[2]
		
		# ***** ***** ***** ***** ***** ***** *****
		# Compute the aspect ratio order parameter
		# ***** ***** ***** ***** ***** ***** *****

		lat_diff = maxlc-minlc # The aspect ratio of the longest to shortest lattice parameter
		
		# ***** ***** ***** ***** ***** ***** *****
		# Compute the angle distribution order parameters
		# ***** ***** ***** ***** ***** ***** *****

		ang_bins, ang_probs = get_angles(xyz, ncube) # The tetragonal distortion parameter is 0 for cubic and 1 for orthorhombic lattices.
		xi_ang1, xi_ang2, xi, KL_cubic, KL_ortho = calc_order_param(ang_bins, ang_probs, ang1, ang2)
		
		return size, xi_ang1, xi_ang2, xi, KL_cubic, KL_ortho, lat_diff, latpbx, latpby, latpbz


def main():
	print('\nRUNNING PROGRAM: calc_anisotropy.py...')
	# Take in options to configure script and output
	loop_flag = 0
	if 'loop' in sys.argv:
		loop_flag = 'loop'
	xyz_flag = 0 
	if 'xyz' in sys.argv:
		xyz_flag = 'xyz'
	temp_flag = 0
	if 'temp' in sys.argv:
		temp_flag = sys.argv[(sys.argv).index('temp')+1] 
	order_param = 'aspect'
	if 'angle' in sys.argv:
		order_param = 'angle'
		ang1 = float(sys.argv[(sys.argv).index('angle')+1])
		ang2 = float(sys.argv[(sys.argv).index('angle')+2])

	cwd = os.getcwd()
	print('Working directory: ', cwd)
	if loop_flag == 'loop':
		if 'noavg' in sys.argv:
			avg_flag = 0
		else:
			avg_flag = 1 # This is the correct way: average of the order parameter, not order parameter of the average.

		if temp_flag != 0:
			order_param_out = open('lat_anis_{}K_allsizes_out_{}_{}.dat'.format(temp_flag, order_param,avg_flag), 'w')
			order_param_out.write('size xi_ang1    var    xi_ang2     var    xi         var    KL_cubic   var   KL_ortho   var\n')
		else:
			order_param_out = open('lat_anis_0K_allsizes_out_{}.dat'.format(order_param), 'w')
			order_param_out.write('size xi_ang1 xi_ang2 KL_cubic KL_ortho\n')

		dir_list = ['2x2x2','3x3x3','4x4x4','5x5x5','6x6x6', '7x7x7']#,'8x8x8','9x9x9']#,'10x10x10']#,'11x11x11']#,'12x12x12', '14x14x14','15x15x15','16x16x16','17x17x17']#'10x10x10','11x11x11','12x12x12','18x18x18','19x19x19','24x24x24']
		
		print('\nLooping through directories to calculate size_dep anisotropy')
		for i in range(len(dir_list)):
			os.chdir(cwd + '/' + dir_list[i]+ '/{}'.format(temp_flag))
			print('\nCurrent dir: ', dir_list[i]+ '/{}'.format(temp_flag))
			
			if temp_flag:
				outfile = open('lat_const_{}K_out_{}.dat'.format(temp_flag, order_param), 'w')
				
				if avg_flag == 0:
					coords = get_coords(xyz_flag, temp_flag, avg_flag)
					write_output(dir_list[i]+'_{}K_avg'.format(temp_flag), coords)
					size, xi_ang1, xi_ang2, xi, KL_cubic, KL_ortho, lat_anis, latpbx, latpby, latpbz = calc_anisotropy(coords, temp_flag)
					order_param_out.write('{:.2f} {} {} {} {} {}\n'.format(np.mean(size), xi_ang1, xi_ang2, xi, KL_cubic, KL_ortho))
					outfile.write('lattice constant (diff) from Pb = %lf \n' %(lat_anis))
				elif avg_flag == 1:
					
					coords = get_coords(xyz_flag, temp_flag, avg_flag)
					size, xi_ang1, xi_ang2, xi, KL_cubic, KL_ortho, lat_anis, latpbx, latpby, latpbz = calc_anisotropy(coords, temp_flag, 1, ang1=ang1, ang2=ang2)
					order_param_out.write('{:.2f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(np.mean(size), xi_ang1[0], xi_ang1[1], xi_ang2[0], \
						xi_ang2[1], xi[0], xi[1], KL_cubic[0], KL_cubic[1], KL_ortho[0], KL_ortho[1]))
					outfile.write('lattice constant (diff) from Pb = %lf  %lf \n' %(lat_anis[0], lat_anis[1]))

				outfile.write('{:.2f}\n'.format(np.mean(size)))
				outfile.write('lattice constant (x) from Pb = %lf \n' %latpbx)
				outfile.write('lattice constant (y) from Pb = %lf \n' %latpby)
				outfile.write('lattice constant (z) from Pb = %lf \n' %latpbz)
				
				outfile.close()
			else:
				outfile = open('lat_const_0K_out_{}.dat'.format(order_param), 'w')
			
				coords = get_coords(xyz_flag, temp_flag, 0)
				size, xi_ang1, xi_ang2, xi, KL_cubic, KL_ortho, lat_anis, latpbx, latpby, latpbz  = calc_anisotropy(coords, temp_flag, ang1=ang1, ang2=ang2)
			
				order_param_out.write('{:.2f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(np.mean(size), xi_ang1, xi_ang2, xi, KL_cubic, KL_ortho))
				outfile.write('{:.2f}\n'.format(size))
				outfile.write('lattice constant (x) from Pb = %lf \n' %latpbx)
				outfile.write('lattice constant (y) from Pb = %lf \n' %latpby)
				outfile.write('lattice constant (z) from Pb = %lf \n' %latpbz)
				outfile.write('xi_ang1 = {} xi_ang2 = {} xi = {} KL_cubic = {} KL_ortho = {} lat_diff = {}\n'.format(xi_ang1, xi_ang2, xi, KL_cubic, KL_ortho, lat_anis))
				outfile.close()

	else:
		if temp_flag != 0:
			if 'noavg' in sys.argv:
				avg_flag = 0
			else:
				avg_flag = 1
				
			order_param_out = open('lat_anis_{}K_out_{}_{}.dat'.format(temp_flag,order_param,avg_flag), 'w')
			order_param_out.write('size xi_ang1    var    xi_ang2     var    xi         var    KL_cubic   var   KL_ortho   var\n')
			outfile = open('lat_const_{}K_out_{}_{}.dat'.format(temp_flag,order_param,avg_flag), 'w')

			coords = get_coords(xyz_flag, temp_flag, avg_flag)
			if avg_flag == 0:
				size, xi_ang1, xi_ang2, xi, KL_cubic, KL_ortho, lat_anis, latpbx, latpby, latpbz = calc_anisotropy(coords, temp_flag, order_param)
				write_output(str(size)+'_300K_avg', coords)
				order_param_out.write('{:.2f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(np.mean(size), xi_ang1, xi_ang2, xi, KL_cubic, KL_ortho))
				outfile.write('lattice constant (diff) from Pb = %lf \n' %(lat_anis))
			elif avg_flag == 1:
				
				size, xi_ang1, xi_ang2, xi, KL_cubic, KL_ortho, lat_anis, latpbx, latpby, latpbz = calc_anisotropy(coords, temp_flag, 1, ang1=ang1, ang2=ang2)
				order_param_out.write('{:.2f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(np.mean(size), xi_ang1[0], xi_ang1[1], xi_ang2[0], \
						xi_ang2[1], xi[0], xi[1], KL_cubic[0], KL_cubic[1], KL_ortho[0], KL_ortho[1]))
				outfile.write('lattice constant (diff) from Pb = %lf  %lf \n' %(lat_anis[0], lat_anis[1]))

			outfile.write('{:.2f}\n'.format(np.mean(size)))
			outfile.write('lattice constant (x) from Pb = %lf \n' %latpbx)
			outfile.write('lattice constant (y) from Pb = %lf \n' %latpby)
			outfile.write('lattice constant (z) from Pb = %lf \n' %latpbz)
			
			outfile.close()
		else:
			order_param_out = open('lat_anis_0K_out_{}.dat'.format(order_param), 'w')
			outfile = open('lat_const_0K_out_{}.dat'.format(order_param), 'w')

			coords = get_coords(xyz_flag, temp_flag, 0)
			size, xi_ang1, xi_ang2, xi, KL_cubic, KL_ortho, lat_anis, latpbx, latpby, latpbz  = calc_anisotropy(coords, temp_flag)
			
			order_param_out.write('{:.2f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(np.mean(size), xi_ang1, xi_ang2, xi, KL_cubic, KL_ortho))
			outfile.write('{:.2f}\n'.format(size))
			outfile.write('lattice constant (x) from Pb = %lf \n' %latpbx)
			outfile.write('lattice constant (y) from Pb = %lf \n' %latpby)
			outfile.write('lattice constant (z) from Pb = %lf \n' %latpbz)
			outfile.write('lattice constant (diff) from Pb = %lf \n' %order_param)
			outfile.close()
	print('\nDONE CALCULATING ANISOTROPY!')



main()

		