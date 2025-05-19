import numpy as np;
import math
import sys


#########################################################################################
#read inputs

if len(sys.argv)!=4 and len(sys.argv)!=5:
    print("Invalid number of arguments\nUsage: python perovMixBuild.py nx ny nz [par]\n")
    exit()


nx = int(sys.argv[1])
ny = int(sys.argv[2])
nz = int(sys.argv[3])


#par = 0 means cubic, par = 1 means ortho
if len(sys.argv)==5: 
	par = float(sys.argv[4])
	if par == 0: pars = 'cubic'
	else: pars = 'ortho'
	if(0>par or par>1):
		print(f"Need  0 <= par <= 1!")
		exit()
else: 
	par, pars = 0.0, 'cubic'

if(nx*ny*nz>10000):
	print(f"Woah there! This dot is too big (nx: {nx} ny: {ny} nz: {nz})")
	exit()

#########################################################################################
#set up needed info
#########################################################################################
#set up cubic and ortho parameters


cubicScale = 11.1*math.sqrt(2)
cubicA = 1.0
cubicB = 1.0
cubicC = math.sqrt(2)

orthoScale = 15.594
orthoA = 1.0
orthoB = 0.9940620455647114
orthoC = 1.4220794958797864

#########################################################################################
#interpolate between

parScale = cubicScale*(1-par)+orthoScale*par
parA = cubicA*(1-par)+orthoA*par
parB = cubicB*(1-par)+orthoB*par
parC = cubicC*(1-par)+orthoC*par



d1 = (0.25-0.19462)*par
d2 = (0.25-0.19731)*par
d3 = 0.03577*par
d4 = 0.00113*par
d5 = 0.06202*par
d6 = 0.04005*par
d7 = 0.00509*par

#########################################################################################

#details for basic conf
#units in BOHR!!!

basis_vectors= parScale * np.array([[ parA/math.sqrt(2), -parA/math.sqrt(2), 0 ],[ parB/math.sqrt(2), parB/math.sqrt(2), 0 ], [ 0, 0, parC ]])
atoms = np.array(["Cs", "Cs", "Cs", "Cs", "Pb", "Pb", "Pb", "Pb", "Br", "Br", "Br", "Br", "Br", "Br", "Br", "Br", "Br", "Br", "Br", "Br"])

lxy = 0.5 * parScale*math.sqrt(parA**2+parB**2)
lz = 0.5 * parScale*parC

atom_positions = np.array([
[0.50-d6, 0.5+d7, 0.25000], #Cs
[0.50+d6, 0.5-d7, 0.75000], #Cs
[1.00-d6, 0.0+d7, 0.25000], #Cs
[0.00+d6, 1.0-d7, 0.75000], #Cs
[0.50000, 0.00000, 0.0000], #Pb
[0.00000, 0.50000, 0.0000], #Pb
[0.50000, 0.00000, 0.5000], #Pb
[0.00000, 0.50000, 0.5000], #Pb
[0.25-d1, 0.25-d2, 0.0+d3], #Br
[0.75-d1, 0.25+d2, 0.0+d3], #Br
[0.75+d1, 0.75+d2, 1.0-d3], #Br
[0.25+d1, 0.75-d2, 1.0-d3], #Br

[0.25-d1, 0.25-d2, 0.5-d3], #Br
[0.75-d1, 0.25+d2, 0.5-d3], #Br
[0.75+d1, 0.75+d2, 0.5+d3], #Br
[0.25+d1, 0.75-d2, 0.5+d3], #Br

[0.00+d4, 0.50+d5, 0.2500], #Br
[0.50+d4, 1.00-d5, 0.2500], #Br
[1.00-d4, 0.50-d5, 0.7500], #Br
[0.50-d4, 0.00+d5, 0.7500]])#Br


#details for lammps config
natom_types = 3
atom_type_map = np.array(["Cs","Pb","Br"])
atom_numbers = [55, 82, 35, 55]
atom_masses = [132.91, 207.20, 79.904, 132.91]
atom_charges = [0.86, 1.03,-0.63, 0.86]
formal_charges = [1,2,-1,1]


#########################################################################################
#build bulk to cut dot from
#cut the nanocrystal from bulk
xmin = -lxy * (math.floor(nx/2.0)+0.1)
xmax =  lxy * (math.ceil(nx/2.0)+0.1)

ymin = -lxy * (math.floor(ny/2.0)+0.1)
ymax =  lxy * (math.ceil(ny/2.0)+0.1)

zmin = -0.51*lz
zmax = lz * (nz-0.49)



built_dot = []


for x_cell in range(-nx,int(nx)+1):
	for y_cell in range (-ny, int(ny)+1):
		for z_cell in range(-1,int(nz)+1):
				for i,atom in enumerate(atoms):
					pos = basis_vectors.transpose() @ atom_positions[i] + basis_vectors.transpose() @ [x_cell,y_cell,z_cell]
					if pos[0]>=xmin and pos[0]<=xmax and pos[1]>=ymin and pos[1]<=ymax and pos[2] >= zmin and pos[2]<=zmax:
						print(atom, pos[0], pos[1], pos[2])
						built_dot.append([atom, pos[0], pos[1], pos[2]])




#########################################################################################
#balance formal charge

natoms = len(built_dot)

formal_charge = 0
surface_list = []
to_remove = []

print(ny*basis_vectors[1][1])

for i in range(natoms):
	if built_dot[i][1]<= 0 or built_dot[i][2]<=0 or built_dot[i][3]<=0:
		surface_list.append(i)
	elif built_dot[i][1]>=(nx-1)*basis_vectors[0][0] or built_dot[i][2]>=(ny-1)*basis_vectors[1][1] or built_dot[i][3]>=(nz-1)*basis_vectors[2][2]:
		surface_list.append(i)

np.savetxt('surface.txt',surface_list, '%d')




#########################################################################################
#write outputs



#filter output


filter_of = open("conf.par", "w")
filter_of.write(f"{natoms}\n")

for i in range(natoms):
	if not(i in to_remove):
		filter_of.write(f"{built_dot[i][0]} {built_dot[i][1]} {built_dot[i][2]} {built_dot[i][3]}\n")
filter_of.close()


#lammps output
bohr_to_ang = 0.529177
built_dot = np.array(built_dot)
xpos=built_dot[:,1].astype(float)*bohr_to_ang
ypos=built_dot[:,2].astype(float)*bohr_to_ang
zpos=built_dot[:,3].astype(float)*bohr_to_ang


xlo = min(xpos)-500
xhi = max(xpos)+500

ylo = min(ypos)-500
yhi = max(ypos)+500

zlo = min(zpos)-500
zhi = max(zpos)+500

total_charge = 0

lammps_of = open("lammpsconf.par", "w")
lammps_of.write(f"#LAMMPS configuration file for perovskite NanoCube (nx: {nx} ny: {ny} nz: {nz})\n")
lammps_of.write(f"\n")
lammps_of.write(f"{natoms-formal_charge} atoms\n")
lammps_of.write(f"\n")
lammps_of.write(f"{natom_types} atom types\n")
lammps_of.write(f"\n")
lammps_of.write(f"{xlo} {xhi} xlo xhi\n")
lammps_of.write(f"{ylo} {yhi} ylo yhi\n")
lammps_of.write(f"{zlo} {zhi} zlo zhi\n")
lammps_of.write(f"\n")
lammps_of.write(f"Masses\n")
lammps_of.write(f"\n")
for i in range(natom_types):
	lammps_of.write(f"{i+1} {atom_masses[i]} #{atom_type_map[i]}\n")
lammps_of.write(f"\n")
lammps_of.write(f"Atoms\n")
lammps_of.write(f"\n")
for i in range(natoms):
	atom_type = np.where(atom_type_map == built_dot[i,0])[0][0]
	if not(i in to_remove):
		total_charge+=atom_charges[atom_type]
		lammps_of.write(f"{i+1} {atom_type+1} {atom_charges[atom_type]} {xpos[i]} {ypos[i]} {zpos[i]}\n")
print(f"total charge: {total_charge}")
lammps_of.close()


#xyz output
xyz_of = open("{}x{}x{}_{}.xyz".format(nx, ny, nz, pars), "w")
xyz_of.write(f"{natoms}\n")
xyz_of.write(f"#xyz file for perovskite NanoCube (nx: {nx} ny: {ny} nz: {nz})\n")
for i in range(natoms):
	if not(i in to_remove):
		xyz_of.write(f"{built_dot[i,0]} {xpos[i]} {ypos[i]} {zpos[i]}\n")
xyz_of.close()


#xyz output
xyz_of = open("{}x{}x{}_{}.xyz".format(nx, ny, nz, pars), "w")
xyz_of.write(f"{natoms}\n")
xyz_of.write(f"#xyz file for perovskite NanoCube (nx: {nx} ny: {ny} nz: {nz})\n")
for i in range(natoms):
	if not(i in to_remove):
		xyz_of.write(f"{built_dot[i,0]} {xpos[i]} {ypos[i]} {zpos[i]}\n")
xyz_of.close()


exit()


