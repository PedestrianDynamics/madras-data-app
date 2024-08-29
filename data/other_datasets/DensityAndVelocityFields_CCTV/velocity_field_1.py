#! coding: utf-8 #

import numpy as np
import pandas as pd
from pathlib import Path
from math import atan, cos, sin, pi, ceil, floor, e
import os, re, pickle
from matplotlib import pyplot as plt
from scipy.signal import butter,filtfilt

########## PARAMETERS ########## 
start_time= 0. #  start time of the video sequence, in seconds, since 8pm on Friday
duration= 10. # duration of the animation, in seconds
dt= 1.0
xi= 0.75
r_c= 4.0*xi
r_cg= 0.25


FolderTraj=  "??????????????????????????????"  # folder with all the trajectory files
FolderSave= FolderTraj+"../Analysis/start=%.1f_dur=%.1f_dt=%.1f_xi=%.2f_rcg=%.2f/"%(start_time,duration,dt,xi,r_cg) # folder where the fields have been saved by the Python script "velocity_field_0.py"

x_min= 500
x_max= -1
y_min= 500
y_max= -1


DELTA= int(ceil(r_c/r_cg))+1 


################################

def norm(v: np.ndarray) -> float:
    """
    Calculate the Euclidean norm of a 2D vector.

    Parameters:
    - v (np.ndarray): A 2D vector.

    Returns:
    - float: The Euclidean norm of the vector.
    """
    return np.sqrt(v[0]**2 + v[1]**2)
    
def distance(pos1,pos2):
	return ( (pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2 )**0.5

	
def get_phi(r):
	if r>r_c:
		return 0.0
	return e**(-0.5 * (r/xi)**2) # prefactor is omitted, because it would cancel out anyway

def get_r(i,j):
	return ( float(i) * r_cg + 0.5 * r_cg + x_min, float(j) * r_cg + 0.5 * r_cg + y_min)
	
def get_Cell(r):
	i= int(floor((r[0]-x_min)/r_cg)) # removed "+ 0.5" on July 25th
	j= int(floor((r[1]-y_min)/r_cg)) # removed "+ 0.5" on July 25th
	return (i,j)
	
	
def from_xy_to_ij(x,y):
	return ( int(nb_cg_x*(x-x_min)/(x_max-x_min)), int(nb_cg_y*(y-y_min)/(y_max-y_min)) )

# Butterworth Filter requirements.
cutoff = 0.25     # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
Delta_t= 0.1

def butter_lowpass_filter(data, Delta_t, order):
	nyq = 0.5 / Delta_t # Nyquist Frequency
	normal_cutoff = cutoff / nyq
	# Get the filter coefficients
	b, a = butter(order, normal_cutoff, btype='low', analog=False)
	y = filtfilt(b, a, data, padlen=int(1/Delta_t)+1)
	return y

################################


#####
cpt_traj=0
nb_traj= len(os.listdir(FolderTraj))

all_trajs= {}

list_skipped= list([])
cum_time= 0. # cumulated time during which people are observed

max_files= 800 #!!

for trajFile in os.listdir(FolderTraj):
	max_files-=1
	if max_files<0:
		break
	m= re.search(r"converted_(\d+).txt",trajFile)

	if m==None:
		continue
			
	traj_no= int(m.group(1))
	
	
	# Display a progress bar #
	progress_pc= 100*cpt_traj//nb_traj
	progress_string= "Analysis progress   |"
	for cpt in range(20):
		progress_string+=">" if cpt<=progress_pc/5 else " "
	progress_string+= "| (%i/%i trajectories)"%(cpt_traj,nb_traj)
	cpt_traj+= 1
	print(progress_string, end='\r')
	# End of progress bar display #
	
	
	all_trajs[traj_no]= pd.read_csv(FolderTraj+"/"+trajFile, sep=';', header=0, usecols=["time_loc","x_loc","y_loc"])
	
	# Initiate speeds #
	all_trajs[traj_no]["vx"]= -100.
	all_trajs[traj_no]["vy"]= -100.
	
	
	# Smooth trajectories by applying Butterworth

	try:
		X_bw= butter_lowpass_filter(all_trajs[traj_no].x_loc,Delta_t,2)
		Y_bw= butter_lowpass_filter(all_trajs[traj_no].y_loc,Delta_t,2)
	except:
		print("I am passing on this one")
		X_bw= all_trajs[traj_no].x_loc
		Y_bw= all_trajs[traj_no].y_loc
		
	# For the beginning and end of trajectories, interpolate between actual and Butterworth-smoothed trajectories
	ts= all_trajs[traj_no]["time_loc"].min()
	te= all_trajs[traj_no]["time_loc"].max()
	alpha= all_trajs[traj_no]["time_loc"].apply(lambda t: max( np.exp(- 4.0*cutoff*(t - ts)),np.exp(- 4.0*cutoff*(te - t)) ) )# interpolation coefficient, should be 0 if we are in the bulk of the trajectory, 1 at the very start/end
	all_trajs[traj_no]["x_loc"]= (1.0-alpha) * X_bw + alpha * all_trajs[traj_no].x_loc
	all_trajs[traj_no]["y_loc"]= (1.0-alpha) * Y_bw + alpha * all_trajs[traj_no].y_loc
	#plt.plot(all_trajs[traj_no].x_loc, all_trajs[traj_no].y_loc,'b-')
	#plt.plot(all_trajs[traj_no]["X_bw"],all_trajs[traj_no]["Y_bw"], 'r--')
			
			
	
	# restrict focus
	all_trajs[traj_no]= all_trajs[traj_no].loc [ (all_trajs[traj_no]["time_loc"] >= start_time) & ( all_trajs[traj_no]["time_loc"] < start_time+duration+2.0*dt) ]
		
	# if the trajectory contains a single position, make a list out of it
	if all_trajs[traj_no].shape[0]==0:
		continue
	cum_time+= all_trajs[traj_no]["time_loc"].max() - all_trajs[traj_no]["time_loc"].min()
	#print(cum_time, " after examining ", traj_no)
	
	x_min= min(x_min, all_trajs[traj_no]["x_loc"].min())
	x_max= max(x_max, all_trajs[traj_no]["x_loc"].max())
	y_min= min(y_min, all_trajs[traj_no]["y_loc"].min())
	y_max= max(y_max, all_trajs[traj_no]["y_loc"].max())
	
	
	time_max= all_trajs[traj_no]["time_loc"].max()
	
	for row in all_trajs[traj_no].loc[ all_trajs[traj_no]["time_loc"] < time_max - dt].itertuples():
		current_time= row.time_loc
		next_row= all_trajs[traj_no][ all_trajs[traj_no]["time_loc"]>=current_time+dt ].iloc[0]
		next_time= next_row.time_loc
			
		if next_time-current_time > 2.*dt:
			list_skipped.append(traj_no)
			continue
		all_trajs[traj_no].at[row.Index,"vx"]= (next_row.x_loc-row.x_loc)/ (next_time-current_time)
		all_trajs[traj_no].at[row.Index,"vy"]= (next_row.y_loc-row.y_loc)/ (next_time-current_time)
		
	all_trajs[traj_no]= all_trajs[traj_no].loc[ all_trajs[traj_no]["vx"] > -100]
	

print("%i trajs have been skipped: "%(len(list_skipped)), list_skipped)


nb_cg_x= int( (x_max-x_min)/r_cg)+DELTA+2
nb_cg_y= int( (y_max-y_min)/r_cg)+DELTA+2
	




# Read the data #
with open(FolderSave+"X.pickle",'rb') as mydumpfile:
	X= pickle.load(mydumpfile)
with open(FolderSave+"Y.pickle",'rb') as mydumpfile:
	Y= pickle.load(mydumpfile)
with open(FolderSave+"Somme_phi.pickle",'rb') as mydumpfile:
	rho_array= pickle.load(mydumpfile)
with open(FolderSave+"vx_mean.pickle",'rb') as mydumpfile:
	vxs= pickle.load(mydumpfile)
with open(FolderSave+"vy_mean.pickle",'rb') as mydumpfile:
	vys= pickle.load(mydumpfile)
with open(FolderSave+"vx_var.pickle",'rb') as mydumpfile:
	vxs2= pickle.load(mydumpfile)
with open(FolderSave+"vy_var.pickle",'rb') as mydumpfile:
	vys2= pickle.load(mydumpfile)
with open(FolderSave+"v_std.pickle",'rb') as mydumpfile:
	std_vs= pickle.load(mydumpfile)




#######################

# PLOTTING #

fig= plt.figure(figsize=(5.5,7))
ax= plt.gca()

nb_ped= len(all_trajs.keys())

density= np.zeros( (nb_cg_x,nb_cg_y), dtype='d')

### PLOT DENSITY ###
# Renormalise density array

for traj in all_trajs.values():
	traj= traj.loc[ (traj["time_loc"] >= start_time) & (traj["time_loc"] < start_time+duration+2.0*dt) ]
	
	if traj.shape[0]==0:
		continue
	
	# Plot initial configuration
	#ax.plot(traj.x_loc,traj.y_loc,'ro',markersize=0.2)

	for row in traj.itertuples():
		R= (row.x_loc,row.y_loc)
		i_rel,j_rel= get_Cell(R)
		for i in range(i_rel-DELTA, i_rel+DELTA+1):
			for j in range(j_rel-DELTA, j_rel+DELTA+1):
				if i<0 or i>= nb_cg_x or j<0 or j>= nb_cg_y:
					continue
				phi_r= get_phi( distance( get_r(i,j) , R ) )
				density[i,j]+= phi_r
			
N_nonrenormalised= r_cg**2 * sum(sum(density))
density*= float(cum_time/duration) / float(N_nonrenormalised)
print("Total number of ped: ", nb_ped, " vs ", cum_time/duration)

Xp1= np.zeros( (nb_cg_x+1,nb_cg_y+1), dtype='d')
Yp1= np.zeros( (nb_cg_x+1,nb_cg_y+1), dtype='d')

for i in range(nb_cg_x+1):
	for j in range(nb_cg_y+1):
		Xp1[i,j]= float(i) * r_cg + x_min
		Yp1[i,j]= float(j) * r_cg + y_min 



cmesh= ax.pcolormesh(Xp1,Yp1,density, cmap='YlOrRd', vmax=4, shading='flat')
cb= plt.colorbar(cmesh)
cb.ax.set_title('Density (ped/m²)')

ax.set_xlabel("x (m)",fontsize=14)
ax.set_ylabel("y (m)",fontsize=14)


step=4
for i in range(nb_cg_x):
	for j in range(nb_cg_y):
		if X[i,j]>11.5:
			vxs[i,j]=0
			vys[i,j]=0
ax.quiver( X[::step,::step], Y[::step,::step], vxs[::step,::step], vys[::step,::step], linewidth=2, headwidth=7, scale=5.0 )
ax.quiver( 9,3.5,1,0, linewidth=3, headwidth=7, scale=5.0, color='green' )
ax.text(9.5,3.9,"1 m/s", color="green", fontsize=12)

ax.axis("equal")
plt.tight_layout()

# Boundaries
# 0.149 2.183	
# 11.637;3.086
# 11.964;26.187
# -2.724;24.941
plt.show()
quit()

