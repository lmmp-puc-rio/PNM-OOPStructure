# importing Packages
import openpnm as op
import numpy as np
import os as os
import glob
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip

op.visualization.set_mpl_style()
path = os.path.dirname(__file__)

# Setup Necessary Objects

np.random.seed(10)
fps = 5
n_pores = 20
pressure_points = 30
spacing = 1e-2
azim = -45
elev = 15

pn = op.network.Demo(shape=[n_pores, n_pores, n_pores], spacing=spacing)
max_lenght = n_pores*spacing
air = op.phase.Air(network=pn)


fig0,ax0 = plt.subplots()
op.visualization.plot_coordinates(pn, size_by=pn['pore.diameter'], markersize=50, c='b', ax=ax0)
op.visualization.plot_connections(pn, size_by=pn['throat.diameter'], linewidth=3, c='b', ax=ax0)
fig0.savefig(path+"/results/Network3D.png")

fig, (ax2, ax2) = plt.subplots(1, 2, figsize=[12,4])
ax2.hist(pn['pore.diameter'], bins=25, edgecolor='k')
ax2.set_title('Pore Diameter')
ax2.hist(pn['throat.diameter'], bins=25, edgecolor='k')
ax2.set_title('Throat Diameter');

## Define capillary pressure as 70% of throats mean entry pressure
air['pore.contact_angle'] = 120
air['pore.surface_tension'] = 0.072
f = op.models.physics.capillary_pressure.washburn
air.add_model(propname='throat.entry_pressure',
              model=f, 
              surface_tension='throat.surface_tension',
              contact_angle='throat.contact_angle',
              diameter='throat.diameter',)
Pc = air['throat.entry_pressure'].mean()*0.7
Ts = 1.0*(air['throat.entry_pressure'] < Pc)

os.makedirs(path+"/results/videos/drn3D_trapping/frames_"+str(n_pores)+"_pores/", exist_ok=True)
files = glob.glob(path+"/results/videos/drn3D_trapping/frames_"+str(n_pores)+"_pores/*")
for f in files:
    os.remove(f)
image_files = []

# Using Drainage Algorithm

InletPores = pn.pores('left')
OutletPores = pn.pores('right')
InletNeighborThroats = pn.find_neighbor_throats(pores=InletPores, mode='or')
# Outlet = pn.pores('right')
pn['pore.volume'][InletPores] = 0.0
pn['pore.volume'][OutletPores] = 0.0
drn = op.algorithms.Drainage(network=pn, phase=air)
drn.set_inlet_BC(pores=InletPores)
drn.set_outlet_BC(pores=OutletPores)

# Run Drainage Algorithm
drn.run(pressures=pressure_points)
drn.set_outlet_BC(pores=pn.pores('right'), mode='overwrite')
drn.apply_trapping()
data_drn3D_trapping = drn.pc_curve()

print(pn)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
Inlet = pn.pores('left')
Domain = pn.pores('left', mode='not')
k=0
op.visualization.plot_coordinates(pn, pn.pores('left',mode= 'nor'),size_by=pn['pore.diameter'], markersize=100, c='b', ax=ax2)
op.visualization.plot_connections(pn, pn.throats() ,size_by=pn['throat.diameter'], linewidth=7, c='b' ,alpha=0.3,ax=ax2)
op.visualization.plot_coordinates(pn, pn.pores('left'), size_by=pn['pore.diameter'], markersize=100, c='r',ax=ax2)
fig2.set_size_inches(n_pores, n_pores)
ax2.set_aspect('auto')
ax2.set_title('Pressure = '+str(data_drn3D_trapping.pc[k])+' Pa',fontsize=2*n_pores)
ax2.set_xlim((0, max_lenght))
ax2.set_ylim((0, max_lenght))
ax2.set_zlim((0, max_lenght))
ax2.view_init(elev=elev, azim=azim)
# plt.show()
fig2.savefig(path+"/results/videos/drn3D_trapping/frames_"+str(n_pores)+"_pores/frame0.png")
print(len(drn['throat.invasion_sequence']))

print(drn)
for sequence in np.unique(drn['throat.invasion_sequence'][drn['throat.invasion_sequence']!= np.inf]):
    invasion_pressure = max(drn['throat.invasion_pressure'][drn['throat.invasion_sequence'] == sequence])
    k += 1
    inv_throat_pattern = drn['throat.invasion_sequence'] <= sequence
    inv_pore_pattern = drn['pore.invasion_sequence'] <= sequence
    op.visualization.plot_connections(pn, inv_throat_pattern,size_by=pn['throat.diameter'],alpha=1, linewidth=3, c='r' ,ax=ax2);
    op.visualization.plot_coordinates(pn, inv_pore_pattern, size_by=pn['pore.diameter'], markersize=50, c='r',ax=ax2);
    ax2.set_aspect('auto')
    ax2.set_title('Pressure = '+str(invasion_pressure)+' Pa',fontsize=2*n_pores)
    ax2.set_xlim((0, max_lenght))
    ax2.set_ylim((0, max_lenght))
    ax2.set_zlim((0, max_lenght))
    ax2.view_init(elev=elev, azim=azim)
    fig2.savefig(path+"/results/videos/drn3D_trapping/frames_"+str(n_pores)+"_pores/frame"+str(k)+".png")
    image_files.append(path+"/results/videos/drn3D_trapping/frames_"+str(n_pores)+"_pores/frame"+str(k)+".png")

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile(path+"/results/videos/drn3D_trapping/saturation_drn3D_trapping_"+str(n_pores)+".mp4")

# print(len(data_drn3D_trapping.pc))
# print(len(np.unique(drn['throat.invasion_sequence'][drn['throat.invasion_sequence']!= np.inf])))

fig3,ax3 = plt.subplots()
fig3.set_size_inches(10,10)
line1, = ax3.plot(data_drn3D_trapping.pc, data_drn3D_trapping.snwp*100, 'b-o', label='without trapping', linewidth=1)
ax3.set_xlabel('Capillary Pressure [Pa]')
ax3.set_ylabel('Non-Wetting Phase Saturation[%]')
ax3.legend(handles=[line1], loc='best')
ax3.set_title('Cubic Network with Drainage - '+str(n_pores)+'_pores',fontsize=20);
ax3.set_ylim(0, 105)
fig3.savefig(path+"/results/drn3D_CapillaryPressureXSaturation_"+str(n_pores)+".png")