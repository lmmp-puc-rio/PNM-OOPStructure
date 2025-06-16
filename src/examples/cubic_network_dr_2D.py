# Cubic Network with Invasion Percolation
## importing Packages
import os, glob
import openpnm as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heapq as hq
import io
from PIL import Image, ImageGrab
import moviepy.video.io.ImageSequenceClip



op.visualization.set_mpl_style()
path = os.path.dirname(__file__)

# Create network and phases
"""First, we create a network and assign phases and properties in a similar way that we used to do for the other examples.""" 

# Creating a Cubic Network
npores = 25
np.random.seed(15)
fps = 10
trapping = 'trapping'  # Options: 'trapping', 'no_trapping'
Lc = 1e-6
spacing = 1e-4
pn = op.network.Cubic(shape=[npores, npores, 1], spacing=spacing)
pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
pn.regenerate_models()
pn['pore.diameter'] = np.random.rand(pn.Np)*Lc
pn['throat.diameter'] = np.random.rand(pn.Nt)*Lc
air = op.phase.Air(network=pn)
water = op.phase.Water(network=pn,name='water')
water.add_model_collection(op.models.collections.phase.water)
water.add_model_collection(op.models.collections.physics.basic)
water.regenerate_models()

os.makedirs(path+"/results/DR_"+trapping+"/videos/frames_"+str(npores)+"pores", exist_ok=True)
files = glob.glob(path+"/results/DR_"+trapping+"/videos/frames_"+str(npores)+"pores/*")
for f in files:
    os.remove(f)

# Pore size distribution
air['pore.contact_angle'] = 120
air['pore.surface_tension'] = 0.072
washburn = op.models.physics.capillary_pressure.washburn
air.add_model(propname='throat.entry_pressure',
              model=washburn, 
              surface_tension='throat.surface_tension',
              contact_angle='throat.contact_angle',
              diameter='throat.diameter',)

# Network Visualization
fig1,ax1 = plt.subplots()
op.visualization.plot_coordinates(pn, pn.pores('left'), size_by=pn['pore.diameter'], markersize=50, c='r',ax=ax1)
op.visualization.plot_connections(pn, pn.throats() ,size_by=pn['throat.diameter'], linewidth=5, alpha=0.5 , c='b' ,ax=ax1)
op.visualization.plot_coordinates(pn, pn.pores('left',mode= 'nor'),size_by=pn['pore.diameter'], markersize=50, c='b', ax=ax1)
fig1.savefig(path+"/results/DR_"+trapping+"/videos/frames_"+str(npores)+"pores/frame0.png")


fig3,ax3 = plt.subplots()
dr = op.algorithms.Drainage(network=pn, phase=air)
Inlet = pn.pores('left')
Outlet = pn.pores('right')
pn['pore.volume'][Inlet] = 0.0
dr.set_inlet_BC(pores=Inlet)
# dr.set_outlet_BC(pores=Outlet)

dr.run(pressures=100)
data_ip_no_trapping = dr.pc_curve(pressures=100)

## Apply Trapping
if trapping == 'trapping':
    dr.set_outlet_BC(pores=Outlet, mode='overwrite')
    dr.apply_trapping()

data_ip = dr.pc_curve(pressures=100)

image_files = []

op.visualization.plot_coordinates(pn, size_by=pn['pore.diameter'], markersize=50, c='b', ax=ax3);
op.visualization.plot_connections(pn, pn.throats() ,size_by=pn['throat.diameter'], linewidth=3, c='b' ,ax=ax3)
op.visualization.plot_coordinates(pn, pn.pores('left'), size_by=pn['pore.diameter'], c='r', ax=ax3)
fig3.savefig(path+"/results/DR_"+trapping+"/videos/frames_"+str(npores)+"pores/frame0.png")


for sequence in np.unique(dr['throat.invasion_sequence'][dr['throat.invasion_sequence']!= np.inf]):
    inv_throat_pattern = dr['throat.invasion_sequence'] <= sequence
    inv_pore_pattern = dr['pore.invasion_sequence'] <= sequence
    op.visualization.plot_coordinates(pn, inv_pore_pattern, size_by=pn['pore.diameter'], markersize=50, c='r', ax=ax3)
    op.visualization.plot_connections(pn, inv_throat_pattern,size_by=pn['throat.diameter'], linewidth=3, c='r' ,ax=ax3)
    # ax3.legend(['water_pores', 'water_throats', 'gas_pores', 'gas_throats'], loc='best')
    fig3.savefig(path+"/results/DR_"+trapping+"/videos/frames_"+str(npores)+"pores/frame"+str(sequence)+".png")
    image_files.append(path+"/results/DR_"+trapping+"/videos/frames_"+str(npores)+"pores/frame"+str(sequence)+".png")

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile(path+"/results/DR_"+trapping+"/videos/saturation_"+str(npores)+"pores.mp4")

#Capilary Pressure vs Non-Wetting Phase Saturation
fig5,ax5 = plt.subplots()
fig5.set_size_inches(10,10)
line1, = ax5.semilogx(data_ip.pc, data_ip.snwp*100, 'b-', label=trapping)
## Apply Trapping
if trapping == 'trapping':
    print(1)
    line2, = ax5.semilogx(data_ip_no_trapping.pc, data_ip_no_trapping.snwp*100, 'r--', label="no trapping")
    # dr.set_outlet_BC(pores=Outlet, mode='overwrite')
    lineHandles = [line1,line2]
else:
    print(2)
    lineHandles = [line1]
ax5.set_xlabel('Capillary Pressure [Pa]')
ax5.set_ylabel('Non-Wetting Phase Saturation[%]')
ax5.legend(handles=lineHandles, loc='best');
ax5.set_title('Cubic Network with Invasion Percolation')
# ax5.set_xlim(1e5, 1e8)
ax5.set_ylim(0, 105)
fig5.savefig(path+"/results/DR_"+trapping+"/CapillaryPressureXSaturation.png")