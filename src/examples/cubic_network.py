# importing Packages
import os
import openpnm as op
import numpy as np
import matplotlib.pyplot as plt

op.visualization.set_mpl_style()
path = os.path.dirname(__file__)

# Creating a Cubic Network
Nx, Ny, Nz = 10, 10 ,10
Lc = 1e-4
pn = op.network.Cubic([Nx, Ny, Nz], spacing=Lc)
fig1 = plt.figure() 
ax1 = fig1.add_subplot(111, projection='3d')  # Specify a 3D subplot
ax1 = op.visualization.plot_coordinates(pn, ax=ax1)
ax1 = op.visualization.plot_connections(pn, ax=ax1)
fig1.savefig(path+"/results/Network.png")

pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
pn.regenerate_models()
print(pn)
# inlet_nodes = pn.where(pn['pore.left'])[0]

#Defining a Phase
hg = op.phase.Mercury(network=pn)
hg.add_model(propname='throat.entry_pressure',
             model=op.models.physics.capillary_pressure.washburn)
hg.regenerate_models()
print(hg)

# Performing a Drainage Simulation
mip = op.algorithms.Drainage(network=pn, phase=hg)
mip.set_inlet_BC(pores=pn.pores(['left', 'right']))
mip.run(pressures=np.logspace(4, 6))

data = mip.pc_curve()
fig2, ax2 = plt.subplots(figsize=(5.5, 4))
ax2.semilogx(data.pc, data.snwp, 'k-o')
ax2.set_xlabel('capillary pressure [Pa]')
ax2.set_ylabel('mercury saturation');
fig2.savefig(path+"/results/PermeabilityCurve.png")