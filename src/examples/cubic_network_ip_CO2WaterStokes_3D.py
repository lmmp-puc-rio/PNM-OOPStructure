# importing Packages
import openpnm as op
import numpy as np
import os as os
import glob
import matplotlib.pyplot as plt
import moviepy.video.io.ImageSequenceClip
from rich.progress import Progress  

op.visualization.set_mpl_style()
path = os.path.dirname(__file__)

npores = 10
np.random.seed(15)
fps = 10
trapping = 'trapping'  # Options: 'trapping', 'no_trapping'
Lc = 1e-6
spacing = 1e-4
# Create a cubic network
pn = op.network.Cubic(shape=[npores, npores, npores], spacing=spacing)

msize = 100  # marker size for visualization
lwidth = 3  # line width for visualization
azim = -60
elev = 15

max_lenght = npores*spacing

pn_dim = 2
if len(np.unique(pn['pore.coords'].T[2])) > 1:
    pn_dim = 3

pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
pn.regenerate_models()
pn['pore.diameter'] = np.random.rand(pn.Np)*Lc
pn['throat.diameter'] = np.random.rand(pn.Nt)*Lc
co2 = op.phase.Air(network=pn,name='CO2')
co2['pore.surface_tension'] = 0.023 # Surface tension of CO2 (N/m)
co2['pore.contact_angle'] = 180.0

co2.add_model_collection(op.models.collections.phase.air)
co2.add_model_collection(op.models.collections.physics.basic)
co2.regenerate_models()
water = op.phase.Water(network=pn,name='water')
water['throat.diffusivity'] = 1e-8  # Water's diffusivity in scCO2 ( m²/s )
water['pore.viscosity'] = 1e-5  # Water's viscosity ( Pas )
water.add_model_collection(op.models.collections.phase.water)
water.add_model_collection(op.models.collections.physics.basic)
water.regenerate_models()

graph_path = os.path.join(path, 'results', f'IPRelPerm_{trapping}', 'graphs')
video_path = os.path.join(path, 'results', f'IPRelPerm_{trapping}', 'videos')
frame_path = os.path.join(video_path, f'frames_{npores}_pores')

os.makedirs(graph_path, exist_ok=True)
os.makedirs(frame_path , exist_ok=True)

fig0,ax0 = plt.subplots()
op.visualization.plot_coordinates(pn, size_by=pn['pore.diameter'], markersize=msize,  c='b',alpha=0.8, ax=ax0)
op.visualization.plot_connections(pn, size_by=pn['throat.diameter'], linewidth=lwidth, c='b',alpha=0.8, ax=ax0)
fig0.savefig(os.path.join(graph_path, f'Network3D_CO2WaterStokes_{trapping}{npores}.png'))
dr = op.algorithms.Drainage(network=pn, phase=co2)
Inlet = pn.pores('left')
Outlet = pn.pores('right')
dr.set_inlet_BC(pores=Inlet)
pn['pore.volume'][Inlet] = 0.0
# Set the capillary pressure model
dr.run(pressures=200)

Snwp_num=30
flow_in = pn.pores('left')
flow_out = pn.pores('right')

def sat_occ_update(network, nwp, wp, dr, i):
    r"""
        Calculates the saturation of each phase using the invasion
        sequence from either invasion percolation.
        Parameters
        ----------
        network: network
        nwp : phase
            non-wetting phase
        wp : phase
            wetting phase
        dr : IP
            invasion percolation (ran before calling this function)
        i: int
            The invasion_sequence limit for masking pores/throats that
            have already been invaded within this limit range. The
            saturation is found by adding the volume of pores and thorats
            that meet this sequence limit divided by the bulk volume.
    """
    pore_mask = dr['pore.invasion_sequence'] < i
    throat_mask = dr['throat.invasion_sequence'] < i
    sat_p = np.sum(network['pore.volume'][pore_mask])
    sat_t = np.sum(network['throat.volume'][throat_mask])
    sat1 = sat_p + sat_t
    bulk = network['pore.volume'].sum() + network['throat.volume'].sum()
    sat = sat1/bulk
    nwp['pore.occupancy'] = pore_mask
    nwp['throat.occupancy'] = throat_mask
    wp['throat.occupancy'] = 1-throat_mask
    wp['pore.occupancy'] = 1-pore_mask
    return sat

def Rate_calc(network, phase, left, outlet, conductance):
    phase.regenerate_models()
    St_p = op.algorithms.StokesFlow(network=network, phase=phase)
    St_p.settings._update({'conductance' : conductance})
    St_p.set_value_BC(pores=left, values=1)
    St_p.set_value_BC(pores=outlet, values=0)
    St_p.run()
    val = np.abs(St_p.rate(pores=left, mode='group'))
    return val

# Define Multiphase Conductance model
model_mp_cond = op.models.physics.multiphase.conduit_conductance
co2.add_model(model=model_mp_cond, propname='throat.conduit_hydraulic_conductance',
              throat_conductance='throat.hydraulic_conductance', mode='medium', regen_mode='deferred')
water.add_model(model=model_mp_cond, propname='throat.conduit_hydraulic_conductance',
              throat_conductance='throat.hydraulic_conductance', mode='medium', regen_mode='deferred')


# Max Invasion Sequence
max_seq = np.max(dr['throat.invasion_sequence'])

start = 0 # max_seq//Snwp_num
stop = max_seq+1
step = max_seq//Snwp_num
print("Start: ", start, " Stop: ", stop, " Step: ", step)
Snwparr = []
relperm_nwp = []
relperm_wp = []


## Apply Trapping
if trapping == 'trapping':
    dr.set_outlet_BC(pores=pn.pores('right'), mode='overwrite')
    dr.apply_trapping()


tmask = np.isfinite(dr['throat.invasion_sequence'])

max_seq = np.max(dr['throat.invasion_sequence'][tmask])
start = 0 # max_seq//Snwp_num
stop = max_seq+1
step = max_seq//Snwp_num
print("Start: ", start, " Stop: ", stop, " Step: ", step)
Snwparr = []
relperm_nwp = []
relperm_wp = []

# Loop through invasion sequence to calculate saturation and relative permeability
for i in range(start, int(stop), int(step)):
    co2.regenerate_models()
    water.regenerate_models()
    sat = sat_occ_update(network=pn, nwp=co2, wp=water, dr=dr, i=i)
    Snwparr.append(sat)
    Rate_abs_nwp = Rate_calc(pn, co2, flow_in, flow_out, conductance = 'throat.hydraulic_conductance')
    Rate_abs_wp = Rate_calc(pn, water, flow_in, flow_out, conductance = 'throat.hydraulic_conductance')
    Rate_enwp = Rate_calc(pn, co2, flow_in, flow_out, conductance = 'throat.conduit_hydraulic_conductance')
    Rate_ewp = Rate_calc(pn, water, flow_in, flow_out, conductance = 'throat.conduit_hydraulic_conductance')
    relperm_nwp.append(Rate_enwp/Rate_abs_nwp)
    relperm_wp.append(Rate_ewp/Rate_abs_wp)


left = pn.pores('left')
Domain = pn.pores('left', mode='not')
k=0

data_ip3D_trapping = dr.pc_curve()

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(Snwparr, relperm_nwp, '*-', label='Kr_nwp')
ax1.plot(Snwparr, relperm_wp, 'o-', label='Kr_wp')
ax1.set_xlabel('Snwp')
ax1.set_ylabel('Kr')
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1.05)
ax1.set_title(f'Relative Permeability in x direction - {trapping.replace("_"," ")}', fontsize=16)
ax1.legend()
fig1.savefig(os.path.join(graph_path, f'IPRelPerm_{trapping}.png'))


# Making fames of Invasion Percolation saturation field
files = glob.glob(os.path.join(frame_path, '*'))
for f in files:
    os.remove(f)


image_files = []

fig2 = plt.figure()
ax2 = fig2.add_subplot(111) if pn_dim==2 else fig2.add_subplot(111, projection = '3d')
op.visualization.plot_coordinates(pn, pn.pores('left',mode= 'nor'),size_by=pn['pore.diameter'], markersize=msize,alpha=0.3, c='b', ax=ax2)
op.visualization.plot_connections(pn, pn.throats() ,size_by=pn['throat.diameter'], linewidth=lwidth, c='b' ,alpha=0.3,ax=ax2)
op.visualization.plot_coordinates(pn, pn.pores('left'), size_by=pn['pore.diameter'], markersize=msize, c='r',alpha=0.5,ax=ax2)
fig2.set_size_inches(npores, npores)
ax2.set_aspect('auto')
ax2.set_title(f'Pressure = {(data_ip3D_trapping.pc[k])} Pa',fontsize=16)
ax2.set_xlim((0, max_lenght))
ax2.set_ylim((0, max_lenght))
if pn_dim == 3:
    ax2.set_zlim((0, max_lenght))
    ax2.view_init(elev=elev, azim=azim)
# plt.show()
fig2.savefig(os.path.join(frame_path,'frame0.png'))
invasion_sequence = np.unique(dr['throat.invasion_sequence'][dr['throat.invasion_sequence']!= np.inf])

steps = 20
single_step = len(invasion_sequence)//20

with Progress() as p:
    t = p.add_task("Generating Video:", total=steps)
    for sequence in invasion_sequence[::single_step]:
        invasion_pressure = max(dr['throat.invasion_pressure'][dr['throat.invasion_sequence'] == sequence])/1000
        k += 1
        inv_throat_pattern = dr['throat.invasion_sequence'] <= sequence
        inv_pore_pattern = dr['pore.invasion_sequence'] <= sequence
        op.visualization.plot_connections(pn, inv_throat_pattern,size_by=pn['throat.diameter'],alpha=0.8, linewidth=lwidth, c='r' ,ax=ax2)
        op.visualization.plot_coordinates(pn, inv_pore_pattern, size_by=pn['pore.diameter'], markersize=msize, c='r',ax=ax2)
        ax2.set_aspect('auto')
        ax2.set_title(f'Pressure = {invasion_pressure:.2f} kPa',fontsize=16)
        ax2.set_xlim((0, max_lenght))
        ax2.set_ylim((0, max_lenght))
        if pn_dim == 3:
            ax2.set_zlim((0, max_lenght))
            ax2.view_init(elev=elev, azim=azim)
        fig2.savefig(os.path.join(frame_path,f'frame{k}.png'))
        image_files.append(os.path.join(frame_path,f'frame{k}.png'))
        p.update(t, advance=1)
        non_invaded_pores = dr['pore.invasion_sequence'] > sequence
        non_invaded_throats = dr['throat.invasion_sequence'] > sequence
    t = p.add_task("Irreducible Water:", total=10)
    if trapping == 'trapping':
        for j in range(0, 10, 1):
            for collection in ax2.collections:
                collection.remove()
            op.visualization.plot_coordinates(pn, non_invaded_pores, size_by=pn['pore.diameter'], alpha=j/10,markersize=msize, c='b',ax=ax2)
            op.visualization.plot_connections(pn, non_invaded_throats ,size_by=pn['throat.diameter'], linewidth=lwidth,alpha=j/10, c='b' ,ax=ax2)
            op.visualization.plot_connections(pn, inv_throat_pattern,size_by=pn['throat.diameter'], linewidth=lwidth,alpha=1-j/10, c='r' ,ax=ax2)
            op.visualization.plot_coordinates(pn, inv_pore_pattern, size_by=pn['pore.diameter'], markersize=msize,alpha=1-j/10, c='r',ax=ax2)
            ax2.set_title(f'Pressure = {invasion_pressure} kPa',fontsize=16)
            ax2.set_aspect('auto')
            ax2.set_xlim((0, max_lenght))
            ax2.set_ylim((0, max_lenght))
            if pn_dim == 3:
                ax2.set_zlim((0, max_lenght))
                ax2.view_init(elev=elev, azim=azim)
            k += 1
            fig2.savefig(os.path.join(frame_path,f'frame{k}.png'))
            image_files.append(os.path.join(frame_path,f'frame{k}.png'))
            p.update(t, advance=1)
    if pn_dim == 3:
        t = p.add_task("Rotação:", total=36)
        for l in range(0, 36, 1):
            ax2.view_init(elev=elev, azim=azim+l*10)
            k += 1
            fig2.savefig(os.path.join(frame_path,f'frame{k}.png'))
            image_files.append(os.path.join(frame_path,f'frame{k}.png'))
            p.update(t, advance=1)

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile(os.path.join(video_path,f'saturation_ipRelPerm_{trapping}_{npores}.mp4'))
