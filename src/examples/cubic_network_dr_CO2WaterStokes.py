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
pn = op.network.Cubic(shape=[npores, npores, 1], spacing=spacing)

msize = 100  # marker size for visualization
lwidth = 3  # line width for visualization
azim = -60
elev = 15

max_lenght = npores*spacing

pn_dim = '2D'
if len(np.unique(pn['pore.coords'].T[2])) > 1:
    pn_dim = '3D'

pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
pn.regenerate_models()
pn['pore.diameter'] = np.random.rand(pn.Np)*Lc
pn['throat.diameter'] = np.random.rand(pn.Nt)*Lc
co2 = op.phase.Air(network=pn,name='CO2')
co2['pore.surface_tension'] = 0.023 # Surface tension of CO2 (N/m)
co2['pore.contact_angle'] = 150.0

co2.add_model_collection(op.models.collections.phase.air)
co2.add_model_collection(op.models.collections.physics.basic)
co2.regenerate_models(exclude=['pore.surface_tension','pore.contact_angle'])
water = op.phase.Water(network=pn,name='water')
water['throat.diffusivity'] = 1e-8  # Water's diffusivity in scCO2 ( m²/s )
water['pore.viscosity'] = 1e-5  # Water's viscosity ( Pas )
water['pore.surface_tension'] = 0.023 # Surface tension of CO2 (N/m)
water['pore.contact_angle'] = 30.0
water.add_model_collection(op.models.collections.phase.water)
water.add_model_collection(op.models.collections.physics.basic)
water.regenerate_models(exclude=['pore.contact_angle','pore.surface_tension','throat.diffusivity','pore.viscosity'])

graph_path = os.path.join(path, 'results', f'DRRelPerm{pn_dim}_{trapping}', 'graphs')
video_path = os.path.join(path, 'results', f'DRRelPerm{pn_dim}_{trapping}', 'videos')
frame_path = os.path.join(video_path, f'frames_{npores}_pores')

os.makedirs(graph_path, exist_ok=True)
os.makedirs(frame_path , exist_ok=True)

fig0 = plt.figure()
ax0 = fig0.add_subplot(111, projection = '3d') if pn_dim == '3D' else fig0.add_subplot(111)
fig0.set_size_inches(npores, npores)
ax0.set_aspect('auto')
ax0.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
ax0.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
ax0.grid(False)
ax0.set_title(f'Pore Network',fontsize=16)

linewidth = pn['throat.diameter'] / pn['throat.diameter'].max() * lwidth
markersize = pn['pore.diameter'] / pn['pore.diameter'].max() * msize

if pn_dim == '3D':
    ax0.view_init(elev=elev, azim=azim)
op.visualization.plot_coordinates(pn, markersize=markersize, c='b', alpha=0.8, ax=ax0)
op.visualization.plot_connections(pn, linewidth=linewidth, c='b', alpha=0.8, ax=ax0)
fig0.savefig(os.path.join(graph_path, f'Network{pn_dim}_CO2WaterStokes_{trapping}{npores}.png'))
dr = op.algorithms.Drainage(network=pn, phase=co2)
Inlet = pn.pores('left')
Outlet = pn.pores('right')
dr.set_inlet_BC(pores=Inlet)
pn['pore.volume'][Inlet] = 0.0
dr.run(pressures=200)

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
        dr : DR
            drainage (ran before calling this function)
        i: int
            The invasion_sequence limit for masking pores/throats that
            have already been invaded within this limit range. The
            saturation is found by adding the volume of pores and thorats
            that meet this sequence limit divided by the bulk volume.
    """
    phase = next((phase for phase in pn.project.phases if phase.name == dr.settings.phase))
    pore_mask = dr['pore.invasion_sequence'] < i
    throat_mask = dr['throat.invasion_sequence'] < i
    
    if phase['pore.contact_angle'][0]<90:
        pore_mask = ~pore_mask
        throat_mask = ~throat_mask
             
    nwp['pore.occupancy'] = pore_mask
    nwp['throat.occupancy'] = throat_mask
    wp['throat.occupancy'] = ~throat_mask
    wp['pore.occupancy'] = ~pore_mask
    
    sat_p = np.sum(network['pore.volume'][pore_mask])
    sat_t = np.sum(network['throat.volume'][throat_mask])
    sat1 = sat_p + sat_t
    bulk = network['pore.volume'].sum() + network['throat.volume'].sum()
    sat = sat1/bulk
    return sat

def Rate_calc(network, phase, left, outlet, conductance):
    phase.regenerate_models(exclude=['pore.contact_angle','pore.surface_tension','throat.diffusivity','pore.viscosity'])
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


## Apply Trapping
if trapping == 'trapping':
    dr.set_outlet_BC(pores=pn.pores('right'), mode='overwrite')
    dr.apply_trapping()

Snwp_num=80

tmask = np.isfinite(dr['throat.invasion_sequence']) & (dr['throat.invasion_sequence'] > 0)
max_seq = np.max(dr['throat.invasion_sequence'][tmask])
start = np.min(dr['throat.invasion_sequence'][tmask])
relperm_sequence = np.linspace(start,max_seq,Snwp_num).astype(int)
Snwparr = []
relperm_nwp = []
relperm_wp = []

# Loop through invasion sequence to calculate saturation and relative permeability
for i in relperm_sequence:
    co2.regenerate_models(exclude=['pore.contact_angle','pore.surface_tension','throat.diffusivity','pore.viscosity'])
    water.regenerate_models(exclude=['pore.contact_angle','pore.surface_tension','throat.diffusivity','pore.viscosity'])
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

data_dr_trapping = dr.pc_curve()

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot(Snwparr, relperm_nwp, '*-', label='Kr_nwp',color='red')
ax1.plot(Snwparr, relperm_wp, 'o-', label='Kr_wp',color='blue')
ax1.set_xlabel('Snwp')
ax1.set_ylabel('Kr')
ax1.set_xlim(0, 1)
ax1.set_ylim(-0.01, 1.05)
ax1.set_title(f'Relative Permeability in x direction - {trapping.replace("_"," ")}', fontsize=16)
ax1.legend()
fig1.savefig(os.path.join(graph_path, f'DRRelPerm{pn_dim}_{trapping}.png'))

# Making fames of Invasion Percolation saturation field
files = glob.glob(os.path.join(frame_path, '*'))
for f in files:
    os.remove(f)

image_files = []

for collection in ax0.collections:
    collection.remove()
throats_water = pn.throats()
pores_water = pn.pores()

throats_air = np.array([])
pores_air = np.array([])
op.visualization.plot_connections(pn, throats_water, linewidth=linewidth[throats_water], c='b' ,alpha=0.8,ax=ax0)
op.visualization.plot_coordinates(pn, pores_water, markersize=markersize[pores_water], c='r',alpha=0.8,ax=ax0)
ax0.set_title(f'Pressure = {(data_dr_trapping.pc[k]):.2f} Pa',fontsize=16)
fig0.savefig(os.path.join(frame_path,'frame0.png'))
invasion_sequence = np.unique(dr['throat.invasion_sequence'][np.isfinite(dr['throat.invasion_sequence'])])


with Progress() as p:
    t = p.add_task("Generating Video:", total=len(invasion_sequence))
    for sequence in invasion_sequence:
        invasion_pressure = max(dr['throat.invasion_pressure'][dr['throat.invasion_sequence'] == sequence])/1000
        for collection in ax0.collections:
            collection.remove()
            
        k += 1
        inv_throat_pattern = dr['throat.invasion_sequence'] <= sequence
        inv_pore_pattern = dr['pore.invasion_sequence'] <= sequence
        
        new_pores = np.setdiff1d(pn.pores()[inv_pore_pattern], pores_air).astype(int)
        new_throats = np.setdiff1d(pn.throats()[inv_throat_pattern], throats_air).astype(int)
        
        throats_water = np.setdiff1d(throats_water,new_throats).astype(int)
        pores_water = np.setdiff1d(pores_water,new_pores).astype(int)
        
        throats_air = np.union1d(throats_air, new_throats).astype(int)
        pores_air = np.union1d(pores_air, new_pores).astype(int)
        if len(throats_water)>0:    
            op.visualization.plot_connections(pn, throats_water, alpha=0.8, linewidth=linewidth[throats_water], c='b' ,ax=ax0)
        if len(throats_air)>0:
            op.visualization.plot_connections(pn, throats_air, alpha=0.8, linewidth=linewidth[throats_air], c='r' ,ax=ax0)
        if len(pores_water)>0:
            op.visualization.plot_coordinates(pn, pores_water, alpha=0.8, markersize=markersize[pores_water], c='b',ax=ax0)
        if len(pores_air)>0:
            op.visualization.plot_coordinates(pn, pores_air, alpha=0.8, markersize=markersize[pores_air], c='r',ax=ax0)
        
        ax0.set_title(f'Pressure = {invasion_pressure:.2f} kPa',fontsize=16)
        fig0.savefig(os.path.join(frame_path,f'frame{k}.png'))
        image_files.append(os.path.join(frame_path,f'frame{k}.png'))
        p.update(t, advance=1)
    t = p.add_task("Irreducible Water:", total=10)
    if trapping == 'trapping':
        for j in range(0, 10, 1):
            for collection in ax0.collections:
                collection.remove()
                
            k += 1
            if len(throats_water)>0:
                op.visualization.plot_connections(pn, throats_water, alpha=j/10, linewidth=linewidth[throats_water], c='b' ,ax=ax0)
            if len(throats_air)>0:
                op.visualization.plot_connections(pn, throats_air, alpha=1-j/10, linewidth=linewidth[throats_air], c='r' ,ax=ax0)
            
            if len(pores_water)>0:
                op.visualization.plot_coordinates(pn, pores_water, alpha=j/10, markersize=markersize[pores_water], c='b',ax=ax0)
            if len(pores_air)>0:
                op.visualization.plot_coordinates(pn, pores_air, alpha=1-j/10, markersize=markersize[pores_air], c='r',ax=ax0)
            
            fig0.savefig(os.path.join(frame_path,f'frame{k}.png'))
            image_files.append(os.path.join(frame_path,f'frame{k}.png'))
            p.update(t, advance=1)
    if pn_dim == '3D':
        t = p.add_task("Rotação:", total=36)
        for l in range(0, 36, 1):
            ax0.view_init(elev=elev, azim=azim+l*10)
            k += 1
            fig0.savefig(os.path.join(frame_path,f'frame{k}.png'))
            image_files.append(os.path.join(frame_path,f'frame{k}.png'))
            p.update(t, advance=1)

#IMBIBITION

im = op.algorithms.Drainage(network=pn, phase=water)
im.set_inlet_BC(pores=Inlet,mode= 'overwrite')
im.set_outlet_BC(pores=Outlet, mode='overwrite')
im['pore.invaded'] = dr['pore.trapped'].copy()
im['throat.invaded'] = dr['throat.trapped'].copy()

hi = water['throat.entry_pressure'].max()
low = 0.80*water['throat.entry_pressure'].min()
steps = 300
x = np.linspace(0,1,steps)
imb_pressures = low * (hi/low) ** x  

water_ic_pore = im['pore.invaded'].copy()
water_ic_throat = im['throat.invaded'].copy()

im.run(pressures=imb_pressures)

for collection in ax0.collections:
    collection.remove()
k += 1

throats_water = pn.throats()[water_ic_throat]
pores_water = pn.pores()[water_ic_pore]

throats_air = pn.throats()[~water_ic_throat]
pores_air = pn.pores()[~water_ic_pore]

op.visualization.plot_connections(pn, throats_water, alpha=0.3, linewidth=linewidth[throats_water], c='b' ,ax=ax0)
op.visualization.plot_coordinates(pn, pores_water, alpha=0.3, markersize=markersize[pores_water], c='b',ax=ax0)
op.visualization.plot_connections(pn, throats_air, alpha=0.3, linewidth=linewidth[throats_air], c='r' ,ax=ax0)
op.visualization.plot_coordinates(pn, pores_air, alpha=0.3, markersize=markersize[pores_air], c='r',ax=ax0)

fig0.savefig(os.path.join(frame_path,f'frame{k}.png'))
image_files.append(os.path.join(frame_path,f'frame{k}.png'))

for sequence in np.unique(im['throat.invasion_sequence'][np.isfinite(im['throat.invasion_sequence'])]):
    k += 1
    for collection in ax0.collections:
            collection.remove()
    inv_throat_pattern = im['throat.invasion_sequence'] <= sequence
    inv_pore_pattern = im['pore.invasion_sequence'] <= sequence
    
    new_pores = np.setdiff1d(pn.pores()[inv_pore_pattern], pores_water).astype(int)
    new_throats = np.setdiff1d(pn.throats()[inv_throat_pattern], throats_water).astype(int)
    
    throats_water = np.union1d(throats_water,new_throats).astype(int)
    pores_water = np.union1d(pores_water,new_pores).astype(int)
    
    throats_air = np.setdiff1d(throats_air, new_throats).astype(int)
    pores_air = np.setdiff1d(pores_air, new_pores).astype(int)
    
    op.visualization.plot_connections(pn, throats_water, alpha=0.8, linewidth=linewidth[throats_water], c='b' ,ax=ax0)
    op.visualization.plot_connections(pn, throats_air, alpha=0.8, linewidth=linewidth[throats_air], c='r' ,ax=ax0)
    
    op.visualization.plot_coordinates(pn, pores_water, alpha=0.8, markersize=markersize[pores_water], c='b',ax=ax0)
    op.visualization.plot_coordinates(pn, pores_air, alpha=0.8, markersize=markersize[pores_air], c='r',ax=ax0)
    
    fig0.savefig(os.path.join(frame_path,f'frame{k}.png'))
    image_files.append(os.path.join(frame_path,f'frame{k}.png'))
if trapping == 'trapping':
    for j in range(0, 10, 1):
        for collection in ax0.collections:
            collection.remove()
            
        k += 1
        if len(throats_water)>0:
            op.visualization.plot_connections(pn, throats_water, alpha=1-j/10, linewidth=linewidth[throats_water], c='b' ,ax=ax0)
        if len(throats_air)>0:
            op.visualization.plot_connections(pn, throats_air, alpha=j/10, linewidth=linewidth[throats_air], c='r' ,ax=ax0)
        
        if len(pores_water)>0:
            op.visualization.plot_coordinates(pn, pores_water, alpha=1-j/10, markersize=markersize[pores_water], c='b',ax=ax0)
        if len(pores_air)>0:
            op.visualization.plot_coordinates(pn, pores_air, alpha=j/10, markersize=markersize[pores_air], c='r',ax=ax0)
        
        fig0.savefig(os.path.join(frame_path,f'frame{k}.png'))
        image_files.append(os.path.join(frame_path,f'frame{k}.png'))
        
if pn_dim == '3D':
    t = p.add_task("Rotação:", total=36)
    for l in range(0, 36, 1):
        ax0.view_init(elev=elev, azim=azim+l*10)
        k += 1
        fig0.savefig(os.path.join(frame_path,f'frame{k}.png'))
        image_files.append(os.path.join(frame_path,f'frame{k}.png'))

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile(os.path.join(video_path,f'saturation_DRRelPerm{pn_dim}_{trapping}_{npores}.mp4'))


tmask = np.isfinite(im['throat.invasion_sequence']) & (im['throat.invasion_sequence'] > 0)
max_seq = np.max(im['throat.invasion_sequence'][tmask])
start = np.min(im['throat.invasion_sequence'][tmask])
relperm_sequence = np.linspace(start,max_seq,Snwp_num).astype(int)
Snwparr = []
relperm_nwp = []
relperm_wp = []

# Loop through invasion sequence to calculate saturation and relative permeability
for i in relperm_sequence:
    co2.regenerate_models(exclude=['pore.contact_angle','pore.surface_tension','throat.diffusivity','pore.viscosity'])
    water.regenerate_models(exclude=['pore.contact_angle','pore.surface_tension','throat.diffusivity','pore.viscosity'])
    sat = sat_occ_update(network=pn, nwp=co2, wp=water, dr=im, i=i)
    Snwparr.append(sat)
    Rate_abs_nwp = Rate_calc(pn, co2, flow_in, flow_out, conductance = 'throat.hydraulic_conductance')
    Rate_abs_wp = Rate_calc(pn, water, flow_in, flow_out, conductance = 'throat.hydraulic_conductance')
    Rate_enwp = Rate_calc(pn, co2, flow_in, flow_out, conductance = 'throat.conduit_hydraulic_conductance')
    Rate_ewp = Rate_calc(pn, water, flow_in, flow_out, conductance = 'throat.conduit_hydraulic_conductance')
    relperm_nwp.append(Rate_enwp/Rate_abs_nwp)
    relperm_wp.append(Rate_ewp/Rate_abs_wp)

for collection in ax1.lines:
    collection.remove()
    
ax1.plot(Snwparr, relperm_nwp, '*-', label='Kr_nwp',color='red')
ax1.plot(Snwparr, relperm_wp, 'o-', label='Kr_wp',color='blue')
fig1.savefig(os.path.join(graph_path, f'IMRelPerm{pn_dim}_{trapping}.png'))