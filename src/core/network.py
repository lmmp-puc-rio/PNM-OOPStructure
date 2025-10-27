from utils.config_parser import NetworkType
import numpy as np
import openpnm as op
import skimage.io as sc
import porespy as ps
import matplotlib.pyplot as plt
import os

class Network:
    r"""
    Network(config)

    Creates and manages a pore network for OpenPNM simulations.

    Parameters
    ----------
    config : ProjectConfig
        Parsed configuration object from ConfigParser.

    Attributes
    ----------
    network : openpnm.network.Network
        The OpenPNM network object.
    project_name : str
        Name of the project.
    dim : str
        Dimensionality of the network ('2D' or '3D').
    """
    def __init__(self, config):
        self.config = config.network
        if self.config.type == NetworkType.TOMOGRAPHIC:
            self.pnextract_config = config.pnextract_config
        self.project_name = getattr(self.config, 'project_name', 'project')
        np.random.seed(self.config.seed)
        self.network = self._create_network()
        self.dim = '2D'
        # Determine dimensionality based on pore coordinates
        if len(np.unique(self.network['pore.coords'].T[2])) > 1:
            self.dim = '3D'
        self._setup_boundary_conditions()
        self._setup_domain_properties()
        self._clean_disconnected_pores()
        if self.config.cross_sec == "circles_and_triangles":
            self._throat_geometry_and_triangle_angles()
        self.add_hydraulic_conductance_model()
        self.network.regenerate_models()

    def _create_network(self):
        r"""
        Creates the network based on the configuration type.
        Returns
        -------
        network : openpnm.network.Network
            The generated OpenPNM network object.
        """
        if self.config.type == NetworkType.CUBIC:
            return self._create_cubic()
        elif self.config.type == NetworkType.IMPORTED:
            return self._create_imported()
        elif self.config.type == NetworkType.IMAGE:
            return self._create_image()
        elif self.config.type == NetworkType.TOMOGRAPHIC:
            self._update_mhd_file()
            self._generate_network_to_be_imported()
            return self._create_imported()
        else:
            raise ValueError(f"NetworkType: {self.config.type}")

    def _create_cubic(self):
        r"""
        Creates a cubic network using OpenPNM's Cubic generator.
        Returns
        -------
        pn : openpnm.network.Cubic
            The generated cubic network.
        """
        pn = op.network.Cubic(shape=self.config.size, spacing=self.config.spacing)
        pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
        pn.regenerate_models()
        return pn

    def _create_imported(self):
        r"""
        Imports a network from Statoil format and applies geometry models.
        Returns
        -------
        pn : openpnm.network.GenericNetwork
            The imported and processed network.
        """
        project = op.io.network_from_statoil(path=self.config.path, prefix=self.config.prefix)
        pn = project.network
        pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
        pn.add_model(propname='pore.cluster_number', model=op.models.network.cluster_number)
        pn.add_model(propname='pore.cluster_size', model=op.models.network.cluster_size)
        pn.regenerate_models()
        return pn
        
    def _create_image(self):
        r"""
        Creates a network from an image file.
        """
        im = sc.imread(self.config.file, as_gray=True)
        
        snow_params = {}
        for param_name in ['voxel_size', 'r_max', 'accuracy', 'sigma','boundary_width']:
            param_value = self.config.properties.get(param_name)
            if param_value is not None:
                snow_params[param_name] = param_value
        
        snow= ps.networks.snow2(im.T, **snow_params)
        pn = op.io.network_from_porespy(snow.network)
        h = op.utils.check_network_health(pn)
        op.topotools.trim(network=pn, pores=h['disconnected_pores'])
        pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
        pn['pore.diameter'] = pn["pore.inscribed_diameter"]
        pn['throat.diameter'] = pn["throat.inscribed_diameter"]
        pn['pore.inlets'] = pn["pore.xmin"]
        pn['pore.outlets'] = pn["pore.xmax"]
        pn['throat.spacing'] = pn['throat.total_length']
        pn.regenerate_models()
        return pn
    
    def _generate_network_to_be_imported(self):
        r"""
        Uses pnextract to generate a Statoil format pore network. https://github.com/ImperialCollegeLondon/pnextract
        
        The pnextract binary executable file is in the utils folder 
        -------
        """
        os.system("src/utils/pnextract " + os.path.join(self.config.path, "Image.mhd")) 
        os.system("mv Image_* " + self.config.path)  


    def _update_mhd_file(self):
        lines = []
        filepath = os.path.join(self.config.path, "Image.mhd")
        with open(filepath, "r") as f:
            for line in f:
                if line.startswith("DimSize"):
                    line = f"DimSize =\t{self.pnextract_config.N}\t{self.pnextract_config.N}\t{self.pnextract_config.N}\n"
                elif line.startswith("ElementSize"):
                    line = f"ElementSize =\t{self.pnextract_config.ElementSize}\t{self.pnextract_config.ElementSize}\t{self.pnextract_config.ElementSize}\n"
                elif line.startswith("Offset"):
                    line = f"Offset =\t{self.pnextract_config.Offset}\t{self.pnextract_config.Offset}\t{self.pnextract_config.Offset}\n"
                lines.append(line)

        with open(filepath, "w") as f:
            f.writelines(lines)

 
    def _setup_boundary_conditions(self):
        r"""
        Set up inlet and outlet boundary condition labels on the network.
        
        This method sets up 'pore.inlet' and 'pore.outlet' labels based on the
        configuration and removes any throats connecting inlet-inlet or outlet-outlet
        pores to maintain proper boundary conditions.
        """
        pn = self.network
        inlet_label = self.config.inlet
        outlet_label = None
        if self.config.outlet is not None:
            outlet_label = self.config.outlet

        inlet_pores = pn.pores(inlet_label)
        pn['pore.inlet'] = np.isin(pn.Ps, inlet_pores)
        
        if outlet_label is not None:
            outlet_pores = pn.pores(outlet_label)
            pn['pore.outlet'] = np.isin(pn.Ps, outlet_pores)
        
    def _setup_domain_properties(self):
        r"""
        Calculate domain area and length as network attributes.
        """
        pn = self.network
        inlet_pores = pn.pores('inlet')
        outlet_pores = pn.pores('outlet')
        
        if self.dim == '3D':
            self.domain_area = op.topotools.get_domain_area(pn, inlets=inlet_pores, outlets=outlet_pores)
            self.domain_length = op.topotools.get_domain_length(pn, inlets=inlet_pores, outlets=outlet_pores)
        else:
            coords = pn['pore.coords']
            inlet_coords = coords[inlet_pores]
            # Calculate cross-sectional area (y-range for 2D, assuming z-thickness = 1)
            self.domain_area = np.max(inlet_coords[:, 1]) - np.min(inlet_coords[:, 1])
        
            # Calculate domain length
            inlet_x = np.mean(inlet_coords[:, 0])
            outlet_x = np.mean(coords[outlet_pores][:, 0])
            self.domain_length = abs(outlet_x - inlet_x)
        
    def _clean_disconnected_pores(self):
        r"""
        Removes disconnected pores from the network and regenerates models.
        """
        pn = self.network
        h = op.utils.check_network_health(pn)
        op.topotools.trim(network=pn, pores=h['disconnected_pores'])
        pn.regenerate_models()

    def _throat_geometry_and_triangle_angles(self):
        r"""
        Decides weather the throat will follow a circular cross section or a triangular one.
        I am ignoring square section throats. Sq throats should be considered when G>=sqrt(3)/36 and G<0.079
        We're assigning G=sqrt(3)/36 to G:sqrt(3)/36<G<0.079  so we can treat them as triangle shaped

        https://doi.org/10.1029/2003WR002627   Valvatne and Blunt 2004
        https://doi.org/10.1103/PhysRevE.96.013312   Raeni et al. 2017
        """
        pn = self.network
        G = pn["throat.shape_factor"]
        circular_throats = G > 0.079
        triangular_throats = G <  np.sqrt(3)/36
        sq_throats = ~(circular_throats | triangular_throats)

        G[sq_throats] = np.sqrt(3)/36 * .99
        pn["throat.shape_factor"] = G
        triangular_throats = G <=  np.sqrt(3)/36

        num_throats = pn["throat.shape_factor"].shape[0]
        num_triangular = np.where(triangular_throats)[0].shape[0]

        pn["throat.corner_angles"] = np.zeros((num_throats, 3))
        b2_min = np.arctan( (2/np.sqrt(3)) * np.cos( np.arccos(-12*np.sqrt(3)*G[triangular_throats])/3 + (4*np.pi)/3 ) )
        b2_max = np.arctan( (2/np.sqrt(3)) * np.cos( np.arccos(-12*np.sqrt(3)*G[triangular_throats])/3 ) )
        b2 = np.random.rand(num_triangular) * (b2_max - b2_min) + b2_min

        b1 = -1/2*b2 + 1/2*np.arcsin( ( np.tan(b2) + 4*G[triangular_throats] )/( np.tan(b2) - 4*G[triangular_throats] )*np.sin(b2) )
        b3 = np.pi/2 - b1 - b2

        pn["throat.corner_angles"][triangular_throats, 0] = b1
        pn["throat.corner_angles"][triangular_throats, 1] = b2
        pn["throat.corner_angles"][triangular_throats, 2] = b3


    def set_inlet_outlet_pores(self, inlet_pores=None, outlet_pores=None):
        r"""
        Set inlet and outlet pores using specific pore numbers.
        
        This method allows direct specification of which pores should be
        designated as inlets and outlets using their indices. You can set
        only inlets, only outlets, or both.
        
        Parameters
        ----------
        inlet_pores : array_like, optional
            List or array of pore indices to use as inlets. If None, inlets are not modified.
        outlet_pores : array_like, optional
            List or array of pore indices to use as outlets. If None, outlets are not modified.
        """        
        pn = self.network
        
        if inlet_pores is not None:
            inlet_pores = np.asarray(inlet_pores)
            pn['pore.inlets'] = False
            pn['pore.inlets'][inlet_pores] = True
        
        if outlet_pores is not None:
            outlet_pores = np.asarray(outlet_pores)
            pn['pore.outlets'] = False
            pn['pore.outlets'][outlet_pores] = True
            
    def get_inlet_outlet_info(self):
        r"""
        Get information about current inlet and outlet pores.
        
        Returns
        -------
        info : dict
            Dictionary containing inlet/outlet information
        """
        pn = self.network
        
        inlet_pores = pn.pores('inlets')
        outlet_pores = pn.pores('outlets')
        
        info = {
            'num_inlets': len(inlet_pores),
            'num_outlets': len(outlet_pores),
            'inlet_pores': inlet_pores,
            'outlet_pores': outlet_pores,
            'total_pores': pn.Np,
            'inlet_fraction': len(inlet_pores) / pn.Np,
            'outlet_fraction': len(outlet_pores) / pn.Np
        }
            
        return info
    
    def redefine_throat_radius(self, mean=1.0, mode='norm', plot_hist=False):
        r"""
        Redefine throat radius.

        Arguments:
        ----------
        mean : float
            It is the value for the mean value of the adjusted throat radius distribution.

        mode: str
            Method applied to adjust the distribution
            'norm': normalizes the distribution by its mean value and then multiply it for the value set to be the new mean.

        plot_hist: bool
            If True plots the throat radius distribution histogram.    

        """
        R = self.network['throat.diameter']/2
        if mode == 'norm':
            R = R/np.mean(R)*(mean)
            self.network['throat.diameter'] = 2*R

        if plot_hist:
            plt.hist(R/(10**-6))
            plt.title(r"after adjustment")
            plt.xlabel(r"radius [$\mu$ m]")
            plt.ylabel(r"frequency")
            plt.savefig("hist_throat_radius.png")
            plt.close()
    
    def calculate_permeability(self):
        r"""
        Calculate permeability.
        
        Returns
        -------
        K : float
            permeability in m²
        """
        pn = self.network
        R = pn['throat.diameter']/2
        L = pn['throat.length']
        reference_phase = op.phase.Phase(network=pn)
        reference_phase.add_model_collection(op.models.collections.physics.basic)
        reference_phase['pore.viscosity'] = 1.0
        
        inlet_pores = pn.pores('inlet')
        outlet_pores = pn.pores('outlet')

        flow = op.algorithms.StokesFlow(network=pn, phase=reference_phase)
        flow.set_value_BC(pores=inlet_pores, values=1)
        flow.set_value_BC(pores=outlet_pores, values=0)
        flow.run()
        
        # Calculate permeability: K = Q * L * μ / (A * ΔP)
        # With μ = 1 and ΔP = 1, this simplifies to K = Q * L / A
        Q = flow.rate(pores=inlet_pores, mode='group')[0]
        
        # Use pre-calculated domain properties
        K = Q * self.domain_length / self.domain_area
        print(f'K = {K/10**-12/10**-3} mD')
        
        return K

    def add_hydraulic_conductance_model(self):
        r"""
        Add a simple Hagen-Poiseuille model for 'throat.hydraulic_conductance' on the network.
        """
        pn = self.network

        def _hp_conductance(prop, length):
            D = prop
            L = pn[length]
            return np.pi*(D/2.0)**4/(8.0*L)

        pn.add_model(
            propname='throat.hydraulic_conductance',
            model=op.models.misc.generic_function,
            func=_hp_conductance,
            prop='throat.diameter',
            length='throat.length',
            regen_mode='deferred'
        )

    def calculate_porosity(self):
        r"""
        Calculate porosity of the pore network.
        
        The porosity is calculated as the ratio of void volume (pores + throats)
        to the total bulk volume of the network domain.
        
        Returns
        -------
        porosity : float
            Network porosity (dimensionless)
        """
        pn = self.network
        Vol_void = np.sum(pn['pore.volume']) + np.sum(pn['throat.volume'])
        
        # Use pre-calculated domain properties
        Vol_bulk = self.domain_area * self.domain_length
        porosity = Vol_void / Vol_bulk
        return porosity
    
