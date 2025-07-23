
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json

# =============================================================================
# ConfigParser: Parses JSON config and validates simulation setup
# =============================================================================

class NetworkType(Enum):
    r"""
    Enum for supported network types.
    """
    CUBIC       = "cubic"
    IMPORTED    = "imported"
    
    @classmethod
    def _missing_(cls, value):
        value = value.lower()
        if value in ("cubic", "cube"):
            return cls.CUBIC
        if value in ("imported", "import", "data", "dat"):
            return cls.IMPORTED
        raise ValueError(f"NetworkType: {value}")
    
class PhaseModel(Enum):
    r"""
    Enum for supported phase models.
    """
    WATER   = "water"
    AIR     = "air"
    
    @classmethod
    def _missing_(cls, value):
        value = value.lower()
        if value in ("water"):
            return cls.WATER
        if value in ("air","co2"):
            return cls.AIR
        raise ValueError(f"PhaseModel: {value}")
    
class AlgorithmType(Enum):
    r"""
    Enum for supported algorithm types.
    """
    DRAINAGE    = "drainage"
    STOKES      = "stokes"
    
    @classmethod
    def _missing_(cls, value):
        value = value.lower()
        if value in ("drainage", "imbibition"):
            return cls.DRAINAGE
        if value in ("stokes"):
            return cls.STOKES
        raise ValueError(f"AlgorithmType: {value}")

@dataclass
class NetworkConfig:
    r"""
    Configuration for the pore network.

    Parameters
    ----------
    type : NetworkType
        Type of network (e.g., NetworkType.CUBIC, NetworkType.IMPORTED).
    project_name : str
        Name of the project.
    size : tuple of int, optional
        Network size (required for cubic).
    path : str, optional
        Path to imported network (required for imported).
    prefix : str, optional
        Prefix for imported network files.
    spacing : float, optional
        Spacing between pore centers in each direction (required for cubic).
    seed : int, optional
        Random seed (required for cubic).
    """
    type:           NetworkType
    project_name:   str
    size:           tuple[int, ...] | None = None
    path:           str | None = None
    prefix:         str | None = None
    spacing:        float | None = None
    seed:           int | None = None
    
    def __post_init__(self):
        r"""
        Validates required fields for each network type.
        """
        if self.type is NetworkType.CUBIC:
            missing = [p for p in ("size", "spacing", "seed") if getattr(self, p) is None]
            if missing:
                raise ValueError(f"Cubic Network missing parameters: {', '.join(missing)}")
        elif self.type is NetworkType.IMPORTED:
            missing = [p for p in ("path", "prefix") if getattr(self, p) is None]
            if missing:
                raise ValueError(f"Imported Network missing parameters: {', '.join(missing)}")
        if isinstance(self.size, list):
            self.size = tuple(self.size)

@dataclass
class PhaseConfig:
    r"""
    Configuration for a phase (fluid).

    Parameters
    ----------
    model : PhaseModel
        Phase model (e.g., PhaseModel.WATER, PhaseModel.AIR).
    name : str
        Name of the phase.
    color : str
        Color for visualization.
    properties : dict, optional
        Additional phase properties.
    """
    model:      PhaseModel
    name:       str
    color:      str
    properties: dict | None = None

@dataclass
class AlgorithmConfig:
    r"""
    Configuration for a simulation algorithm.

    Parameters
    ----------
    type : AlgorithmType
        Type of algorithm (e.g., AlgorithmType.DRAINAGE, AlgorithmType.STOKES).
    name : str
        Name of the algorithm.
    phase : str
        Name of the phase to use.
    inlet : tuple of str
        Inlet boundary condition(s).
    outlet : tuple of str, optional
        Outlet boundary condition(s).
    pressures : int, optional
        Number of pressure steps.
    """
    type:               AlgorithmType
    name:               str
    phase:              str
    inlet:              tuple[str, ...]
    outlet:             tuple[str, ...] | None = None
    pressures:          int | None = None
    initial_pressure:   float | None = None
    final_pressure:     float | None = None
    
    def __post_init__(self):
        r"""
        Ensures tuple types for inlets/outlets and integer for pressures.
        """
        if isinstance(self.inlet, str):
            self.inlet = (self.inlet,)
        if isinstance(self.inlet, list):
            self.inlet = tuple(self.inlet)
        if isinstance(self.outlet, str):
            self.outlet = (self.outlet,)
        if isinstance(self.outlet, list):
            self.outlet = tuple(self.outlet)
        if isinstance(self.pressures, float):
            self.pressures = int(self.pressures)
        if self.type is AlgorithmType.DRAINAGE:
            missing = [p for p in ("name", "phase", "inlet") if getattr(self, p) is None]
            if missing:
                raise ValueError(f"Drainage/Imbibition Algorithm missing parameters: {', '.join(missing)}")
        if self.type is AlgorithmType.STOKES:
            missing = [p for p in ("name", "phase", "inlet", "outlet", "pressures", 
                                   "initial_pressure", "final_pressure") if getattr(self, p) is None]
            if missing:
                raise ValueError(f"Stokes Algorithm missing parameters: {', '.join(missing)}")

@dataclass
class ProjectConfig:
    r"""
    Container for the full simulation configuration.
    """
    network:    NetworkConfig
    phases:     tuple[PhaseConfig, ...]
    algorithm:  tuple[AlgorithmConfig, ...]
    
class ConfigParser:
    r"""
    Parses a JSON configuration file and builds validated config objects.
    """
    
    @classmethod
    def from_file(cls, path: str | Path):
        r"""
        Loads and parses a JSON config file.
        """
        with open(path, "r", encoding="utf-8") as f:
            raw_confing = json.load(f)
        return cls._build_config(raw_confing)
    
    @classmethod
    def _build_config(cls, raw: dict):
        r"""
        Builds the full ProjectConfig from the raw JSON dict.
        """
        return ProjectConfig(
            network     = cls._build_network(raw["network"]),
            phases      = cls._build_phases(raw["phases"]),
            algorithm   = cls._build_algorithm(raw["algorithm"])
        )
    
    @classmethod
    def _build_network(cls, network_data: dict):
        r"""
        Builds and validates the network configuration.
        """
        return NetworkConfig(
            type            = NetworkType(network_data.get("type")),
            project_name    = network_data.get("project_name"),
            path            = network_data.get("path"),
            prefix          = network_data.get("prefix"),
            size            = network_data.get("size"),
            spacing         = network_data.get("spacing"), 
            seed            = network_data.get("seed"),
        )
        
    @classmethod
    def _build_phases(cls, phase_data: dict):
        r"""
        Builds the list of PhaseConfig objects from JSON.
        """
        phases = []
        for phase in phase_data:
            phases.append(
                PhaseConfig(
                    model       = PhaseModel(phase.get("model")),
                    name        = phase.get("name"),
                    color       = phase.get("color"),
                    properties  = phase.get("properties")
                )
            )
        return tuple(phases)
    
    @classmethod
    def _build_algorithm(cls, algorithm_data: dict):
        r"""
        Builds the list of AlgorithmConfig objects from JSON.
        """
        algorithms = []
        for algorithm in algorithm_data:
            algorithms.append(
                AlgorithmConfig(
                    type                = AlgorithmType(algorithm.get("type")),
                    name                = algorithm.get("name"),
                    phase               = algorithm.get("phase"),
                    inlet               = algorithm.get("inlet"),
                    outlet              = algorithm.get("outlet"),
                    pressures           = algorithm.get("pressures"),
                    initial_pressure    = algorithm.get("initial_pressure"),
                    final_pressure      = algorithm.get("final_pressure")
                )
            )
        return tuple(algorithms)