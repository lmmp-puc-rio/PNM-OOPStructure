from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

class NetworkType(Enum):
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

@dataclass
class NetworkConfig:
    type:           NetworkType
    project_name:   str
    size:           tuple[int, ...] | None = None
    path:           str | None = None
    prefix:         str | None = None
    spacing:        float | None = None
    seed:           int | None = None
    
    def __post_init__(self):
        if self.type is NetworkType.CUBIC:
            missing = [p for p in ("size", "spacing", "seed")
                       if getattr(self, p) is None]
            if missing:
                raise ValueError(f"Cubic Network missing parameters: {', '.join(missing)}")
            
        elif self.type is NetworkType.IMPORTED:
            missing = [p for p in ("path", "prefix")
                       if getattr(self, p) is None]
            if missing:
                raise ValueError(f"Imported Network missing parameters: {', '.join(missing)}")
            
        if isinstance(self.size, list):
            self.size = tuple(self.size)

@dataclass
class PhaseConfig:
    model:      PhaseModel
    name:       str
    color:      str
    properties: dict | None = None

@dataclass
class AlgorithmConfig:
    name:           str
    phase:          str
    inlet:          tuple[str, ...]
    outlet:         tuple[str, ...] | None = None
    pressures:      int | None = None
    
    def __post_init__(self):
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

@dataclass
class ProjectConfig:
    network:    NetworkConfig
    phases:     tuple[PhaseConfig, ...]
    algorithm:  tuple[AlgorithmConfig, ...]
    
class ConfigParser:
    
    @classmethod
    def from_file(cls, path: str | Path):
        with open(path, "r", encoding="utf-8") as f:
            raw_confing = json.load(f)
        return cls._build_config(raw_confing)
    
    @classmethod
    def _build_config(cls, raw: dict):
        
        return ProjectConfig(
            network     = cls._build_network(raw["network"]),
            phases      = cls._build_phases(raw["phases"]),
            algorithm   = cls._build_algorithm(raw["algorithm"])
            )
    
    @classmethod
    def _build_network(cls, network_data: dict):

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
        algorithms = []
        for algorithm in algorithm_data:
            algorithms.append(
                AlgorithmConfig(
                    name            = algorithm.get("name"),
                    phase           = algorithm.get("phase"),
                    inlet           = algorithm.get("inlet"),
                    outlet          = algorithm.get("outlet"),
                    pressures       = algorithm.get("pressures"),
                )
            )
        return tuple(algorithms)