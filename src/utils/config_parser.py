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

class AlgorithmType(Enum):
    DRAINAGE    = "drainage"
    IMBIBITION  = "imbibition"

@dataclass
class NetworkConfig:
    type:       str
    size:       tuple | None = None
    path:       str | None = None
    prefix:     str | None = None
    spacing:    float | None = None
    seed:       int | None = None

@dataclass
class PhaseConfig:
    model:      str
    name:       str
    color:      str
    properties: dict | None = None

@dataclass
class AlgorithmConfig:
    type:   AlgorithmType

@dataclass
class ProjectConfig:
    network:    NetworkConfig
    phases:     tuple[PhaseConfig, ...]
    # algorithm:    AlgorithmConfig
    
class ConfigParser:
    
    @classmethod
    def from_file(cls, path: str | Path):
        with open(path, "r", encoding="utf-8") as f:
            raw_confing = json.load(f)
        return cls._build_config(raw_confing)
    
    @classmethod
    def _build_config(cls, raw: dict):
        
        return ProjectConfig(
            network = cls._build_network(raw["network"]),
            phases  = cls._build_phases(raw["phases"]),
            # algorithm= cls._build_boundaries(raw["algorithm"])
            )
    
    @classmethod
    def _build_network(cls, network_data: dict):

        return NetworkConfig(
            type        = NetworkType(network_data.get("type")),
            path        = network_data.get("path"),
            prefix      = network_data.get("prefix"),
            size        = network_data.get("size"),
            spacing     = network_data.get("spacing"), 
            seed        = network_data.get("seed"),
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