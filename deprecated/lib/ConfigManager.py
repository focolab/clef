from dataclasses import dataclass
import yaml




@dataclass
class HardwareConfig:
    # fields matching hardware.yaml structure
    # Pydantic models for configs
    # YAML loading with validation
    # Compatibility checks (stimulus in hardware? etc.)
    pass

@dataclass
class ExperimentConfig:
    pass

@dataclass
class AlgorithmConfig:
    pass


class ConfigManager:
    def load_hardware(path) -> HardwareConfig:
        pass
    def load_experiment(path) -> ExperimentConfig:
        pass
    def load_algorithm(path) -> AlgorithmConfig:
        pass