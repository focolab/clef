"""
Stimulus controllers module.

Provides factory function to create appropriate stimulus controllers
based on hardware interface type.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def create_stimulus_controller(
    stim_interface: str,
    hardware_manager,
    config: Dict[str, Any]
):
    """
    Factory to create appropriate stimulus controller.
    
    Args:
        stim_interface: Stimulus interface type (e.g., "InvCore-SpinningDisk-639")
        hardware_manager: HardwareManager instance
        config: Configuration dictionary
    
    Returns:
        BaseStimulusController subclass instance
    
    Raises:
        ValueError: If stimulus interface is unknown
    """
    from hardware.stimulus_controllers.dummy_controller import DummyStimulusController
    from hardware.stimulus_controllers.widefield_controller import WidefieldStimulusController
    from hardware.stimulus_controllers.polygon_controller import PolygonStimulusController
    from hardware.stimulus_controllers.demo_lorenz_controller import LorenzStimulusController
    from hardware.stimulus_controllers.input_stimulus_controller import InputStimulusController  # NEW
    
    # Normalize interface string
    interface_lower = stim_interface.lower()
    
    # Dummy/test interfaces
    if interface_lower in ["no stim", "dummy", "test"]:
        logger.info(f"Creating DummyStimulusController for '{stim_interface}'")
        return DummyStimulusController(hardware_manager, config)
    
    # Widefield interfaces
    elif "spinningdisk" in interface_lower or "spinning-disk" in interface_lower or "639" in interface_lower:
        logger.info(f"Creating WidefieldStimulusController for '{stim_interface}'")
        return WidefieldStimulusController(hardware_manager, config)
    
    # Polygon/LDI interfaces
    elif "polygon" in interface_lower or "ldi" in interface_lower or "640" in interface_lower:
        logger.info(f"Creating PolygonStimulusController for '{stim_interface}'")
        return PolygonStimulusController(hardware_manager, config)
    
    # Thunderscope LED
    elif "thunderscope" in interface_lower or "led" in interface_lower:
        # For now, use widefield controller (can be specialized later)
        logger.info(f"Creating WidefieldStimulusController for '{stim_interface}' (LED)")
        return WidefieldStimulusController(hardware_manager, config)
    
    elif "lorenz" in interface_lower:
        logger.info(f"Creating DummyStimulusController for '{stim_interface}'")
        return LorenzStimulusController(hardware_manager, config)
    
    elif "computer_input" in interface_lower or "keyboard" in interface_lower or "input" in interface_lower:
        logger.info(f"Creating InputStimulusController for '{stim_interface}'")
        return InputStimulusController(hardware_manager, config)
    
    else:
        raise ValueError(
            f"Unknown stimulus interface: {stim_interface}. "
            f"Supported: 'no stim', 'dummy', 'InvCore-SpinningDisk-639', "
            f"'InvCore-LDI-Polygon-640', 'InvCore-ThunderscopeLED3', 'lorenz', "
            f"'keyboard', 'input'"
        )


__all__ = [
    'create_stimulus_controller',
    'BaseStimulusController',
    'DummyStimulusController', 
    'WidefieldStimulusController',
    'PolygonStimulusController',
]