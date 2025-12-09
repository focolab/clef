"""
Input stimulus controller using pynput for keyboard/mouse events.

Provides stimulus delivery through keyboard presses and mouse events.
"""

import logging
import time
from typing import Dict, Any, List, Tuple, Optional

from hardware.stimulus_controllers.base_controller import BaseStimulusController

logger = logging.getLogger(__name__)


class InputStimulusController(BaseStimulusController):
    """
    Stimulus controller for keyboard and mouse input events.
    
    Uses pynput to deliver keyboard presses and mouse clicks/movements
    as stimulus events during closed-loop experiments.
    """
    
    def __init__(self, hardware_manager, config: Dict[str, Any]):
        """
        Initialize input stimulus controller.
        
        Args:
            hardware_manager: HardwareManager instance
            config: Configuration dictionary
        """
        super().__init__(hardware_manager, config)
        
        # Import pynput
        try:
            from pynput.keyboard import Controller as KeyboardController, Key
            from pynput.mouse import Controller as MouseController, Button
            
            self.keyboard = KeyboardController()
            self.mouse = MouseController()
            self.Key = Key
            self.Button = Button
            
            logger.info("pynput controllers initialized")
            
        except ImportError:
            logger.error("pynput not available - input stimulus disabled")
            self.keyboard = None
            self.mouse = None
        
        # Parse config for default input sequences
        gooey_args = config.get("gooey_args", {})
        
        # Default keyboard sequence (can be overridden per stimulus)
        self.default_keys = self._parse_key_sequence(
            gooey_args.get("default_key_sequence", ["space"])
        )
        
        # Default mouse action
        self.default_mouse_action = gooey_args.get("default_mouse_action", "click")
        self.default_mouse_button = gooey_args.get("default_mouse_button", "left")
        
        # Input event history
        self.input_events = []
        
        logger.info(
            f"InputStimulusController initialized with defaults: "
            f"keys={self.default_keys}, mouse={self.default_mouse_action}"
        )
    
    def _parse_key_sequence(self, key_list: List[str]) -> List:
        """
        Parse key names to pynput Key objects.
        
        Args:
            key_list: List of key names (e.g., ["space", "enter", "a"])
            
        Returns:
            List of Key objects or characters
        """
        if not self.keyboard:
            return []
        
        parsed_keys = []
        for key_name in key_list:
            key_name = key_name.lower()
            
            # Special keys
            if hasattr(self.Key, key_name):
                parsed_keys.append(getattr(self.Key, key_name))
            # Single character keys
            elif len(key_name) == 1:
                parsed_keys.append(key_name)
            else:
                logger.warning(f"Unknown key: {key_name}")
        
        return parsed_keys
    
    def submit_stim_params(self, stim_params: Dict[str, Any], image_ndx: int) -> None:
        """
        Process stimulus parameters and schedule input events.
        
        Args:
            stim_params: Dictionary containing stimulus parameters:
                - keys: list of key names to press (optional)
                - mouse_action: "click", "double_click", "move" (optional)
                - mouse_button: "left", "right", "middle" (optional)
                - mouse_position: (x, y) for move action (optional)
                - stim_on: frame to activate
                - stim_off: frame to deactivate
            image_ndx: Current image index
        """
        if not stim_params or not self.keyboard:
            return
        
        logger.debug(
            f"InputStimulusController: Received params at frame {image_ndx}: {stim_params}"
        )
        
        # Track parameters
        if stim_params.get("stim_on") is not None:
            self.stim_param_list.append(stim_params)
            self.stim_on_list.append(stim_params["stim_on"])
            
            # Track off time if pulsed
            if stim_params.get("stim_off") is not None:
                self.stim_off_list.append(stim_params["stim_off"])
    
    def _activate_hardware(self, intensity: float) -> None:
        """
        Activate input stimulus (deliver keyboard/mouse events).
        
        Args:
            intensity: Not used for input events
        """
        if not self.keyboard:
            logger.warning("pynput not available, skipping input stimulus")
            return
        
        # Get current stimulus params
        if not self.stim_param_list:
            logger.warning("No stimulus params available")
            return
        
        current_params = self.stim_param_list[-1]
        
        # Execute keyboard events
        keys = current_params.get('keys', self.default_keys)
        if keys:
            self._execute_keyboard_sequence(keys)
        
        # Execute mouse events
        mouse_action = current_params.get('mouse_action', self.default_mouse_action)
        if mouse_action:
            self._execute_mouse_action(current_params)
        
        logger.info(f"InputStimulusController: Activated stimulus (keys={keys}, mouse={mouse_action})")
    
    def _deactivate_hardware(self) -> None:
        """
        Deactivate input stimulus.
        
        For discrete input events, deactivation is a no-op.
        """
        logger.debug("InputStimulusController: Deactivated stimulus")
    
    def _execute_keyboard_sequence(self, keys: List) -> None:
        """
        Execute keyboard key sequence.
        
        Args:
            keys: List of keys to press (Key objects or characters)
        """
        if isinstance(keys, str):
            keys = self._parse_key_sequence([keys])
        elif isinstance(keys, list) and len(keys) > 0 and isinstance(keys[0], str):
            keys = self._parse_key_sequence(keys)
        
        for key in keys:
            try:
                self.keyboard.press(key)
                time.sleep(0.01)  # Small delay between press/release
                self.keyboard.release(key)
                
                # Record event
                self.input_events.append({
                    'type': 'keyboard',
                    'key': str(key),
                    'time': time.time(),
                })
                
                logger.debug(f"Pressed key: {key}")
                
            except Exception as e:
                logger.error(f"Error pressing key {key}: {e}")
    
    def _execute_mouse_action(self, params: Dict[str, Any]) -> None:
        """
        Execute mouse action.
        
        Args:
            params: Dictionary with mouse parameters:
                - mouse_action: "click", "double_click", "move"
                - mouse_button: "left", "right", "middle"
                - mouse_position: (x, y) for move action
        """
        action = params.get('mouse_action', self.default_mouse_action)
        button_name = params.get('mouse_button', self.default_mouse_button)
        
        # Parse button
        button = self.Button.left
        if button_name == 'right':
            button = self.Button.right
        elif button_name == 'middle':
            button = self.Button.middle
        
        try:
            if action == 'click':
                self.mouse.click(button, 1)
                logger.debug(f"Mouse {button_name} click")
                
            elif action == 'double_click':
                self.mouse.click(button, 2)
                logger.debug(f"Mouse {button_name} double-click")
                
            elif action == 'move':
                position = params.get('mouse_position')
                if position:
                    self.mouse.position = position
                    logger.debug(f"Mouse moved to {position}")
            
            # Record event
            self.input_events.append({
                'type': 'mouse',
                'action': action,
                'button': button_name,
                'time': time.time(),
            })
            
        except Exception as e:
            logger.error(f"Error executing mouse action {action}: {e}")
    
    def get_metadata(self, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get stimulus metadata including input events.
        
        Args:
            args: Optional arguments
        
        Returns:
            Dictionary containing stimulus metadata
        """
        metadata = super().get_metadata(args)
        
        # Add input-specific metadata
        metadata.update({
            'input_events': self.input_events,
            'num_input_events': len(self.input_events),
            'default_keys': [str(k) for k in self.default_keys],
            'default_mouse_action': self.default_mouse_action,
        })
        
        return metadata
    
    def close(self) -> None:
        """Clean up resources."""
        logger.info(f"InputStimulusController closing. Delivered {len(self.input_events)} input events")
