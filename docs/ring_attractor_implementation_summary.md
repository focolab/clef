# Ring Attractor Closed-Loop Demo - Implementation Summary

## Overview
Complete implementation of a dual ring attractor demonstration showcasing CLEF's closed-loop capabilities with interactive stimulus control.

## Files Created

### 1. Backend (`hardware/backends/demo_ring_attractor_backend.py`)
**Classes:**
- `RingAttractorDynamics`: Manages state [theta, ring_index] and dynamics
- `RingCamera`: Generates 100×100 uint16 images with Gaussian puncta
- `RingStimulus`: Toggles ring and perturbs theta
- `RingAttractorBackend`: Main hardware backend

**Key Features:**
- Dual concentric rings (configurable radii)
- Constant angular velocity
- Stimulus toggles between rings and adds random theta offset
- Uses `ImageDataInterface` for uint16 data

### 2. Stimulus Controller (`hardware/stimulus_controllers/demo_ring_attractor_controller.py`)
**Class:** `RingAttractorStimulusController`

**Features:**
- Extends `DummyStimulusController`
- Accepts intensity parameter (0-100%)
- Passes intensity to stimulus interface

### 3. Algorithm (`algorithms/demo/ring_attractor_algorithm.py`)
**Classes:**
- `RingAttractorAlgorithm`: Main algorithm with state extraction
- `RingVisualizer`: PyQt-based real-time visualization

**Key Features:**
- Extracts puncta position via centroid of brightest pixels
- Converts pixel coords → (theta, ring_index)
- Tracks transitions between rings
- Interactive GUI with:
  - Manual "Trigger Stimulus" button
  - Intensity slider (0-100%)
  - Auto-stim checkbox for theta ∈ [0, π/4]
  - Real-time trajectory visualization
  - Info panel with frame count, current state, cooldown
- State space visualization:
  - Left: Camera image
  - Right top: 2D trajectory plot with dotted ring circles
  - Right bottom: Controls and info
- Configurable fading trajectory trail

### 4. Configuration Files

#### `config/demo/demo_ring_attractor_hardware.yaml`
- Backend: `ring_attractor_demo`
- Ring parameters: inner_radius (30px), outer_radius (45px)
- Angular velocity: 0.1 rad/step
- Image: 100×100, noise_level=100, puncta_brightness=50000
- Exposure: 10ms

#### `config/demo/demo_ring_attractor_experiment.yaml`
- 2000 samples
- Subject type: "dual_ring_attractor"
- Output: `./demo_output/ring_attractor`

#### `config/demo/demo_ring_attractor_algorithm.yaml`
- Algorithm type: `RingAttractorDemo`
- Real-time visualization: enabled
- Default intensity: 50%
- Cooldown: 50 frames
- Auto-stim: disabled by default, theta ∈ [0, π/4]
- Fading trajectory: 100 samples

### 5. Demo Script (`demo/20251209_demo_ring_attractor.py`)
**Features:**
- Mirrors Lorenz demo structure
- Comprehensive validation of outputs
- Checks for:
  - State timeseries (theta, ring_index)
  - Stimulus events
  - Ring transitions
  - Image TIFF (100×100)
  - Metadata
  - Algorithm plots

## Registry Updates Required

### 1. `algorithms/algorithm_factory.py`
Add import and registration:
```python
try:
    from algorithms.demo import RingAttractorAlgorithm
except ImportError:
    RingAttractorAlgorithm = None

# In _initialize_registry:
try:
    self.register("RingAttractorDemo", RingAttractorAlgorithm)
    self.register("ring_attractor_demo", RingAttractorAlgorithm)
    logger.info("Registered RingAttractorAlgorithm")
except ImportError as e:
    logger.warning(f"Could not import RingAttractorAlgorithm: {e}")
```

### 2. `algorithms/demo/__init__.py`
Add export:
```python
from algorithms.demo.ring_attractor_algorithm import RingAttractorAlgorithm

__all__ = [
    'LorenzDemoAlgorithm',
    'DisplayRGBAlgorithm',
    'RingAttractorAlgorithm',
]
```

### 3. `hardware/stimulus_controllers/__init__.py`
Add import and case:
```python
from hardware.stimulus_controllers.demo_ring_attractor_controller import RingAttractorStimulusController

# In create_stimulus_controller:
elif "ring_attractor" in interface_lower or "ring" in interface_lower:
    logger.info(f"Creating RingAttractorStimulusController for '{stim_interface}'")
    return RingAttractorStimulusController(hardware_manager, config)
```

### 4. `hardware/hardware_manager.py`
Add import and backend selection:
```python
from hardware.backends.demo_ring_attractor_backend import RingAttractorBackend

# In _select_backend:
elif backend_type == "ring_attractor_demo":
    self._backend = RingAttractorBackend(self.config)
    logger.info("Selected RingAttractorBackend")
```

### 5. `config/config_manager.py`
Add new parameters to models:
```python
# In AlgorithmParameters:
default_stim_intensity: int = Field(default=50, ge=0, le=100)
auto_stim_enabled: bool = Field(default=False)
auto_stim_theta_min: float = Field(default=0.0)
auto_stim_theta_max: float = Field(default=0.785)
fading_trajectory_samples: int = Field(default=100, ge=10)

# In HardwareConfig:
ring_params: Dict[str, Any] = Field(default_factory=dict)
```

## Usage

```bash
# Run the demo
python demo/20251209_demo_ring_attractor.py

# Or via CLI
clef-cli --hardware config/demo/demo_ring_attractor_hardware.yaml \
         --experiment config/demo/demo_ring_attractor_experiment.yaml \
         --algorithm config/demo/demo_ring_attractor_algorithm.yaml
```

## Expected Outputs

1. **Real-time visualization:**
   - Camera images with puncta
   - State space plot with dual rings
   - Fading trajectory trail (color-coded by ring)
   - Interactive controls

2. **Saved files:**
   - `<session_id>.tiff`: 100×100 uint16 image stack
   - `<session_id>_metadata.json`: Full metadata including:
     - theta_history, ring_history, frame_indices
     - stim_events with trigger_type (manual/auto)
     - num_transitions
     - State statistics
   - `<session_id>_live_stim_fig.svg`: Four-panel plot:
     - Polar trajectory with ring markers
     - Cartesian trajectory with ring circles
     - Theta time series
     - Ring index time series with transition count

## Key Design Decisions

1. **Manual stimulus as primary interface:** Button-triggered instead of auto-triggered by default
2. **Adjustable intensity:** Slider scales theta perturbation magnitude
3. **Optional auto-trigger:** Checkbox enables theta-based triggering
4. **Ring classification:** Nearest-neighbor to inner/outer ring
5. **Puncta detection:** Centroid of brightest pixels (>99th percentile)
6. **Visualization:** PyQt for responsive GUI, separate from algorithm logic
7. **Cooldown:** 50 frames prevents rapid re-triggering
8. **Trajectory display:** Configurable fading trail for clear visualization

## Implementation Notes

- Follows Lorenz demo patterns exactly
- Uses `ImageDataInterface` for uint16 camera data
- Stimulus is manual (button) with optional auto mode
- Slider value stored in algorithm, passed to controller
- Ring radii in pixel units for debugging clarity
- All parameters configurable via YAML

## Testing Checklist

- [ ] Backend initializes with config parameters
- [ ] Camera generates 100×100 images with puncta
- [ ] Puncta orbits correctly on both rings
- [ ] State extraction converts pixels → (theta, ring)
- [ ] Manual trigger button works
- [ ] Intensity slider updates stimulation
- [ ] Auto-stim checkbox enables/disables triggering
- [ ] Stimulus toggles ring
- [ ] Stimulus perturbs theta
- [ ] Cooldown prevents rapid triggers
- [ ] Visualization updates in real-time
- [ ] Transitions are detected and counted
- [ ] Metadata captures all state and events
- [ ] Plots show trajectories correctly
- [ ] TIFF saves with correct dimensions

## Future Enhancements

- Custom auto-trigger theta ranges (currently hardcoded)
- Multiple trigger zones
- Ring radius adjustment in GUI
- Angular velocity control
- Perturbation direction control (currently random)
- 3D visualization option
