# Ring Attractor Closed-Loop Demo - Implementation Summary

## Overview
Complete implementation of a dual ring attractor demonstration showcasing CLEF's closed-loop capabilities with interactive stimulus control.

## Files Created

### 1. Backend (`hardware/backends/demo_ring_attractor_backend.py`)
**Classes:**
- `RingAttractorDynamics`: **Dynamical system with bistable limit cycles**
  - Two stable concentric ring attractors at radii r1 and r2
  - Radial dynamics: `dr/dt = -k*(r - r1)*(r - r2) + perturbation`
  - Angular dynamics: `dθ/dt = ω` (constant rotation)
  - Creates smooth orbital motion on stable limit cycles
  - Perturbation pushes system between rings
  - State in Cartesian (x, y) coordinates
- `RingCamera`: Generates 100×100 uint16 images with Gaussian puncta at trajectory position
- `RingStimulus`: Applies radial perturbation to induce ring transitions
- `RingAttractorBackend`: Main hardware backend

**Key Features:**
- **Proper dynamical system:** Bistable potential with two stable limit cycles
- Radial force pulls trajectory toward nearest ring
- Constant angular velocity creates orbital motion
- Perturbation switches between attractors
- Uses Euler integration for continuous dynamics
- State tracked in Cartesian, converted to polar for extraction
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
- **Dynamical system parameters:**
  - r1 = 30px (inner ring radius)
  - r2 = 45px (outer ring radius)
  - k = 5.0 (radial stiffness)
  - ω = 1.0 rad/step (angular velocity)
  - dt = 0.01 (integration time step)
- Dynamics: `dr/dt = -k*(r-r1)*(r-r2) + u`, `dθ/dt = ω`
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

1. **Dynamical system with bistable limit cycles:** Classical dynamical system, not neural network
   - Radial dynamics create two stable attractors at r1 and r2
   - Bistable potential: `-k*(r-r1)*(r-r2)` creates double-well
   - Constant angular velocity creates smooth orbital motion
   - Perturbation pushes system over potential barrier to switch rings
   - Euler integration for continuous dynamics

2. **Manual stimulus as primary interface:** Button-triggered instead of auto-triggered by default
3. **Adjustable intensity:** Slider scales radial perturbation magnitude
4. **Optional auto-trigger:** Checkbox enables theta-based triggering
5. **Ring classification:** Based on proximity to r1 vs r2
6. **Puncta detection:** Centroid of brightest pixels (>99th percentile)
7. **Visualization:** PyQt for responsive GUI, separate from algorithm logic
8. **Cooldown:** 50 frames prevents rapid re-triggering
9. **Trajectory display:** Configurable fading trail shows orbital paths

## Implementation Notes

- Follows Lorenz demo patterns exactly
- **Uses classical dynamical system:** Bistable limit cycle attractor, not neural network
- Radial dynamics: `dr/dt = -k*(r - r1)*(r - r2) + u` creates double-well potential
- Angular dynamics: `dθ/dt = ω` creates constant rotation
- Two stable fixed points in radial direction (r1 and r2)
- Perturbation provides energy to cross potential barrier
- Euler integration: `x(t+dt) = x(t) + dx/dt * dt`
- State in Cartesian (x,y), converted to polar (r,θ) for classification
- Uses `ImageDataInterface` for uint16 camera data
- Stimulus is manual (button) with optional auto mode
- Slider value controls perturbation magnitude
- Ring radii in pixel units for debugging clarity
- All parameters configurable via YAML
- k parameter controls basin of attraction strength
- ω parameter controls orbital speed

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
