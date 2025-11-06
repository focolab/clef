# Gooey to YAML Configuration Migration Guide

## Overview

This guide maps all parameters from the old `gooey-setup.py` GUI launcher to the new YAML configuration system.

## Complete Parameter Mapping

### Acquisition Controls Group

| Old Gooey Parameter | New YAML Location | Notes |
|---------------------|-------------------|-------|
| `output_folder` | `experiment.yaml` → `output_dir` | Directory path for saving |
| `-d, --total-frames` | `experiment.yaml` → `acquisition.num_frames` | Default: 30000 |
| `-cfg, --mm-configuration-file` | `hardware.yaml` → `mm_config_path` | MM .cfg file path |
| `-z, --zsize` | `experiment.yaml` → `acquisition.z_planes` | Number of Z planes |
| `-mip, --save-mip` | `experiment.yaml` → `save_mip_video` | Boolean flag |
| `-strobe, --strobe-acquisition` | `hardware.yaml` → `strobe_acquisition` | Boolean flag |
| `-sifi, --strobe-inter-frame-interval` | `hardware.yaml` → `strobe_inter_frame_interval_ms` | Default: 80ms |
| `-struct, --save-structural-scan` | `experiment.yaml` → `acquisition.save_structural_scan` | Options: none/GFP+RFP/NeuroPAL |

### Experiment Metadata Group

| Old Gooey Parameter | New YAML Location | Notes |
|---------------------|-------------------|-------|
| `-strn, --subject-strain` | `experiment.yaml` → `subject.genotype` | Renamed for generality |
| `-preprx, --subject-condition` | `experiment.yaml` → `subject.treatment_details.condition` | Treatment condition |
| `-atr, --atr-concentration` | `experiment.yaml` → `subject.treatment_details.atr_concentration_uM` | Float (µM) |
| `-zstp, --z-step-size` | `experiment.yaml` → `z_step_size_um` | Also in `acquisition.z_step` |
| `-nose, --nose-orientation` | `experiment.yaml` → `subject.orientation.nose` | left/right/other |
| `-vnc, --vnc-orientation` | `experiment.yaml` → `subject.orientation.vnc` | up/down/other |
| `-egg, --num-eggs` | `experiment.yaml` → `subject.num_eggs` | Integer count |
| `-app, --microscope-name` | `hardware.yaml` → `microscope_name` | Will be removed in Phase 2 |
| `-notes, --experimental-notes` | `experiment.yaml` → `subject.notes` | Free text |

### Closed-Loop Controls Group

| Old Gooey Parameter | New YAML Location | Notes |
|---------------------|-------------------|-------|
| `-alg, --trigger-algorithm` | `algorithm.yaml` → `algorithm_type` | Algorithm class name |
| `-gui, --GUI-mode` | `algorithm.yaml` → `gui_mode` | neural_imaging/behavior |
| `-bl, --rec-baseline` | `experiment.yaml` → `acquisition.baseline_frames` | Frames before stims allowed |
| `-smp, --save-alg-model-plot` | `algorithm.yaml` → `save_algorithm_plot` | Boolean flag |

### Stimulus Settings Group

| Old Gooey Parameter | New YAML Location | Notes |
|---------------------|-------------------|-------|
| `-stint, --stim-interface` | `hardware.yaml` → `stim_interface` | Stimulus hardware class |
| `-stroi, --use-static-stim-roi` | `hardware.yaml` → `use_static_stim_roi` | Boolean flag |
| `-sfo, --frames-to-stimulate-for-options` | `algorithm.yaml` → `stimulus_params.duration_frames_options` | List of integers |
| `-sio, --stim-intensity-options` | `algorithm.yaml` → `stimulus_params.intensity_percent_options` | List of 0-100% |
| `-sdia, --stimulus-diameter` | `algorithm.yaml` → `algorithm_params.stimulus_diameter_pixels` | For PointAndClick |

### Closed-Loop Algorithm Params (Commented in Original)

| Old Gooey Parameter | New YAML Location | Notes |
|---------------------|-------------------|-------|
| `-stp, --stim-threshold-pos` | `algorithm.yaml` → `algorithm_params.stim_threshold_pos` | Default: 0.06 |
| `-stn, --stim-threshold-neg` | `algorithm.yaml` → `algorithm_params.stim_threshold_neg` | Default: 0.06 |
| `-scd, --stim-cooldown` | `algorithm.yaml` → `algorithm_params.stim_cooldown_frames` | Default: 900 |
| `-ssp, --skip-stimulation-probability` | `algorithm.yaml` → `algorithm_params.skip_stimulation_probability` | Default: 0.1 |
| `-sdp, --delay-stimulation-probability` | `algorithm.yaml` → `algorithm_params.delay_stimulation_probability` | Default: 0.4 |
| `-sdo, --stim-delay-frames-options` | `algorithm.yaml` → `algorithm_params.stim_delay_frames_options` | List e.g. [200, 400] |
| `-userstimlist, --stim-onset-list-options` | `algorithm.yaml` → `algorithm_params.stim_onset_list` | For StimOnsetFromList |

### Development Options Group

| Old Gooey Parameter | New YAML Location | Notes |
|---------------------|-------------------|-------|
| `-i, --input-recording` | `experiment.yaml` → `input_recording_path` | TIFF playback path |
| `-be, --acquisition-backend` | `hardware.yaml` → `backend` | pycromanager/pymmcore/dummy |
| `-nsi, --no-save-images` | `experiment.yaml` → `save_images` | Inverted: true/false |
| `-nsm, --no-save-metadata` | `experiment.yaml` → `save_metadata` | Inverted: true/false |
| `-sgd, --save-gooey-defaults` | N/A | Removed (YAML persists naturally) |
| `-pfwb, --prefill-wb-ops` | `experiment.yaml` → `dev_options.prefill_wb_ops` | Boolean flag |
| `-sms, --send-sms` | `experiment.yaml` → `dev_options.send_sms_on_completion` | Boolean flag |

## Migration Examples

### Example 1: Basic Neural Imaging Experiment

**Old Gooey Command Line (reconstructed):**
```bash
python gooey-setup.py \
  ./data/experiment_001 \
  --total-frames 30000 \
  --mm-configuration-file "C:\MMConfigs\CSUW1-LDI-Polygon.cfg" \
  --subject-strain "OH15500" \
  --atr-concentration 1.0 \
  --z-step-size 1.0 \
  --nose-orientation left \
  --vnc-orientation down \
  --microscope-name "innovation core spinning disk" \
  --experimental-notes "Test experiment" \
  --trigger-algorithm Brainalyzer \
  --stim-interface "InvCore-LDI-Polygon-640"
```

**New YAML Configs:**

`hardware.yaml`:
```yaml
backend: "pycromanager"
mm_config_path: "C:\\MMConfigs\\CSUW1-LDI-Polygon.cfg"
stim_interface: "InvCore-LDI-Polygon-640"
microscope_name: "innovation core spinning disk"
```

`experiment.yaml`:
```yaml
experiment_name: "experiment_001"
output_dir: "./data/experiment_001"
acquisition:
  num_frames: 30000
  z_step: 1.0
subject:
  genotype: "OH15500"
  treatment_details:
    atr_concentration_uM: 1.0
  orientation:
    nose: "left"
    vnc: "down"
  notes: "Test experiment"
z_step_size_um: 1.0
```

`algorithm.yaml`:
```yaml
algorithm_type: "Brainalyzer"
enable_gui: true
```

**New CLI Command:**
```bash
clef-run --hardware hardware.yaml --experiment experiment.yaml --algorithm algorithm.yaml
```

### Example 2: Point-and-Click Manual Stimulation

**Old Gooey Settings:**
- Algorithm: PointAndClick
- Stimulus Interface: InvCore-LDI-Polygon-640
- Stimulus Diameter: 15 pixels
- Frames to stimulate: 48
- Intensity: 20%

**New YAML (`algorithm.yaml`):**
```yaml
algorithm_type: "PointAndClick"
enable_gui: true
gui_mode: "neural_imaging"

algorithm_params:
  stimulus_diameter_pixels: 15

stimulus_params:
  enabled: true
  duration_frames_options: [48]
  intensity_percent_options: [20]
  duration_frames: 48
  intensity_percent: 20
```

### Example 3: Fixed Timing Stimulation

**Old Gooey Settings:**
- Algorithm: StimOnsetFromList
- Stim onset list: "1000, 2000, 3000, 4000"

**New YAML (`algorithm.yaml`):**
```yaml
algorithm_type: "StimOnsetFromList"
enable_gui: false

algorithm_params:
  stim_onset_list: [1000, 2000, 3000, 4000]

stimulus_params:
  enabled: true
  duration_frames: 48
  intensity_percent: 15
```

## Key Differences from Gooey

### 1. Configuration Persistence
- **Old:** Gooey saved to `gooey_config_reload.json` only when `--save-gooey-defaults` flag used
- **New:** YAML configs naturally persist, can be version controlled

### 2. Configuration Validation
- **Old:** No validation until runtime in ClosedLoopEngine
- **New:** Pydantic validates on load with helpful error messages

### 3. Configuration Organization
- **Old:** All parameters in flat namespace
- **New:** Organized into hardware/experiment/algorithm with nested structure

### 4. Hardware Abstraction
- **Old:** `microscope_name` used for if/elif conditionals in code
- **New:** `microscope_name` temporary field, will be removed when hardware abstraction complete

### 5. Terminology Changes
- `subject-strain` → `subject.genotype` (more general)
- `experimental-notes` → `subject.notes` (clearer location)
- `no-save-*` flags → positive `save_*` booleans (easier to understand)

## Backward Compatibility Notes

During Phase 1-2 of refactoring:
- ConfigManager can load YAML and convert to old `args` dict format
- Gooey GUI can write YAML instead of JSON
- Existing code still receives familiar data structure

Full migration complete by Phase 3 when:
- All code uses Config objects instead of `args` dict
- `microscope_name` removed from hardware.yaml
- HardwareManager fully abstracts backend selection

## Converting Old JSON Configs to YAML

If you have saved `gooey_config_reload.json` files:

```python
import json
import yaml

# Load old JSON
with open('gooey_config_reload.json', 'r') as f:
    old_config = json.load(f)

# Extract values from Gooey format
# (Gooey stores in nested structure under 'widgets')
# Manual extraction required - see example scripts

# Write to new YAML
with open('experiment.yaml', 'w') as f:
    yaml.dump(new_experiment_config, f)
```

A conversion script will be provided to automate this process.

## Troubleshooting

**Problem:** "microscope_name not found in hardware.yaml"
- **Solution:** Add `microscope_name` field temporarily (Phase 1-2 only)

**Problem:** Gooey parameter not in new configs
- **Solution:** Check this mapping guide - parameter may be renamed or moved

**Problem:** Algorithm doesn't recognize new parameter names
- **Solution:** Legacy algorithms still use old names internally; update in Phase 2

## Next Steps

1. **Review** your typical Gooey configurations
2. **Create** YAML configs using examples above
3. **Test** with `clef-run --validate-config` before running
4. **Migrate** gradually - new configs work alongside old Gooey during transition