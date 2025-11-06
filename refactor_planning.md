# Refactor Planning Document

## Project: Closed-loop experimental framework (CLEF)
**Document Version:** 1.0  
**Created:** 2025-10-24
**Last Updated:** 2025-10-24
**Owner:** Raymond Dunn

---

## 1. Executive Summary

### What We're Refactoring
This repo contains code for a platform enabling closed-loop microscopy experimental design. This code runs a core data acquisition loop, receiving input data from, for example, a camera. This platform allows for this real-time data to be processed and, based on that processing, coordinates various microscopy components to carry out experimental perturbations determined by some logic. Finally the platform saves the data and all pertinent information about expeirmental procedure and apparatus for later analysis and auditing.   

### Why We're Refactoring
The primary goal for this refactor is to prepare this platform, clef 1.0, to be submited to the Journal of Open Source Software. This means we need to have a distributable, remove specific references to our hardware and put those into configuration files, and remove worm/neuron specific references. Finally, we need to produce a demo with fake actuators/components which can be validated by reviewers. 

### Key Goals
1. Remove specific references to our hardware, and place them in configuration files.
2. Remove specific references to our experiments, such as neurons/worms. 
3. Develop a test suite that can enables validation by an outside authority.
4. Make the clef app distributable/installable.

## Secondary goals
1. Dummy/non-functional stubs to simulate hardware.
2. Test suite for components which are spec'd out by configuration files.
3. Standardize how platform parameters are passed around system, to different modules.

---

## 2. Current State Analysis

### Codebase Overview
```
Repository: Closed-loop experimental framework (CLEF) v1.0
Purpose: Real-time microscopy data acquisition and closed-loop experimental control platform

- Main modules:
  - ClosedLoopEngine.py: Core acquisition loop orchestrating data capture, processing, and stimulus triggering
  - MMSubroutines.py: Micro-Manager hardware interface for microscope control, configuration, and image acquisition
  - StimBaseClass.py: Abstract base class for stimulus interface implementations
  - Stimulus Interfaces (hardware-specific implementations):
    * InvCoreLDIPolygon.py: SLM-based spatial light modulation for targeted illumination
    * InvCoreSpinningDisk639.py: 639nm laser stimulation for spinning disk microscopy
    * InvCoreThunderscopeLED3.py: LED-based widefield stimulation
    * TorstoscopeBLSPolygon.py, TorstoscopeLMM5.py, TorstoscopeSolenoid.py: Legacy hardware interfaces (need refactoring)
  - Trigger Algorithms:
    * Brainalyzer.py: Real-time neural activity analysis with GUI visualization
    * BrainalyzerWorker.py: Subprocess worker for GUI rendering and user interaction
    * DummyAlg.py: No-op algorithm for testing
    * Legacy algorithms (in algs-need-to-subclass/ folder, need refactoring):
      - DynamicRangeDeriv.py: Blob detection with derivative-based triggering
      - RoiDeriv.py: User-drawn ROI with derivative-based triggering and motion correction
      - StimOnsetFromList.py: Fixed stimulus timing from user-provided list
      - PointAndClick.py: Interactive GUI for manual stimulus targeting with motion tracking
      - HammerOfDawn.py: Real-time cursor-following stimulus (continuous tracking mode)
  - Visualization:
    * QtVisualizer.py: PyQtGraph-based real-time image display
    * XYStageTracker.py: Real-time stage tracking and control for behavior experiments
  - Utilities:
    * wbliveUtils.py: Image processing, coordinate transforms, ROI handling, notifications
    * ImageProcessor.py: Blob detection and image segmentation algorithms
    * DummyMMC.py, DummyStim.py: Mock objects for testing without hardware

- Technology stack:
  - Language: Python 3.x
  - Hardware Interface: Micro-Manager (pymmcore, pycromanager)
  - GUI Framework: PyQt5/PyQtGraph for real-time visualization
  - Image Processing: NumPy, OpenCV, scipy
  - Performance: Numba JIT compilation for critical paths
  - Data I/O: tifffile for microscopy data, JSON for metadata
  - Key dependencies: 
    * pymmcore/pycromanager (microscope control)
    * pyqtgraph (visualization)
    * numpy, opencv-python (image processing)
    * numba (performance optimization)
    * tifffile (TIFF I/O)
    * scipy (signal processing)
    * imageio (video generation)

- Hardware dependencies:
  - Specific microscopes: "torstoscope spinning disk", "innovation core spinning disk", "innovation core thunderscope"
  - Stimulus devices: Mightex Polygon SLM, 89 North LDI, various laser/LED controllers
  - Cameras: Photometrics PRIME BSI, others via Micro-Manager
  - Stage controllers: ASI stages with serial communication

- Entry points and configuration:
  - gooey-setup.py: GUI launcher with Gooey framework for parameter configuration
    * Tabbed interface for acquisition, experiment metadata, closed-loop, stimulus settings
    * Hardcoded hardware-specific choices (microscope names, MM config paths, stimulus interfaces)
    * Experiment-specific metadata fields (strain, ATR concentration, nose/VNC orientation, eggs)
    * Saves/loads configuration via JSON for persistence
  
- Calibration and setup utilities:
  - calibrate_polygon.py: SLM calibration tool for coordinate transformation
    * Three-point calibration between image space and polygon (SLM) space
    * Auto-thresholding and blob detection for finding patterned spots
    * Saves calibrations to JSON with microscope configuration metadata
  - polygon_drawer.py: Interactive ROI drawing tool using Napari
    * Coordinate transformation from image space to polygon space
    * Socket-based communication with external C++ polygon control app
    * Legacy tool for manual mask creation

- Configuration files (referenced but hardware-specific):
  - Micro-Manager .cfg files (hardcoded paths in multiple locations)
  - Polygon calibration JSON (res/peripherals/Mightex Polygon P1000/calibrations.json)
  - Gooey build config JSON (gooey_config_reload.json, gooey_config.json)
```

### Pain Points

1. **Hardware-Specific References Throughout Codebase**
   - Issue: Hardcoded microscope names, device names, and configuration paths scattered across multiple files
   - Impact: Cannot distribute to other labs without extensive code modification; reviewers cannot validate without specific hardware
   - Affected files/modules: 
     * gooey-setup.py (lines with microscope_name choices, MM config paths, stim_interface choices)
     * MMSubroutines.py (scope-specific logic in prepare_live_acquisition, structural_scan_channel)
     * InvCoreLDIPolygon.py, InvCoreSpinningDisk639.py, etc. (device property names)
     * StimBaseClass.py (initialize_stim_interface method with hardcoded mappings)
     * wbliveStimClass.py (legacy file with similar hardcoded device strings)

2. **Experiment-Specific (Worm/Neuron) References**
   - Issue: C. elegans-specific terminology and parameters embedded in code and GUI
   - Impact: Platform appears specialized for worm neuroscience rather than general closed-loop microscopy
   - Affected files/modules:
     * gooey-setup.py ("subject-strain", "VNC orientation", "nose orientation", "num-eggs", "NeuroPAL")
     * MMSubroutines.py ("NeuroPAL" structural scan presets)
     * Variable names throughout (e.g., "wb" prefix likely stands for "worm brain")
     * Legacy algorithm files contain C. elegans assumptions:
       - RoiDeriv.py: Uses structural scans and ROI masks tied to worm anatomy
       - Multiple algorithms reference "structural_scan_dir" for neuronal landmarks

3. **Configuration Management**
   - Issue: Configuration scattered between JSON files, hardcoded strings, and GUI defaults
   - Impact: Difficult to adapt to new hardware setups; no clear separation of hardware config from experimental parameters
   - Affected files/modules:
     * Polygon calibrations in separate JSON with manual editing required
     * MM configuration files referenced by absolute paths
     * Gooey config dump/reload mechanism separate from runtime configuration

4. **Mixed Legacy and Current Code**
   - Issue: Multiple implementations of same functionality (old wbliveStimClass.py vs new StimBaseClass hierarchy; Torstoscope classes marked incomplete); Legacy trigger algorithms in folder "algs-need-to-subclass" don't follow current architecture patterns
   - Impact: Code confusion, maintenance burden, unclear which version is authoritative; algorithms duplicating GUI/visualization code
   - Affected files/modules:
     * wbliveStimClass.py (legacy, 500+ lines duplicating StimBaseClass functionality)
     * TorstoscopeBLSPolygon.py, TorstoscopeLMM5.py, TorstoscopeSolenoid.py (incomplete, print statements saying "Not done!")
     * algs-need-to-subclass/ folder:
       - DynamicRangeDeriv.py: ~350 LOC, uses ImageProcessor for blob detection
       - RoiDeriv.py: ~400 LOC, requires napari GUI for ROI selection, motion correction
       - StimOnsetFromList.py: ~200 LOC, fixed timing algorithm
       - PointAndClick.py: ~900 LOC, full PyQt GUI embedded in algorithm
       - HammerOfDawn.py: ~900 LOC, full PyQt GUI with real-time cursor tracking
     * Common issues in legacy algorithms:
       - Each algorithm reimplements its own visualization/GUI code
       - No common base class or interface (unlike StimBaseClass pattern)
       - Mix algorithm logic with visualization code
       - Inconsistent metadata handling and timing collection

5. **Testing Without Hardware**
   - Issue: Limited demo/testing infrastructure; DummyMMC exists but DummyStim minimal
   - Impact: Reviewers cannot easily run and validate system; development requires full hardware setup
   - Affected files/modules:
     * DummyMMC.py (functional for playback)
     * DummyStim.py (minimal implementation)
     * No integration test suite demonstrating full closed-loop workflow
     * MMConfig_demo.cfg referenced but demo mode not fully functional

6. **Absolute File Paths**
   - Issue: Windows-specific absolute paths hardcoded throughout
   - Impact: Not portable across machines or operating systems
   - Affected files/modules:
     * gooey-setup.py (C:\\ paths for MM configs)
     * calibrate_polygon.py (C:/Users/... paths)
     * polygon_drawer.py (C:/Users/confocal/... paths)
     * MMSubroutines.py (COM port specifications, C:\\ paths)

7. **Algorithm Architecture Inconsistency**
   - Issue: No unified interface or base class for trigger algorithms; each algorithm reimplements visualization, event handling, and metadata collection differently
   - Impact: Difficult to add new algorithms; code duplication; hard to test algorithms in isolation; GUI code mixed with algorithm logic
   - Affected files/modules:
     * No AlgorithmBaseClass equivalent to StimBaseClass
     * PointAndClick.py and HammerOfDawn.py: ~900 LOC each, mostly GUI code
     * Each algorithm has different signatures for initialize_model(), process_frame(), check_stim()
     * Metadata collection inconsistent across algorithms
     * Visualization tightly coupled to algorithm (QtVisualizer classes embedded in algorithm files)
   - Contrast with stimulus interfaces which have clean StimBaseClass abstraction

### Current Metrics Baseline
- **Test Coverage:** 0% (no test suite currently exists)
- **Build Time:** N/A (Python interpreted, no build step)
- **Lines of Code:** ~11,500 LOC (estimated across all modules)
  - Core platform: ~3,500 LOC
  - Stimulus interfaces: ~1,500 LOC
  - Current algorithms (Brainalyzer, Dummy): ~2,000 LOC
  - Legacy algorithms (need refactoring): ~2,750 LOC
  - Utilities and support: ~1,750 LOC
- **Configuration Files:** 
  - 3 types (Micro-Manager .cfg, JSON calibrations, Gooey config)
  - All with absolute paths or hardware-specific content
- **Hardware Dependencies:** 
  - 3 specific microscope systems
  - 7 stimulus interface implementations
  - Multiple device-specific property names
- **Legacy Code:** ~4,250 LOC in files marked incomplete, duplicated, or needing refactoring
  - wbliveStimClass.py: ~500 LOC (duplicates StimBaseClass)
  - Incomplete Torstoscope interfaces: ~1,000 LOC
  - Legacy trigger algorithms: ~2,750 LOC (each with embedded GUI code)

---
## 3. Refactoring Objectives

### Primary Goals (Must Have)
1. **Hardware Abstraction Achieved**
   - All hardware-specific references (device names, COM ports, MM config paths) moved to `hardware.yaml`
   - Zero hardcoded device strings in core Python modules (`MMSubroutines.py`, `ClosedLoopEngine.py`)
   - Measurable: Run automated search for hardcoded strings like "DAC488", "COM6", "C:\\" in core modules → 0 occurrences

2. **Distributable Package Created**
   - CLEF installable via `pip install clef` or similar
   - Entry point script (`clef-run --hardware <path> --experiment <path> --algorithm <path>`)
   - Measurable: Fresh Python environment can install and run minimal test in <5 minutes

3. **Demo Mode Functional**
   - Reviewers can execute full closed-loop workflow without hardware
   - Uses `DummyMMC` and `DummyStim` driven by configs
   - Includes validation test suite (`pytest tests/test_minimal_run.py`)
   - Measurable: CI/CD pipeline runs full demo successfully

### Secondary Goals (Nice to Have)
1. **Remove Experiment-Specific Terminology**
   - Replace worm/neuron references with generic terms in user-facing code
   - Example: "subject" instead of "strain", "orientation" instead of "nose_orientation"
   - These can coexist with domain-specific config fields in `experiment.yaml`

2. **Algorithm Base Class Pattern**
   - Create `AlgorithmBaseClass` similar to `StimBaseClass`
   - Refactor at least one legacy algorithm (e.g., `DynamicRangeDeriv`) to use new pattern
   - Provides template for future algorithm development

3. **Comprehensive Configuration Validation**
   - JSON schema validation for YAML configs
   - Runtime checks for hardware/algorithm compatibility
   - Helpful error messages pointing to config issues

### Non-Goals (Explicitly Out of Scope)
- **Refactoring all legacy algorithms**: Keep `Brainalyzer` + `DummyAlg` working; document others as "legacy-examples"
- **Complete Torstoscope interfaces**: Mark incomplete classes as deprecated, focus on Innovation Core hardware
- **GUI redesign**: Keep existing PyQt/Gooey interfaces, just make them config-driven
- **Performance optimization**: Not changing core acquisition loop unless necessary for configs
- **Cross-platform support**: Focus on Windows (existing target), defer Linux/Mac testing

### Success Metrics
| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| Hardcoded Hardware Refs | ~35 locations | 0 in core modules | `grep -rn "DAC[0-9]\|COM[0-9]\|C:\\\\" MMSubroutines.py ClosedLoopEngine.py StimBaseClass.py` (count matches) |
| Hardcoded Microscope Names | 15+ conditionals | 0 if/else on scope names | `grep -rn "torstoscope\|innovation core" *.py \| wc -l` in core files |
| Test Coverage | 0% (no tests) | 60% (core + demo) | `pytest --cov=clef --cov-report=term` |
| Integration Tests Passing | 0 tests | 3+ test scenarios | `pytest tests/integration/` - minimal, demo, brainalyzer workflows |
| Installability | Manual, ~30 min setup | <5 min fresh install | Time: `pip install clef && clef-run --help` in fresh Python 3.9+ venv |
| Demo Runtime | N/A (no demo exists) | <2 min for 100 frames | `time clef-run --hardware config/test/hardware_minimal.yaml --experiment config/test/experiment_minimal.yaml --algorithm config/test/algorithm_minimal.yaml` |
| Config Files Exist | 0 YAML configs | 6 working templates | Files exist + validate against schema + execute successfully |
| Documentation Pages | Sparse README | 5+ guides | README, install guide, config reference, migration guide, JOSS paper |
| JOSS Submission Ready | No | Yes | Checklist complete: ✓ package ✓ tests ✓ docs ✓ demo ✓ paper draft |
| Lines of Legacy Code | ~4,250 LOC | <1,000 LOC | Measure deprecated/ folder size, track refactored modules |
| Absolute File Paths | ~20 hardcoded paths | 0 (all in configs) | `grep -rn "C:\\\\\|/Users/" *.py` excluding config examples |
| Dependencies on Gooey | Core engine requires it | Optional (CLI alternative) | `python -c "import clef; clef.run()" works without gooey installed` |

---

## 4. Architecture & Design Decisions

### Target Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        CLEF Application                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐      ┌──────────────────────────────────┐   │
│  │ Entry Points │      │     Configuration Layer          │   │
│  ├──────────────┤      ├──────────────────────────────────┤   │
│  │ CLI Runner   │─────>│  ConfigManager (NEW)             │   │
│  │ Gooey GUI    │      │  - YAML loader                   │   │
│  │ (optional)   │      │  - Pydantic validation           │   │
│  └──────────────┘      │  - Schema enforcement            │   │
│                        │  - Merge defaults + user configs │   │
│                        └──────────────┬───────────────────┘   │
│                                       │                        │
│  ┌────────────────────────────────────▼───────────────────┐   │
│  │          ClosedLoopEngine (REFACTORED)                 │   │
│  ├────────────────────────────────────────────────────────┤   │
│  │  - Config-driven orchestration (no hardcoded strings)  │   │
│  │  - Acquisition loop coordination                       │   │
│  │  - Algorithm/Stimulus dispatching                      │   │
│  │  - Metadata collection & saving                        │   │
│  └───┬──────────────────────┬──────────────────────┬──────┘   │
│      │                      │                      │           │
│  ┌───▼──────────┐   ┌───────▼─────────┐   ┌───────▼────────┐ │
│  │ Hardware     │   │ Algorithm        │   │ Stimulus       │ │
│  │ Manager      │   │ Factory (NEW)    │   │ Factory        │ │
│  │ (REFACTORED) │   │                  │   │ (EXISTING)     │ │
│  ├──────────────┤   ├──────────────────┤   ├────────────────┤ │
│  │ Backend      │   │ Creates:         │   │ Creates:       │ │
│  │ Abstraction: │   │ - Brainalyzer    │   │ - LDIPolygon   │ │
│  │ • pycromanager│   │ - DummyAlg       │   │ - SpinningDisk │ │
│  │ • pymmcore   │   │ - (Future algs)  │   │ - Dummy        │ │
│  │ • DummyMMC   │   │                  │   │                │ │
│  └───┬──────────┘   └──────────────────┘   └────────────────┘ │
│      │                                                          │
│  ┌───▼──────────────────────────────────────────────────────┐  │
│  │            Real or Simulated Hardware                    │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │  Micro-Manager Devices | Camera | Stages | Stim Devices │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Key components:
- Component ConfigManager (NEW): Centralized configuration loading, validation, and access. Load YAML files (hardware, experiment, algorithm), Validate against Pydantic schemas, Provide typed configuration objects to other components. New file clef/config/config_manager.py.
- Component HardwareManager (REFACTORED from MMSubroutines.py): Abstract hardware backend selection and device control. Initialize backend (pycromanager/pymmcore/dummy) from hardware.yaml, Map logical device names to physical device properties, Provide unified interface for device interaction. clef/hardware/hardware_manager.py (refactored from MMSubroutines.py)
- Component ClosedLoopEngine (REFACTORED): Orchestrate acquisition loop using configuration objects. Accept Config objects instead of args dict. Dispatch to algorithm and stimulus based on config. Collect metadata from config for saving. No hardcoded device names or conditionals on microscope names. clef/engine/closed_loop_engine.py (refactored from ClosedLoopEngine.py)
- Component AlgorithmFactory (NEW): Instantiate trigger algorithms based on config. Registry of available algorithms (Brainalyzer, DummyAlg, future algorithms), Create algorithm instance with parameters from algorithm.yaml, (Future) enforce AlgorithmBaseClass interface. New file clef/algorithms/algorithm_factory.py
- Component StimulusFactory (EXISTING, enhanced): Instantiate stimulus interfaces based on config. Registry from StimBaseClass.initialize_stim_interface(). Create stimulus instance from hardware.yaml stim_interface field. Pass device mappings from config instead of hardcoded strings. clef/stimulus/stim_base_class.py

Data flow:

Configuration phase:
User → Entry Point → ConfigManager → [hardware.yaml, experiment.yaml, algorithm.yaml]
                                      ↓
                               Pydantic Validation
                                      ↓
                         [HardwareConfig, ExperimentConfig, AlgorithmConfig]

Initialization phase:
ClosedLoopEngine ← Config Objects
           ↓
   HardwareManager.initialize(HardwareConfig)
           ↓
   Backend Selection (pycromanager/pymmcore/dummy)
           ↓
   Load MM Config from path in hardware.yaml
           ↓
   AlgorithmFactory.create(AlgorithmConfig)
           ↓
   StimulusFactory.create(HardwareConfig.stim_interface)

Acquisition Loop phase:
┌─> HardwareManager.snap_image()
   │            ↓
   │   Algorithm.process_frame(image, metadata)
   │            ↓
   │   Algorithm.check_stim() → stimulus_params?
   │            ↓
   │   StimulusInterface.trigger(stimulus_params)
   │            ↓
   │   Save data/metadata (ExperimentConfig.output_dir)
   └────────────┘ (repeat for N frames)

```

### Design Patterns to Implement
1. **Configuration Object Pattern**
   - Where: Throughout codebase, replacing args dict
   - Why: Type safety and IDE autocomplete. Clear schema enforcement.
   - Example: 
    ```
    def run_acquisition(args):
        num_frames = args["num-frames"]  # string key, no validation
        scope_name = args["microscope_name"]  # hardcoded conditional later
    
    # AFTER (refactored):
    def run_acquisition(config: ExperimentConfig):
        num_frames = config.acquisition.num_frames  # typed, validated
        # No scope_name needed - hardware abstracted in HardwareManager
    ```

2. **Adapter Pattern**
   - Where: HardwareManager backend abstraction
   - Why: Uniform interface across pycromanager, pymmcore, DummyMMC, Easy to add new backends (e.g., future hardware control libraries), Reviewers can use DummyMMC without changing any other code
   - Example: 
   ```
   class HardwareBackend(ABC):
      @abstractmethod
      def snap_image(self) -> np.ndarray: ...
      @abstractmethod
      def set_property(self, device: str, prop: str, value: Any): ...
  
  class PycromanagerBackend(HardwareBackend):
      def snap_image(self): return self.bridge.snap_image()
  
  class DummyBackend(HardwareBackend):
      def snap_image(self): return self.image_generator.next_frame()
  
  # HardwareManager selects backend from hardware.yaml:
  backend = BackendFactory.create(config.hardware.backend)
  ```

3. **Factory Pattern**
   - Where: Algorithm and Stimulus instantiation
   - Why: Decouple creation logic from usage, Centralized registry of available algorithms/stimuli, Config-driven selection without if/elif chains
   - Example: 
   ```
   # AlgorithmFactory
  ALGORITHM_REGISTRY = {
      "brainalyzer": Brainalyzer,
      "dummy": DummyAlg,
      # Future: "dynamic_range": DynamicRangeAlgorithm
  }
  
  def create_algorithm(config: AlgorithmConfig):
      alg_class = ALGORITHM_REGISTRY[config.algorithm_type]
      return alg_class(config.algorithm_params)
  
  # In ClosedLoopEngine:
  algorithm = AlgorithmFactory.create(algorithm_config)
  # No hardcoded if algorithm_name == "brainalyzer": ...
  ```

### Key Technical Decisions

#### Decision 1: YAML + Pydantic for Configuration
- **Context:** Need to move all hardware-specific and experiment-specific parameters out of code. Must be human-readable, validatable, and support nested structures (devices, channels, acquisition settings).
- **Options Considered:**
  - Option A: JSON with JSON Schema validation. Pros: Native Python support, good for serialization, widely used. Cons: No comments, verbose for nested configs, less human-friendly for editing.
  - Option B: TOML with custom validation. Pros: Comments supported, clean nested tables, gaining popularity. Cons: Limited complex validation, fewer libraries, less familiar to users.
  - Option C: YAML + Pydantic models. Pros: Comments supported, most readable for humans, Pydantic provides rich validation and type hints, supports complex nested structures. Cons: YAML parsing quirks (tabs vs spaces), requires PyYAML dependency. 
- **Decision:** YAML + Pydantic. 
- **Rationale:** YAML readability critical for scientists/reviewers editing configs manually. Pydantic provides: Automatic type coercion ("100" → int(100)), Rich error messages pointing to exact config line. IDE autocomplete for config objects in code, Easy to define nested models (e.g., HardwareConfig.devices.camera.properties), Comments in YAML essential for documenting hardware-specific settings. Widely adopted in scientific Python tools (e.g., Snakemake, Nextflow)
- **Trade-offs:** YAML indentation sensitivity could frustrate some users (mitigate with examples and schema docs). PyYAML dependency adds ~500KB (acceptable for target users with conda/pip environments).

#### Decision 2: Backend/hardware Abstraction Layer for Hardware
- **Context:** Currently MMSubroutines.py has if/elif conditionals on microscope names and directly uses pymmcore/pycromanager. Need hardware abstraction for dummy mode and future extensibility. 
- **Options Considered:**
  - Option A: Mock objects (DummyMMC) with same interface as pymmcore. Pros: Minimal code changes, existing DummyMMC already works. Cons: Tightly coupled to pymmcore API, can't switch backends easily
  - Option B: Abstract backend interface with adapter implementations. Pros: Clean separation, extensible to future hardware libraries, testable. Cons: Additional abstraction layer, more upfront engineering
  - Option C: Dependency injection with protocols (Python 3.8+ typing.Protocol). Pros: Duck typing, no explicit base class needed, very Pythonic. Cons: Less explicit for other developers, IDE support varies
- **Decision:** Option B - Abstract Backend Interface with Adapters
- **Rationale:** Adapters (PycromanagerBackend, PycoreBackend, DummyBackend) selected by config. 
- **Trade-offs** Fair amount of refactoring necessary. 

#### Decision 3: Keep Gooey as Optional Config Generator
- **Context:** Current entry point is Gooey GUI that directly launches acquisition. Need to decouple config generation from execution for CLI-based workflows and testing.
- **Options Considered:**
  - Option A: Remove Gooey entirely, CLI-only. Pros: Simpler dependency tree, forces config-centric design. Cons: Loses user-friendly GUI that current lab members rely on. 
  - Option B: Refactor Gooey to call CLI runner with generated configs. Pros: Preserves GUI familiarity, enforces config decoupling. Cons: Additional engineering effort, Gooey becomes wrapper around CLI.
  - Option C: Keep Gooey as config writer, separate CLI runner. Pros: Minimal refactoring, GUI users generate YAML then run separately, reviewers use CLI directly. Cons: Two-step workflow (write config, then run) may confuse current users. 
- **Decision:** Option C - Gooey as Config Writer + Separate CLI Runner
- **Rationale:** Preserves value of GUI for non-technical users (postdocs, new students). Gooey naturally writes to files already (gooey_config_reload.json) - just change to YAML output. CLI runner (clef-run) becomes primary entry point. Testable in CI/CD without GUI. Scriptable for batch experiments. Works on headless servers. Gooey becomes optional dependency (pip install clef[gui]). Two-step workflow actually benefits reproducibility: saved YAML configs = audit trail. 
- **Trade-offs** Current users must adapt to "generate config → run config" workflow (mitigate with migration guide). Gooey code must be maintained separately (but no core logic changes, just YAML serialization).

---

## 5. Detailed Refactoring Plan

### Phase 1: [Phase Name]
**Objective:** [What we're achieving in this phase]  
**Duration:** [Estimated time]  
**Dependencies:** [Prerequisites or other phases]

**Affected Components:**
- [Module/file 1]
- [Module/file 2]

**Tasks:**
- [ ] **Task 1.1:** [Task description]
  - Acceptance criteria: [What "done" looks like]
  - Estimated effort: [Hours/days]
  - Files affected: [List]
  
- [ ] **Task 1.2:** [Task description]
  - Acceptance criteria: [What "done" looks like]
  - Estimated effort: [Hours/days]
  - Files affected: [List]

### Phase 2: [Phase Name]
**Objective:** [What we're achieving in this phase]  
**Duration:** [Estimated time]  
**Dependencies:** [Prerequisites or other phases]

**Affected Components:**
- [Module/file 1]
- [Module/file 2]

**Tasks:**
- [ ] **Task 2.1:** [Task description]
  - Acceptance criteria: [What "done" looks like]
  - Estimated effort: [Hours/days]
  - Files affected: [List]

### Phase 3: [Phase Name]
**Objective:** [What we're achieving in this phase]  
**Duration:** [Estimated time]  
**Dependencies:** [Prerequisites or other phases]

**Affected Components:**
- [Module/file 1]
- [Module/file 2]

**Tasks:**
- [ ] **Task 3.1:** [Task description]
  - Acceptance criteria: [What "done" looks like]
  - Estimated effort: [Hours/days]
  - Files affected: [List]

---

## 6. Progress Tracking

### Overall Status
- **Current Phase:** Phase 1: [Phase Name]
- **Overall Completion:** 0%
- **Last Updated:** [Date/Time]
- **Next Milestone:** [Description]
- **Estimated Completion Date:** [Date]

### Phase Status
| Phase | Status | Completion | Start Date | End Date | Notes |
|-------|--------|------------|------------|----------|-------|
| Phase 1: [Name] | ⬜ Not Started | 0% | [Date] | [Date] | - |
| Phase 2: [Name] | ⬜ Not Started | 0% | [Date] | [Date] | Depends on Phase 1 |
| Phase 3: [Name] | ⬜ Not Started | 0% | [Date] | [Date] | Depends on Phase 2 |

**Status Legend:**
- ⬜ Not Started
- 🟡 In Progress
- ✅ Complete
- 🔴 Blocked
- ⚠️ At Risk

### Completed Tasks
*[Tasks will be moved here as they're completed]*

- [x] Task 0.0: Planning document created - Completed [date]

### In Progress Tasks
*[Currently active tasks]*

*None*

### Upcoming Tasks (Next 3-5)
- [ ] Task 1.1: [Description]
- [ ] Task 1.2: [Description]
- [ ] Task 1.3: [Description]

### Blockers & Issues
| ID | Issue | Severity | Status | Assigned To | Created | Resolution |
|----|-------|----------|--------|-------------|---------|------------|
| - | *No blockers yet* | - | - | - | - | - |

**Severity Legend:**
- 🔴 Critical (Stops all progress)
- 🟠 High (Blocks current task)
- 🟡 Medium (Slows progress)
- 🟢 Low (Minor inconvenience)

### Deviations from Plan
*[Track when the actual implementation differs from the plan]*

*None yet*

### Key Decisions Made During Refactoring
*[Capture important decisions made during implementation]*

*None yet*

### Weekly Progress Log

#### Week of [Date]
- **Completed:** [Summary]
- **In Progress:** [Summary]
- **Planned for Next Week:** [Summary]
- **Challenges:** [Any issues encountered]
- **Velocity:** [Tasks completed / Planned tasks]

---

## 7. Testing Strategy

### Test Plan

#### Unit Tests
- **Coverage Target:** [X]%
- **Priority Areas:**
  - [Module/component 1]
  - [Module/component 2]
- **New Test Files:**
  - [test_file_1.py]
  - [test_file_2.py]

#### Integration Tests
- **Scenarios to Cover:**
  1. [Scenario 1]
  2. [Scenario 2]
- **Test Environment:** [Description]

#### Regression Tests
- **Strategy:** [How we'll ensure no breaking changes]
- **Test Suite:** [Which existing tests to run]
- **Frequency:** [When to run]

### Performance Testing
- **Benchmarks to Run:**
  1. [Benchmark 1]: Target [X ms/operations per second]
  2. [Benchmark 2]: Target [Y ms/operations per second]
- **Load Testing:** [Scenarios and acceptance criteria]
- **Tools:** [JMeter / Locust / k6 / etc.]

### QA Checkpoints
- [ ] After Phase 1: [What to validate]
- [ ] After Phase 2: [What to validate]
- [ ] Before production: [What to validate]

---

## 8. Risk Management

### Identified Risks

#### Risk 1: [Risk Name]
- **Description:** [What could go wrong]
- **Probability:** [High/Medium/Low]
- **Impact:** [High/Medium/Low]
- **Mitigation Strategy:** [How to prevent/reduce]
- **Contingency Plan:** [What to do if it happens]
- **Owner:** [Who's responsible for monitoring]

#### Risk 2: [Risk Name]
- **Description:** [What could go wrong]
- **Probability:** [High/Medium/Low]
- **Impact:** [High/Medium/Low]
- **Mitigation Strategy:** [How to prevent/reduce]
- **Contingency Plan:** [What to do if it happens]
- **Owner:** [Who's responsible for monitoring]

### Rollback Plan

#### Conditions That Trigger Rollback
1. [Condition 1 - e.g., Critical bug affecting users]
2. [Condition 2 - e.g., Performance degradation > 20%]
3. [Condition 3 - e.g., Data integrity issues]

#### Rollback Procedure
1. [Step 1]
2. [Step 2]
3. [Step 3]

#### Data Recovery
- **Backup Strategy:** [How we're backing up data]
- **Recovery Time Objective (RTO):** [Target time to recover]
- **Recovery Point Objective (RPO):** [Maximum acceptable data loss]

---

## 9. Communication & Coordination

### Stakeholders
| Name | Role | Interest | Communication Frequency |
|------|------|----------|------------------------|
| [Name] | [Role] | [What they care about] | [Weekly/Bi-weekly/etc.] |
| [Name] | [Role] | [What they care about] | [Weekly/Bi-weekly/etc.] |

### Review Points
- **Phase 1 Completion:** [Who reviews, what they review]
- **Mid-Project Review:** [Date/milestone, participants]
- **Pre-Production Review:** [Who reviews, what they review]

### Documentation Updates Required
- [ ] API documentation
- [ ] Architecture diagrams
- [ ] README files
- [ ] Deployment guides
- [ ] Developer onboarding docs
- [ ] [Other specific docs]

### Training Needs
- [ ] [Training topic 1]: For [team/role]
- [ ] [Training topic 2]: For [team/role]

### Status Report Schedule
- **Frequency:** [Daily/Weekly/etc.]
- **Format:** [Standup/Email/Slack update]
- **Recipients:** [Who gets updates]

---

## 10. Post-Refactoring

### Validation Checklist
- [ ] All unit tests passing (100% of [X] tests)
- [ ] Integration tests passing
- [ ] Performance benchmarks met
  - [ ] [Metric 1] meets target
  - [ ] [Metric 2] meets target
- [ ] Code review completed and approved
- [ ] Documentation updated
- [ ] Deployment successful in staging
- [ ] Smoke tests passed in production
- [ ] Monitoring and alerts configured
- [ ] Stakeholder sign-off received

### Success Metrics Review
| Metric | Before | After | Target | Met? |
|--------|--------|-------|--------|------|
| Test Coverage | [X]% | [Y]% | [Z]% | ✅/❌ |
| Build Time | [X min] | [Y min] | [Z min] | ✅/❌ |
| [Metric] | [X] | [Y] | [Z] | ✅/❌ |

### Lessons Learned
*[To be filled after completion]*

#### What Went Well
1. [Success 1]
2. [Success 2]

#### What Could Be Improved
1. [Area for improvement 1]
2. [Area for improvement 2]

#### Unexpected Challenges
1. [Challenge 1 and how we solved it]
2. [Challenge 2 and how we solved it]

#### Time/Effort Analysis
- **Estimated Total Effort:** [X person-weeks]
- **Actual Total Effort:** [Y person-weeks]
- **Variance:** [±Z%]
- **Insights:** [What we learned about estimation]

### Follow-up Items
*[Technical debt or improvements that remain]*

- [ ] [Item 1]: Priority [High/Medium/Low]
- [ ] [Item 2]: Priority [High/Medium/Low]

---

## 11. Reference Materials

### Related Documents
- [Architecture Decision Records (ADRs)]
- [Original requirements/specification]
- [Design documents]

### Diagrams
- [Link to architecture diagrams]
- [Link to sequence diagrams]
- [Link to ERD/database schemas]

### External Resources
- [Relevant blog posts or articles]
- [Documentation for libraries/frameworks used]
- [Research papers or case studies]

### Repository Information
- **Repository URL:** [URL]
- **Refactor Branch:** [branch-name]
- **Project Board:** [URL to project management board]
- **CI/CD Pipeline:** [URL]

---

## Appendix

### Glossary
- **[Term 1]:** [Definition]
- **[Term 2]:** [Definition]

### Change Log
| Date | Version | Author | Changes |
|------|---------|--------|---------|
| [Date] | 1.0 | [Name] | Initial document creation |

---

**Document Status:** 🟡 In Progress | ✅ Complete | 🔴 Blocked