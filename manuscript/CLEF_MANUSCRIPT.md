# CLEF: A Python Framework for Closed-Loop Neuroscience Experiments

Raymond L. Dunn, Saul Kato

## Summary

CLEF (Closed-Loop Experimental Framework) is an open-source Python platform for real-time closed-loop experiments with automated stimulus control. CLEF primarily targets imaging modalities for biology researchers, especially for neuroscience experiments. CLEF enables researchers to observe neural activity or behavior, process real-time data, and deliver stimuli in a single feedback loop governed entirely by declarative configuration files.

The core philosophy behind CLEF is that a flexible framework will allow custom experimentation to scale. CLEF replaces ad-hoc lab scripts and GUI-only workflows with a modular, config-driven architecture where hardware backends, analysis algorithms, and stimulus controllers are interchangeable components selected at runtime via YAML.

## Closed-Loop Experimentation for Advancing Neuroscience

Traditional neuroscience experiments follow a fixed, open-loop protocol: researchers design the experiment, set parameters, collect data, and analyze results afterward. This approach works for many questions, but it misses something fundamental about how biological systems operate. Brains are highly recurrent networks where activity patterns influence future states in complex ways. To understand causal relationships in these systems, we need experiments that can respond to what they observe.

Closed-loop experimental design allows real-time modification of stimulus protocols based on ongoing measurements. This approach enables several types of experiments that are impossible with fixed protocols:

- **State-dependent interventions:** Perturbing specific neural states when they occur naturally, rather than at arbitrary times.
- **Adaptive testing:** Adjusting stimulus parameters to maintain animals in particular behavioral or neural states.
- **Pattern-triggered stimulation:** Intervening when specific activity patterns appear across neural populations.
- **Predictive control:** Using models to anticipate state transitions and intervene preemptively.

Recent work has demonstrated the value of this approach across multiple model systems (Grosenick et al., 2015). As measurement technologies scale to capture hundreds or thousands of neurons simultaneously, closed-loop methods become increasingly important for understanding network-level mechanisms.

## The Need for Flexible Automation Tools

Closed-loop experiments require coordinating multiple hardware components (cameras, stages, stimulation devices) while performing real-time computation on streaming data. The computational pipeline must extract features from raw measurements, decide what to do based on the current state, and execute stimulus protocols with precise timing. Building this coordination layer from scratch for every new experiment is a barrier to wider adoption of closed-loop methods.

Existing tools each cover part of the problem. Several require non-Python languages (Bonsai is C#, ScanImage is MATLAB, Open Ephys is C++). Others target electrophysiology rather than imaging (RTXI, Open Ephys). Others provide offline analysis libraries without experiment orchestration (CaImAn, Suite2p). Commercial microscopy software exposes limited scripting that works for simple automated protocols but lacks the sophistication needed for responsive, state-dependent experiments. Pycro-Manager (Pinkard et al., 2021) established Python-based microscope control and enabled more complex acquisition sequences, but it does not, on its own, supply the architecture needed for closed-loop experimentation with multiple hardware components and real-time decision making.

Researchers performing volumetric calcium imaging with closed-loop stimulus delivery currently lack a Python-native framework that unifies hardware control, online analysis, and stimulus triggering under a single reproducible configuration. CLEF fills this gap by providing:

- **Symmetric input/output device abstraction:** Independent `BaseInputDevice` and `BaseOutputDevice` interfaces for any data source (cameras, stages, electrode signals) and any actuator (DMDs, light sources, motorized stages, keyboard/mouse), so that experimental logic is independent of specific hardware choices.
- **Pluggable logic registry:** A plugin system for implementing custom decision logic, including simple threshold-based triggers, machine learning models, and offloaded inference on a remote server.
- **Configuration-based deployment:** Three YAML files (`io`, `logic`, `session`) validated by Pydantic models, supporting reproducibility and protocol sharing across labs.
- **User interfaces:** Both headless operation for automated experiments and graphical interfaces for interactive control.

The framework is written entirely in Python, so it is accessible to the large community of scientists already using Python for data analysis. Researchers can use NumPy, SciPy, scikit-learn, and PyTorch directly inside their experimental logic.

![System Overview](diagrams/system_overview.svg)
**Figure 1.** System overview of CLEF. Three YAML configuration files (`io.yaml`, `logic.yaml`, `session.yaml`) are validated by Pydantic models and passed to the ClosedLoopEngine, which orchestrates the real-time acquisition loop. The engine reads samples from one or more registered `input_devices`, passes each sample to the active `logic` algorithm, and dispatches return values to one or more named `output_devices`.

## Architecture

CLEF uses a layered architecture in which independent components communicate through small interfaces. Researchers can customize parts of the system (adding new hardware, implementing new algorithms) without understanding the rest of the codebase.

### Key Concepts

CLEF is organized around four concepts that recur throughout the rest of this paper:

| Keyword   | Purpose                                                                                                                                                                                                                                                            |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `device`  | Components or endpoints that CLEF interacts with. A `device` is either an `input_device` or an `output_device`. Input devices provide data streams (e.g. camera images), and output devices are what the experimenter wants to control (e.g. a stage or a laser). |
| `logic`   | The control algorithm. It processes samples from the input devices and emits updates for the output devices.                                                                                                                                                       |
| `session` | Contextual metadata for an experiment. For example, information about the data subject (cell line, treatment condition), highly specific to the application.                                                                                                       |
| `engine`  | A discrete event loop that orchestrates iterations of data sampling, data processing, and actuation.                                                                                                                                                               |

An experiment can use one or more input devices and one or more output devices: multiple cameras, a camera plus a stage readout, or a DMD plus a laser plus a stage. At runtime, the engine reads from every registered input device, passes the samples to the logic algorithm, and dispatches output commands to any subset of the registered output devices.

### Code Layout

The repository is split into two top-level trees. `core/` contains the framework internals: base classes, managers, the engine, configuration logic, and the CLI. `apps/` contains concrete implementations that register against the core base classes via plugin discovery, plus per-application YAML configs.

- **CLI entry point** (`core/utils/clef_cli.py`) accepts either a preset name (e.g. `clef <app_name>`) that resolves to a directory under `apps/config/`, or explicit `--session`, `--io`, and `--logic` paths.
- **ConfigManager** (`core/config/config_manager.py`) loads, validates, and merges the three configuration files against Pydantic models (`SessionConfig`, `IOConfig`, `ClosedLoopLogicConfig`), with defaults in `core/config/defaults/`. Validation catches invalid parameters before any hardware is initialized.
- **ClosedLoopEngine** (`core/engine/closed_loop_engine.py`) owns an `IOManager` and a `LogicManager` and runs the sample loop: acquire from input devices, pass to logic, dispatch any returned commands to output devices.
- **IOManager** (`core/io/`) instantiates `BaseInputDevice` and `BaseOutputDevice` subclasses listed in `io.yaml`. Each input device is paired with a `BaseDataInterface` that controls how the data stream is buffered, structured, and persisted at the end of the session.
- **LogicManager** (`core/logic/`) instantiates the `BaseClosedLoopLogic` subclass selected by `logic.yaml` and connects it to the input/output devices held by the IOManager.

At runtime, the data flow is symmetric and minimal:

```
input_devices → engine → closed-loop logic → engine → output_devices
```

![Class Hierarchy](diagrams/class_hierarchy.svg)
**Figure 2.** Class hierarchy of CLEF's core abstractions. Each subsystem defines an abstract base class (`BaseInputDevice`, `BaseOutputDevice`, `BaseDataInterface`, `BaseClosedLoopLogic`) with concrete implementations in `apps/` selected at runtime via configuration. All four base classes use `__init_subclass__` to auto-register subclasses by their `device_class`, `data_interface_class`, or `logic_class` ClassVar string. The `IOManager` and `LogicManager` auto-import every `.py` file under `apps/io/` and `apps/logic/` at startup, so dropping a new module into the right folder is sufficient to make it discoverable. No explicit registration step is required.

## Customizing CLEF for a New Experiment

A new closed-loop experiment is built by writing (or reusing) one or more input devices, one or more output devices, one or more logic algorithms, and three configuration files. The four base abstractions (`BaseInputDevice`, `BaseOutputDevice`, `BaseDataInterface`, and `BaseClosedLoopLogic`) each define a small interface that researchers fill in for their own hardware and analysis. A single experiment can fan in samples from several cameras, electrode signals, or stage readouts, and fan out to multiple actuators (a DMD plus a light source plus a stage, for example). The engine and logic compose them through `io.yaml`. The remainder of this section walks through the customization workflow in the order a researcher encounters it.

### Step 1: Create a module for your input device

Input devices (`apps/io/input_device/`) wrap any source of samples: a camera, an electrode amplifier, a tracking stage's position readout, or a TIFF stack on disk. A new input device subclasses `BaseInputDevice`, declares a unique `device_class` string used for registry lookup, and implements four methods. `connect()` opens the connection to the hardware. `configure()` applies settings drawn from `self.config`. `_get_input()` returns the next sample. `close()` releases the hardware at the end of the session. A single experiment can list any number of input devices in `io.yaml` (for example, one camera per imaged region plus a stage-position readout); the engine acquires from all of them on each iteration of the loop. Each input device is paired in `io.yaml` with a `data_interface` (Step 4) that defines the structure of its samples.

As example implementations from our own lab, CLEF currently ships with `micromanager_camera_input` (cameras driven through Micro-Manager / pycro-manager), `screenshot_camera_input` (screen capture as a virtual camera, useful for piping arbitrary GUIs into a CLEF pipeline), `recording_playback_input` (playback of an existing TIFF recording for offline algorithm development), and `xy_tracking_stage_input` (position readout from a motorized XY stage), along with synthetic input devices used by the bundled demos (see Demos).

### Step 2: Create a module for your output device

Output devices (`apps/io/output_device/`) wrap any actuator the experiment needs to drive: a stimulating LED, a DMD, a motorized stage, or a synthetic mouse/keyboard. A new output device subclasses `BaseOutputDevice`, declares a `device_class` string, and implements `connect()`, `configure()`, `_update_output(**kwargs)` (which actually fires the actuator), and `close()`. As with input devices, any number of output devices can be listed in `io.yaml`, and a single iteration of the closed loop can address several at once (for example, configuring a DMD pattern and triggering a laser in the same step). The engine routes a dict of `{output_device_name: {kwargs}}` returned by the logic algorithm (Step 3) to each named device's `update_output`, which records a timestamp and calls `_update_output(**kwargs)`.

As example implementations from our own lab, CLEF ships with `mightex_polygon_output` (Mightex Polygon1000 DMD for spatially patterned optogenetic stimulation), `ldi_89north_output` (89 North LDI light source), `mm_dac_lightsource` (Micro-Manager-controlled DAC light source), `micromanager_stage_output` (Micro-Manager-controlled motorized stage), `xy_tracking_stage_output` (XY tracking stage actuator), and `mouse_keyboard_output` (synthetic mouse and keyboard events for closing the loop on third-party GUI software), along with synthetic output devices used by the bundled demos.

The symmetric input/output split (rather than a monolithic "hardware backend" plus a separate "stimulus controller") keeps every device type behind the same small, narrow interface, so adding a new device does not require touching framework code.

### Step 3: Create a module for your closed-loop logic algorithm

Logic algorithms (`apps/logic/`) implement the experiment's online analysis and decision rules. A new algorithm subclasses `BaseClosedLoopLogic`, declares a `logic_class` string, and implements four methods. `initialize_model()` runs once before the loop starts (load weights, allocate buffers, open GUI windows). `process_sample(sample)` receives the next sample from the input devices and updates internal state. `_check_logic()` decides whether to drive an output device on this frame, returning either `None` (do nothing) or a dict of `{output_device_name: {kwargs}}`. `close()` cleans up at the end of the session.

For events that do not need engine-level timestamping (for example, updating a real-time visualization), `process_sample` may also call output devices directly. Once a logic class is dropped into `apps/logic/`, the framework auto-discovers it at startup. No framework code changes are required.

The logic algorithms shipped with CLEF range in complexity from few-line auto-triggers to multi-process pipelines that hand shared-memory volumes to a worker subprocess for online segmentation and activity extraction. See Demos for concrete examples.

### Step 4: Pair input devices with a data interface

The `BaseDataInterface` abstraction separates _how a sample looks_ from _how it is stored_. Each input device is paired with a `data_interface` in `io.yaml`. The data interface defines sample shape, dtype, and end-of-session serialization (for example, writing a TIFF stack, an HDF5 file, or a JSON-lines log). As example implementations from our own lab, three concrete data interfaces ship with CLEF: `uint16_data_interface` (2-D uint16 frames, the default for grayscale microscopy cameras), `rgb_data_interface` (3-channel RGB images), and `shm_uint16_data_interface` (uint16 frames in POSIX shared memory, for zero-copy handoff to subprocess workers).

Because the data interface is a separate registered class, the same input device can serve different downstream pipelines by switching its data interface in YAML. A new modality (an audio stream, or a 1-D timeseries) can be added by writing a new `BaseDataInterface` subclass with no engine changes.

### Step 5: Write the three configuration files

CLEF is entirely directed by three YAML configuration files, each corresponding to one of the framework's core concepts and validated by Pydantic at startup:

- `io.yaml`. Lists input and output devices for the experiment. Each entry names a `device_class` (the registered Python plugin to load) along with any device-specific parameters that vary across experiments (camera exposure, ROI, illumination properties, serial port, etc.). Each input device entry also names a `data_interface` for storage.
- `logic.yaml`. Selects the closed-loop algorithm via `logic_class` and supplies its tunable parameters (thresholds, gains, target regions, model URLs, etc.).
- `session.yaml`. Describes the run itself rather than the hardware or algorithm: who ran the session, when, on what subject, under what conditions, where outputs are written, and how long the run lasts.

The YAML files for a given application live together under `apps/config/<app_name>/`. Validation catches invalid parameters before any hardware is initialized.

### Step 6: Run the experiment

The CLI shorthand `clef <app_name>` resolves to `apps/config/<app_name>/` and loads all three files. `clef --session s.yaml --io io.yaml --logic l.yaml` accepts arbitrary paths instead. `clef <app_name> --validate-config` runs full Pydantic validation without starting the loop, which is useful when iterating on configuration during development.

At the end of every session, CLEF writes per-device data files (using each device's `data_interface`) plus a single JSON metadata file capturing the merged configuration, every output event with its timestamp, and per-device sample timestamps.

### Offloading heavy logic to a subprocess or remote server

For computationally heavy models, the logic algorithm itself can offload work to a separate process or remote server. Two patterns are supported. A subprocess worker can consume frames out of POSIX shared memory (`shm_uint16_data_interface`) for zero-copy handoff, keeping inference in a separate Python process on the same machine. Alternatively, the logic class can stream samples over a WebSocket to a FastAPI inference server running in a Docker container, locally or on a cloud VM. The `docker/` directory ships a multi-stage `Dockerfile.decoder` and a `deploy_decoder.sh` script demonstrating an Artifact Registry to GCP Compute Engine deploy. Both patterns keep the real-time loop responsive even when inference latency or memory requirements exceed what the acquisition machine can provide. See Demos for concrete examples of each.

### Micro-Manager integration via pycro-manager

CLEF's primary microscope-hardware path interfaces through pycro-manager (Pinkard et al., 2021) and pymmcore, the Python bindings for Micro-Manager (Edelstein et al., 2014). Micro-Manager provides device adapters for hundreds of microscopy components (cameras, stages, filter wheels, light sources, shutters), and pycro-manager exposes this functionality to Python with support for multi-dimensional acquisition, hardware-triggered sequencing, and direct access to the Micro-Manager Core API.

By building on this ecosystem, CLEF inherits broad hardware compatibility without implementing device-specific drivers. CLEF runs alongside a Micro-Manager instance, sending commands and receiving data through pycro-manager's Python API. Researchers can use familiar Micro-Manager configurations and device adapters without modification. The `micromanager_camera_input`, `micromanager_stage_output`, and `mm_dac_lightsource` devices wrap these APIs behind CLEF's input/output device base classes, so the rest of the framework (engine, logic) remains agnostic to the underlying hardware communication layer.

## Key Design Features

| Feature                           | Description                                                                                                                                                                  |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Config-driven reproducibility** | Entire experiments are specified in three version-controllable YAML files (`io`, `logic`, `session`). Pydantic validation catches errors before acquisition begins.          |
| **Symmetric IO abstraction**      | Independent `BaseInputDevice` and `BaseOutputDevice` interfaces. Cameras, stages, lasers, DMDs, and synthetic data sources all sit behind the same kind of small interface.  |
| **Pluggable logic**               | Registry-based discovery. New algorithms are added by dropping a `BaseClosedLoopLogic` subclass into `apps/logic/` and referencing its `logic_class` in YAML.                |
| **Generic data interface**        | `BaseDataInterface` decouples sample shape and storage format from device. Supports uint16 microscopy volumes, RGB screenshots, shared-memory volumes, and other modalities. |
| **Network-service logic**         | Heavy models can run as a separate FastAPI/WebSocket service (locally in Docker or remotely on a cloud VM) and be consumed by a thin client logic class.                     |
| **Per-session metadata bundle**   | At end of session, CLEF writes per-device data files plus a JSON metadata file containing all configuration, events, and per-device timestamps.                              |
| **Testability**                   | Synthetic and playback input devices enable full integration testing without physical hardware (see Demos).                                                                  |

## Installation

CLEF is installable via `pip install -e .` from the repository root. Optional extras gate hardware- and demo-specific dependencies: `[demos]` installs everything required for the synthetic and screenshot demos; `[all]` adds the full set of hardware integrations.

## Data Model and Storage

At the end of every session, CLEF writes a session directory containing one data file per input device (using the format defined by that device's `data_interface`) plus a single JSON metadata file capturing the merged configuration, every output event with its timestamp, and per-device sample timestamps. This makes each session self-describing: the JSON metadata is sufficient to reconstruct exactly what hardware was used, what logic ran, with what parameters, and when each sample and stimulus occurred.

For labs that already use minimo (Borchardt et al., 2021), a linked data and metadata storage system that combines object storage (MinIO) for large raw data files with a document database (MongoDB) for metadata, the per-session bundle maps cleanly onto minimo's object-plus-metadata model: raw input-device files become immutable objects and the JSON metadata file becomes the searchable document. CLEF does not require minimo, however; the session bundle is a plain directory on disk.

## Demos

CLEF ships with several demo configurations covering a range of use cases, including synthetic dynamical systems, recorded data playback, and real microscope hardware. Each demo lives in its own directory under `apps/config/`.

### Limit Cycle Demo (`apps/config/limit_cycle/`)

The limit cycle demo captures the core conceptual motivation for CLEF. Closed-loop experimental frameworks exist to interrogate dynamical biological processes, and brains are perhaps the most compelling example. In this demo, a synthetic dynamical system (a bistable system with two concentric limit cycles) produces time-varying data, and the experimenter delivers perturbations to understand how the system works. The setup is essentially a game: the system evolves according to its own dynamics, and the experimenter must observe, hypothesize, and intervene in real time.

The `limit_cycle_input` device generates 2-D image frames showing a punctum orbiting on one of two concentric rings. The system state is defined by an angular position and a radial mode (inner or outer ring). The `limit_cycle_output` device delivers stimuli that perturb the angular position or toggle the system between rings. An auto-trigger mode is available, in which the logic fires stimuli when the system enters a specified angular region. The demo illustrates how CLEF separates data generation, online analysis, and stimulus control into independently configurable components.

![Limit Cycle Demo](diagrams/screenshot_limit_cycle.png)
**Figure 3.** Screenshot of the limit cycle demo during a live session. The visualization shows the punctum orbiting on one of two concentric rings, with a fading trajectory trail indicating recent history. The experimenter can deliver angular perturbations or toggle the system between rings using the GUI controls, observing the effect of each intervention in real time.

Run with `clef limit_cycle`.

### Recording Playback Demo (`apps/config/recording_playback/`)

The recording playback demo (used by the brainalyzer workflow) provides the same GUI and analysis interface as the physical hardware configuration, but reads data from an existing volumetric calcium imaging dataset (a TIFF stack) via `recording_playback_input` rather than acquiring live from a microscope. Users can develop, test, and refine their analysis pipelines against real neural data without needing access to microscope hardware. The `brainalyzer_logic` algorithm performs real-time quantification of neural activity across z-planes (offloaded to `brainalyzer_worker.py` over shared memory) and supports stimulus parameter exploration through the interactive GUI.

![Brainalyzer Demo](diagrams/screenshot_brainalyzer.png)
**Figure 4.** Screenshot of the brainalyzer pipeline running over the recording-playback demo. The interface displays volumetric calcium imaging data read from a TIFF stack, with real-time quantification of neural activity across z-planes. This demo provides the same analysis and stimulus control interface as the physical hardware configuration, allowing algorithm development and parameter exploration without a connected microscope.

Run with `clef recording_playback`.

### Screenshot Demo (`apps/config/screenshot/`)

The screenshot demo uses `screenshot_camera_input` to treat a region of the desktop as a virtual camera. This is useful for closing the loop on third-party GUI software that does not expose a programmatic data API. Combined with `mouse_keyboard_output`, it allows CLEF to drive arbitrary GUI applications based on what they display on screen.

Run with `clef screenshot` (or via the higher-level alias if configured).

### Speech BCI Demo (`apps/config/speech_bci/`)

The speech BCI demo illustrates the network-service logic pattern. `speech_bci_input` provides neural data, the logic algorithm streams it over WebSocket to a FastAPI decoder server (`apps/subprocess/speech_bci_server.py`) that runs a GRU + n-gram language model inside Docker (locally or on a GCP Compute Engine VM via `docker/deploy_decoder.sh`), and the decoded text is rendered through `speech_bci_output`. The demo shows how CLEF supports models too heavy to run on the acquisition machine itself.

Run with `clef speech_bci` (after deploying the decoder service).

### Physical Hardware Demo (`apps/config/physical_hardware/`)

The physical hardware demo runs a full volumetric calcium imaging experiment with real-time quantification and patterned optogenetic illumination using the Mightex Polygon1000 digital micromirror device. It can only be run with the appropriate microscope hardware configured (camera, stage, Micro-Manager device adapters, and Polygon1000). This is the primary production use case for CLEF: acquiring volumetric calcium imaging data, processing it online via the brainalyzer logic, and delivering spatially patterned optogenetic stimuli in closed loop.

Run with `clef hardware_physical`.

### XY Tracking Stage Demo (`apps/config/tracking_stage/`)

The XY tracking stage demo combines `xy_tracking_stage_input` (position readout) with `xy_tracking_stage_output` (motorized actuation) and `xy_tracking_logic` to keep a moving subject centered in the field of view in real time. The demo illustrates the symmetric input/output split when both ends of the loop touch the same physical device.

### Why These Demos

Each demo was chosen to highlight a different aspect of the framework:

- The **limit cycle** demo shows that CLEF's architecture is not specific to microscopy. Any system that produces sequential data samples and accepts perturbations can be wrapped as a backend, and new users can explore the full closed-loop workflow on any machine, with no hardware dependencies.
- The **screenshot** demo extends this to closing the loop on arbitrary third-party GUI software via virtual camera plus synthetic mouse/keyboard.
- The **recording playback** demo bridges the gap between synthetic and real data, supporting algorithm development against genuine neural recordings without tying up microscope time.
- The **speech BCI** demo shows the network-service logic pattern for models too heavy to run inline.
- The **physical hardware** demo shows the end-to-end production configuration, including CLEF's integration with Micro-Manager, pycro-manager, and real optical hardware.
- The **XY tracking stage** demo shows a single-device closed loop in which the same device appears as both input and output.

## Example Implementation: Whole-Brain Closed-Loop Imaging in _C. elegans_

The first complete CLEF implementation was developed for closed-loop whole-brain imaging in _C. elegans_. The system integrates volumetric calcium imaging with optogenetic stimulation, so that targeted perturbations can be delivered based on observed neural dynamics.

### System Capabilities

The system captures neural activity across the entire _C. elegans_ nervous system (302 neurons) at cellular resolution while the animal is restrained in a microfluidic chip. Real-time image processing identifies and tracks individual neurons, extracts fluorescence signals indicating neural activity, and detects specific patterns across neural populations. Based on these observations, optogenetic stimulation is delivered with spatial precision (targeting specific neurons via DMD) and millisecond temporal precision.

The system provides two operational modes:

- **Interactive mode.** A graphical interface displays real-time visualizations of imaging data, processed signals, and system state. Researchers can monitor experiments, manually trigger stimulation for testing, and adjust parameters on the fly. The interface includes panels showing raw images, segmented neurons, extracted activity traces, and low-dimensional projections of population dynamics.
- **Headless mode.** For automated experiments that do not require real-time monitoring, the system runs without a graphical interface, reducing computational overhead and supporting longer unattended experiments. All data are still logged and available for later review.

The configuration system allows the same experimental logic to run in either mode by swapping a small number of fields in `logic.yaml`. Researchers can prototype interactively and then deploy the same logic for automated data collection.

### Data Pipeline

The closed-loop pipeline operates in several stages:

1. **Volumetric acquisition.** `micromanager_camera_input` paired with `shm_uint16_data_interface` captures z-stacks at rates up to 5 volumes per second. Piezoelectric focus control synchronizes with camera exposures to capture optical sections through the depth of the animal.
2. **Image processing.** Each volume undergoes motion correction, background subtraction, and noise filtering using optimized implementations (Numba-compiled functions, OpenCV routines) inside a subprocess worker, which reads the volume out of shared memory to maintain throughput.
3. **Neuron identification.** Segmentation algorithms identify individual neurons in each volume based on fluorescence reporter expression. The system tracks neurons across volumes to maintain consistent identities despite animal movement.
4. **Activity extraction.** For each identified neuron, fluorescence time series are extracted and processed (baseline correction, detrending) to isolate activity signals from imaging artifacts.
5. **Feature analysis.** Dimensionality reduction and pattern detection extract relevant features from population activity, including principal component analysis, threshold-based event detection, and correlation analysis.
6. **Decision logic.** Based on the extracted features and the current experimental context, the logic algorithm's `_check_logic` returns either `None` or a dict naming `mightex_polygon_output` (and any light source) with the parameters for the next stimulus.
7. **Stimulus delivery.** The engine dispatches the returned dict to `mightex_polygon_output`, which configures the spatial pattern on the DMD, and to the configured light source, which controls laser timing. Stimulus delivery is synchronized with acquisition to avoid crosstalk.

The full pipeline completes in under 100 milliseconds, so the system can respond to neural dynamics on timescales relevant to behavior.

### Applications

This implementation has enabled several types of experiments:

- **State-dependent perturbations.** Monitor population activity to identify specific neural states, then trigger optogenetic stimulation when those states occur. This supports causal testing of how neural states influence behavior or future dynamics.
- **Pattern-triggered interventions.** Detect specific patterns of co-activity across neurons and perturb the network at precise moments in its operation, revealing how temporal coordination contributes to circuit function.
- **Closed-loop behavioral control.** Maintain animals in particular behavioral states by adjusting sensory stimuli based on neural activity. This supports the study of how neural dynamics relate to behavior while controlling for state-dependent effects.

These experiments would be difficult or impossible with open-loop protocols, because the relevant neural states occur at unpredictable times and last for short durations.

## Additional Use Cases

CLEF's modular design supports a range of closed-loop experimental paradigms beyond whole-brain imaging in _C. elegans_.

- **Multi-region recording with targeted stimulation.** Multiple input devices can be listed in a single `io.yaml`, so different cameras can record different brain areas while a central logic algorithm analyzes across all streams to make stimulation decisions. This supports experiments on inter-regional communication.
- **Adaptive behavioral testing.** CLEF can monitor neural and behavioral readouts simultaneously (for example, a microscopy camera plus a tracking stage) and use both to guide experimental logic. Sensory stimuli can be adjusted to maintain animals in specific behavioral states or to test state-dependent perturbations.
- **Closing the loop on external GUI software.** The combination of `screenshot_camera_input` and `mouse_keyboard_output` allows CLEF to wrap third-party applications that do not expose a programmatic API.
- **Heavy-model inference.** The network-service pattern (FastAPI WebSocket server in Docker, deployed locally or to a cloud VM) generalizes to any model whose memory or latency requirements exceed the acquisition machine's budget.
- **Algorithm development and testing.** Because experimental logic is encapsulated in pluggable `BaseClosedLoopLogic` subclasses, researchers can implement new approaches, test them in simulation using `recording_playback_input`, and then deploy the same logic class against `micromanager_camera_input` in live experiments. The configuration system makes it straightforward to compare different algorithms under identical experimental conditions.

## Related Projects

| Tool               | Language    | Domain                           | Closed-Loop      | Config-Driven             | Reference                |
| ------------------ | ----------- | -------------------------------- | ---------------- | ------------------------- | ------------------------ |
| **Bonsai**         | C#          | General experiment control       | Yes              | No (visual XML workflows) | Lopes et al., 2015       |
| **Open Ephys GUI** | C++         | Electrophysiology                | Yes              | No (GUI)                  | Siegle et al., 2017      |
| **CaImAn**         | Python      | Calcium imaging analysis         | Partial (OnACID) | No                        | Giovannucci et al., 2019 |
| **Suite2p**        | Python      | Calcium imaging analysis         | No (offline)     | No                        | Pachitariu et al., 2017  |
| **Micro-Manager**  | Java/C++    | Microscope control               | No               | Partial                   | Edelstein et al., 2014   |
| **pycro-manager**  | Python/Java | Microscope control (Python)      | No               | No                        | Pinkard et al., 2021     |
| **ScanImage**      | MATLAB      | Two-photon microscopy            | Partial          | No                        | Pologruto et al., 2003   |
| **Stytra**         | Python      | Zebrafish behavior + light-sheet | Yes              | Partial                   | Stih et al., 2019        |
| **RTXI**           | C++         | Hard real-time electrophysiology | Yes              | No                        | Patel et al., 2017       |
| **ACQ4**           | Python      | Patch-clamp + imaging            | Partial          | No                        | Campagnola et al., 2014  |
| **Autopilot**      | Python      | Distributed behavioral rigs      | Yes              | Yes                       | Saunders & Wehr, 2019    |
| **LabVIEW**        | G (visual)  | General instrument control       | Partial          | No                        | National Instruments     |

### What Makes CLEF Unique

1. **Python-native closed-loop for volumetric calcium imaging.** No other Python framework combines real-time volumetric calcium imaging acquisition with online analysis and stimulus feedback. CaImAn provides algorithms but not orchestration. Stytra targets zebrafish light-sheet specifically. Pycro-manager provides microscope control but not closed-loop logic.

2. **Declarative, validated configuration.** Three YAML files validated by Pydantic models mean experiments are fully specified in version-controllable text files. Invalid configurations are caught before hardware is initialized. This contrasts with Bonsai's visual XML workflows, ScanImage's MATLAB scripts, and the ad-hoc parameter passing common in lab code.

3. **Symmetric input/output device abstraction with auto-discovery.** Every camera, stage, light source, DMD, or synthetic data source sits behind a small `BaseInputDevice` or `BaseOutputDevice` interface, and dropping a new module into `apps/io/` is sufficient to register it. `__init_subclass__` handles discovery automatically. The same logic class can run against a synthetic source in CI, a recorded dataset on a laptop, or a `micromanager_camera_input` in the lab.

4. **Logic registry pattern.** New online analysis methods are added by implementing `initialize_model()`, `process_sample()`, and `_check_logic()`, then dropping the file into `apps/logic/`. No framework code changes are required.

5. **Network-service logic for heavy models.** A documented pattern (FastAPI WebSocket server, Docker, GCP Compute Engine deployment) is provided for offloading inference to dedicated hardware while keeping the real-time loop responsive.

6. **Full testability without hardware.** Synthetic and playback input devices support integration testing of the complete pipeline, from config loading through stimulus delivery.

7. **Accessibility through Python.** CLEF is written entirely in Python and is immediately accessible to the community of neuroscientists already using Python for data analysis. Researchers can use NumPy, SciPy, scikit-learn, and PyTorch directly inside their experimental logic, which lowers the barrier between offline analysis development and online deployment.

## Availability and Documentation

CLEF is open-source software released under the MIT license. The code is available on GitHub at [https://github.com/focolab/clef](https://github.com/focolab/clef). Documentation includes installation instructions, configuration guides, a vibe-coding quickstart for AI-assisted app development, and the demo configurations described above.

We welcome contributions from the community. The modular architecture supports adding new hardware devices or new logic types without modifying framework code. Issues and pull requests can be opened on the GitHub repository.

## Acknowledgements

We thank members of the Kato lab for valuable discussion.

Data for this study were acquired at the UCSF Innovation Core at the Weill Institute for Neurosciences on a custom imaging system controlled by the open-source software Micro-Manager.

This work was supported by NIH grants NS115572 (R.L.D), R35GM124735 (S.K.), and the Weill Institute for Neurosciences (S.K.).

## References

- Borchardt, J., Dunn, R. & Kato, S. "minimo: a linked data and metadata storage system for small labs." _J. Open Source Softw._ 6, 2979 (2021). DOI: 10.21105/joss.02979
- Campagnola, L., Kratz, M.B. & Bhatt, D.B. "ACQ4: An open-source software platform for data acquisition and analysis in neurophysiology research." _Front. Neuroinform._ 8, 3 (2014). DOI: 10.3389/fninf.2014.00003
- Edelstein, A.D., Tsuchida, M.A., Amodaj, N., Pinkard, H., Vale, R.D. & Stuurman, N. "Advanced methods of microscope control using μManager software." _J. Biol. Methods_ 1, e10 (2014). DOI: 10.14440/jbm.2014.36
- Giovannucci, A. et al. "CaImAn: An open source tool for scalable calcium imaging data analysis." _eLife_ 8, e38173 (2019). DOI: 10.7554/eLife.38173
- Grosenick, L., Marshel, J.H. & Deisseroth, K. "Closed-loop and activity-guided optogenetic control." _Neuron_ 86, 106-139 (2015). DOI: 10.1016/j.neuron.2015.03.034
- Lopes, G. et al. "Bonsai: An event-driven framework for processing and controlling data streams." _Front. Neuroinform._ 9, 7 (2015). DOI: 10.3389/fninf.2015.00007
- Lorenz, E.N. "Deterministic nonperiodic flow." _J. Atmos. Sci._ 20, 130-141 (1963). DOI: 10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2
- Newman, J.P. et al. "Optogenetic feedback control of neural activity." _eLife_ 4, e07192 (2015). DOI: 10.7554/eLife.07192
- Pachitariu, M. et al. "Suite2p: beyond 10,000 neurons with standard two-photon microscopy." _bioRxiv_ (2017). DOI: 10.1101/061507
- Packer, A.M., Russell, L.E., Dalgleish, H.W.P. & Hausser, M. "Simultaneous all-optical manipulation and recording of neural circuit activity with cellular resolution in vivo." _Nat. Methods_ 12, 140-146 (2015). DOI: 10.1038/nmeth.3217
- Pinkard, H., Stuurman, N., Ivanov, I.E., Anthony, N.M., Ouyang, W., Li, B., Yang, B., Tsuchida, M.A., Chhun, B., Zhang, G., Mei, R., Anderson, M., Shepherd, D.P., Hunt-Isaak, I., Dunn, R.L., Jahr, W., Kato, S., Royer, L.A., Thiagarajah, J.R., . . . Waller, L. "Pycro-Manager: open-source software for customized and reproducible microscope control." _Nat. Methods_ 18, 226-228 (2021). DOI: 10.1038/s41592-021-01087-6
- Pologruto, T.A., Sabatini, B.L. & Bhatt, D.B. "ScanImage: Flexible software for operating laser scanning microscopes." _BioMed. Eng. Online_ 2, 13 (2003). DOI: 10.1186/1475-925X-2-13
- Saunders, J. & Wehr, M. "Autopilot: Automating behavioral experiments with lots of Raspberry Pis." _bioRxiv_ (2019). DOI: 10.1101/807693
- Siegle, J.H. et al. "Open Ephys: An open-source, plugin-based platform for multichannel electrophysiology." _J. Neural Eng._ 14, 045003 (2017). DOI: 10.1088/1741-2552/aa5eea
- Stih, V., Petrucco, L., Kist, A.M. & Portugues, R. "Stytra: An open-source, integrated system for stimulation, tracking and closed-loop behavioral experiments." _PLOS Comput. Biol._ 15, e1006699 (2019). DOI: 10.1371/journal.pcbi.1006699
