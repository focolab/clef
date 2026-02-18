# CLEF: A Python Framework for Closed-Loop Neuroscience Experiments

Raymond L. Dunn, Saul Kato

## Summary

CLEF (Closed-Loop Experimental Framework) is an open-source Python platform for real-time closed-loop microscopy experiments with automated stimulus control. The framework provides a modular architecture for data acquisition, online analysis, and stimulus delivery, allowing experimental parameters to adapt dynamically based on observed biological activity. CLEF abstracts hardware complexity behind unified interfaces while maintaining the flexibility needed for sophisticated experimental protocols.

Although CLEF targets volumetric calcium imaging in neuroscience as its primary use case, its architecture is general-purpose. The clean abstractions make it straightforward to add new backends (video files, audio streams, sensor data) following the same patterns, and the generic `DataInterface` supports any sequential data type.

CLEF replaces ad-hoc lab scripts and GUI-only workflows with a config-driven architecture where hardware backends, analysis algorithms, and stimulus controllers are interchangeable components selected at runtime via YAML.

## Closed-Loop Experimentation for Advancing Neuroscience

Traditional neuroscience experiments follow a fixed, open-loop protocol: researchers design the experiment, set parameters, collect data, and analyze results afterward. This approach works for many questions, but it misses something fundamental about how biological systems operate. Brains are highly recurrent networks where activity patterns influence future states in complex ways. To understand causal relationships in these systems, we need experiments that can respond to what they observe.

Closed-loop experimental design allows real-time modification of stimulus protocols based on ongoing measurements. This approach enables several types of experiments that are impossible with fixed protocols:

- **State-dependent interventions:** Perturbing specific neural states when they occur naturally, rather than at arbitrary times.
- **Adaptive testing:** Adjusting stimulus parameters to maintain animals in particular behavioral or neural states.
- **Pattern-triggered stimulation:** Intervening when specific activity patterns appear across neural populations.
- **Predictive control:** Using models to anticipate state transitions and intervene preemptively.

Recent work has demonstrated the value of this approach across multiple model systems (Grosenick et al., 2015). As measurement technologies scale to capture hundreds or thousands of neurons simultaneously, closed-loop methods become increasingly important for understanding network-level mechanisms.

## The Need for Flexible Automation Tools

Implementing closed-loop experiments requires coordinating multiple hardware components (cameras, stages, stimulation devices) while performing real-time computation on streaming data. The computational pipeline must extract relevant features from raw measurements, make decisions based on current state, and execute stimulus protocols with precise timing. This coordination is technically challenging and has been a barrier to wider adoption of closed-loop methods.

Most existing tools either require non-Python languages (Bonsai, C#; ScanImage, MATLAB; Open Ephys, C++), target electrophysiology rather than imaging (RTXI, Open Ephys), or provide analysis libraries without experiment orchestration (CaImAn, Suite2p). Commercial microscopy software provides limited scripting capabilities that work for simple automated protocols but lack the sophistication needed for responsive, state-dependent experiments. Pycro-Manager (Pinkard et al., 2021) established Python-based microscope control and enabled more complex acquisition sequences, but it does not provide the full architecture needed for closed-loop experimentation with multiple hardware components and real-time decision making.

Researchers performing volumetric calcium imaging with closed-loop stimulus delivery currently lack a Python-native framework that unifies hardware control, online analysis, and stimulus triggering under a single reproducible configuration. CLEF fills this gap by providing:

- **Hardware abstraction:** A unified interface for common microscopy components (cameras, stages, digital micromirror devices, optogenetic stimulators) that allows experimental logic to be independent of specific hardware choices.
- **Algorithm registry:** A plugin system for implementing custom decision logic, from simple threshold-based triggers to machine learning models.
- **Configuration-based deployment:** YAML-based hardware and experiment specifications validated by Pydantic models, supporting reproducibility and protocol sharing across labs.
- **User interfaces:** Both headless operation for automated experiments and graphical interfaces for interactive control.

The framework is written entirely in Python, making it accessible to the large community of scientists already using Python for data analysis. This choice also provides access to the scientific computing ecosystem (NumPy, SciPy, scikit-learn, PyTorch) for implementing sophisticated analysis and decision algorithms.

![System Overview](diagrams/system_overview.svg)
**Figure 1.** System overview of CLEF. YAML configuration files are validated by Pydantic models and passed to the ClosedLoopEngine, which orchestrates the real-time acquisition loop. The engine coordinates three subsystems: a HardwareManager that provides data through swappable backends, an Algorithm that processes each sample and detects trigger conditions, and a StimulusController that delivers perturbations to the preparation.

## Architecture

CLEF follows a layered architecture where independent components communicate through well-defined interfaces. This design allows researchers to customize parts of the system (adding new hardware, implementing new algorithms) without understanding the entire codebase.

- **CLI entry point** (`cli/clef_cli.py`) parses YAML configs validated by Pydantic models.
- **ConfigManager** (`config/config_manager.py`) loads, validates, and merges three configuration types (hardware, experiment, algorithm) with sensible defaults. Pydantic validation catches invalid parameters before any hardware is initialized.
- **ClosedLoopEngine** (`engine/closed_loop_engine.py`) orchestrates the real-time loop: initialize hardware, algorithm, and stimulus controller, then iterate sample-by-sample through the acquisition.
- **HardwareManager** (`hardware/hardware_manager.py`) selects a backend implementing `BaseHardwareBackend` and exposes a generic `DataInterface` for sampling. Available backends include `dummy` (testing), `pycromanager`/`pymmcore` (real microscope hardware via Micro-Manager), `lorenz_demo`, `ring_attractor_demo`, and `screenshot`.
- **Algorithm registry** (`algorithms/algorithm_factory.py`) uses a factory pattern to instantiate pluggable algorithms that implement `process_sample()` and `check_stim()`.
- **Stimulus controllers** (`hardware/stimulus_controllers/`) abstract hardware-specific actuation behind `StimulusInterface`, supporting dummy, polygon (DMD), and demo-specific controllers.

![Class Hierarchy](diagrams/class_hierarchy.svg)
**Figure 2.** Class hierarchy of CLEF's core abstractions. Each subsystem defines an abstract base class (BaseHardwareBackend, BaseAlgorithm, BaseStimulusController) with concrete implementations selected at runtime via configuration. The HardwareManager exposes a generic DataInterface, decoupling algorithms from specific data sources. The ClosedLoopEngine composes all three subsystems to run the acquisition loop.

### Hardware Abstraction Layer

The hardware abstraction layer provides consistent interfaces for common microscopy components:

- **Camera Interface:** Supports multiple backends including Micro-Manager/pycromanager, direct camera SDKs, and custom acquisition systems. The interface handles acquisition parameters (exposure time, binning, ROI), triggering modes, and streaming data retrieval.
- **Stage Interface:** Controls motorized stages for sample positioning. Supports multi-axis stages with both absolute positioning and relative moves, and handles coordinate system transformations between logical and physical coordinates.
- **Stimulator Interface:** Manages multi-state devices for optical or electrical stimulation. For optogenetic experiments, this includes controlling laser power, shutter timing, and spatial patterning through devices like digital micromirror devices (DMDs). The interface supports complex temporal patterns and precise timing coordination with acquisition.
- **Backend System:** Hardware devices are grouped into backends that bundle a collection of devices behind a common API. Most commonly this interfaces with Micro-Manager through pycromanager, but the abstraction layer allows other control systems to be used without changing experimental code.

All hardware components are configured through YAML specifications that define device types, connection parameters, and default settings. The same experimental protocol can run on different hardware setups by swapping configuration files.

### Micro-Manager Integration via pycro-manager

CLEF's primary hardware backend interfaces with microscope hardware through pycro-manager (Pinkard et al., 2021) and pymmcore, the Python bindings for Micro-Manager (Edelstein et al., 2014). Micro-Manager provides device adapters for hundreds of microscopy components (cameras, stages, filter wheels, light sources, shutters), and pycro-manager exposes this functionality to Python with support for multi-dimensional acquisition, hardware-triggered sequencing, and direct access to the Micro-Manager Core API.

By building on this ecosystem, CLEF inherits broad hardware compatibility without implementing device-specific drivers. CLEF runs alongside a Micro-Manager instance, sending commands and receiving data through pycromanager's Python API. Researchers can use familiar Micro-Manager configurations and device adapters without modification. The `MicroManagerBackend` wraps these APIs behind CLEF's `BaseHardwareBackend` interface, so that the rest of the framework (engine, algorithms, stimulus controllers) remains agnostic to the underlying hardware communication layer.

### Algorithm Framework

CLEF's algorithm system lets researchers implement custom decision logic through a plugin interface. Each algorithm is a Python class that implements:

- `initialize_model()`: Sets up any required state before the experiment starts.
- `process_sample(sample)`: Receives data from the current acquisition cycle and returns decisions (e.g., `{"triggered": bool, "trigger_value": float}`).
- Cleanup and teardown as needed after the experiment completes.

The `process_sample()` method receives both raw data (images, time series) and context about the experiment state. This allows algorithms to maintain state across multiple calls and make decisions based on experimental history.

Once registered with the algorithm factory via a string name, the framework handles all data routing, timing, and coordination automatically. New analysis methods are added by implementing the interface and registering a name — no framework code changes are required.

## Key Design Features

| Feature | Description |
|---------|-------------|
| **Config-driven reproducibility** | Entire experiments are specified in version-controllable YAML files. Pydantic validation catches errors before acquisition begins. |
| **Hardware abstraction** | Swappable backends (dummy for testing, pycro-manager/pymmcore for real microscopes, demo generators) behind a uniform `DataInterface`. |
| **Pluggable algorithms** | Registry-based factory pattern. New analysis methods are added by implementing `initialize_model()` and `process_sample()`, then registering a string name. |
| **Generic data interface** | Supports uint16 microscopy volumes, RGB screenshots, 1-D timeseries, and other modalities. Algorithms are not locked to a single data type. |
| **Testability** | Dummy and demo backends enable full integration testing without physical hardware, with 377+ automated tests covering the complete pipeline from config loading through stimulus delivery. |

## Configuration and Deployment

CLEF uses a configuration-driven approach where all experimental parameters and hardware specifications are defined in YAML files. These configurations are validated against Pydantic models that ensure type correctness and provide clear documentation of available options.

A typical experiment configuration specifies:

- Hardware devices and their parameters
- Data acquisition settings (timing, resolution, regions of interest)
- Algorithm selection and parameters
- Stimulus protocols
- Storage locations and metadata

This approach provides several benefits. The same experimental logic can run on different hardware by swapping hardware configuration files. Configurations can be version-controlled alongside analysis code. Sharing protocols between labs requires sharing configuration files rather than reimplementing code.

For deployment, CLEF is packaged as a Docker application. The entire software stack (CLEF, dependencies, system libraries) is bundled in a container that runs identically across different operating systems and hardware configurations. Researchers can deploy CLEF by cloning the repository, installing Docker, and running a single command. This containerization removes installation complexity and ensures reproducibility.

## Data Model and Storage

CLEF's data model is designed to work with minimo (Borchardt et al., 2021), a linked data and metadata storage system that combines object storage (MinIO) for large raw data files with a document database (MongoDB) for metadata. This architecture keeps big, immutable raw data in efficient object storage while maintaining rich, searchable metadata in a database.

When CLEF saves experimental data, it stores:

- Raw images or time series data as immutable objects
- Acquisition parameters and hardware state
- Algorithm decisions and stimulus timing
- Experiment configuration files
- Processing logs and provenance information

All metadata are linked to the corresponding raw data objects, ensuring that context is never lost. This makes data findable, understandable, and reusable long after collection. The data model supports both programmatic access (for automated analysis pipelines) and interactive browsing through minimo's web interface.

## Demos

CLEF ships with several demo configurations that illustrate the framework's capabilities across a range of use cases, from synthetic dynamical systems to real microscope hardware.

### Ring Attractor Demo

The ring attractor demo underscores the core conceptual motivation for CLEF. Closed-loop experimental frameworks exist to interrogate dynamical biological processes, and brains are perhaps the most compelling example. In this demo, a synthetic dynamical system (a bistable ring attractor with two concentric limit cycles) produces time-varying data, and the experimenter delivers perturbations to understand how the system works. It is, in essence, a game: the system evolves according to its own dynamics, and the experimenter must observe, hypothesize, and intervene in real time.

The ring attractor backend generates 2D image frames depicting a punctum orbiting on one of two concentric rings. The system state is defined by an angular position and a radial mode (inner or outer ring). The experimenter can deliver stimuli that perturb the angular position or toggle the system between rings. An auto-trigger mode is also available, where the algorithm fires stimuli when the system enters a specified angular region. This demo illustrates how CLEF decouples data generation, online analysis, and stimulus control into independently configurable components.

![Ring Attractor Demo](diagrams/screenshot_ring_attractor.png)
**Figure 3.** Screenshot of the ring attractor demo during a live session. The visualization shows the punctum orbiting on one of two concentric rings, with a fading trajectory trail indicating recent history. The experimenter can deliver angular perturbations or toggle the system between rings using the GUI controls, observing the effect of each intervention in real time.

**Configs:** `config/demo/demo_ring_attractor_hardware.yaml`, `demo_ring_attractor_experiment.yaml`, `demo_ring_attractor_algorithm.yaml`

### Lorenz Attractor Demo

The Lorenz attractor demo uses a classic chaotic dynamical system (Lorenz, 1963) as a synthetic data source. The backend integrates the Lorenz equations in real time and renders the 3D state as a 2D image (a bright punctum on a 100x100 pixel field). The algorithm extracts the system's phase-space coordinates and fires a stimulus when the state enters a configurable trigger volume. The stimulus applies a perturbation vector to the Lorenz state variables, visibly deflecting the trajectory.

This demo is useful for validating the full closed-loop pipeline (data acquisition, algorithm processing, trigger detection, stimulus delivery) without any physical hardware, and for illustrating how CLEF handles chaotic systems where stimulus timing relative to system state is critical.

**Configs:** `config/demo/demo_lorenz_hardware.yaml`, `demo_lorenz_experiment.yaml`, `demo_lorenz_algorithm.yaml`

### Brainalyzer Demo

The Brainalyzer demo provides the same GUI and analysis interface as the physical hardware configuration, but reads data from an existing volumetric calcium imaging dataset (a TIFF stack) rather than acquiring live from a microscope. This allows users to develop, test, and refine their analysis pipelines against real neural data without requiring access to microscope hardware. The algorithm performs real-time quantification of neural activity across z-planes and supports stimulus parameter exploration through the interactive GUI.

![Brainalyzer Demo](diagrams/screenshot_brainalyzer.png)
**Figure 4.** Screenshot of the Brainalyzer demo GUI. The interface displays volumetric calcium imaging data read from a TIFF stack, with real-time quantification of neural activity across z-planes. This demo provides the same analysis and stimulus control interface as the physical hardware configuration, allowing algorithm development and parameter exploration without a connected microscope.

**Configs:** `config/demo/demo_brainalyzer_hardware.yaml`, `demo_brainalyzer_experiment.yaml`, `demo_brainalyzer_algorithm.yaml`

### Physical Hardware Demo

The physical hardware demo provides an example of CLEF running a full volumetric calcium imaging experiment with real-time quantification and patterned optogenetic illumination using the Mightex Polygon1000 digital micromirror device (DMD). This demo can only be run with the appropriate microscope hardware configured (camera, stage, Micro-Manager device adapters, and Polygon1000). It represents the primary production use case for CLEF: acquiring volumetric calcium imaging data, processing it online, and delivering spatially patterned optogenetic stimuli in closed loop.

### Why These Demos

Each demo was chosen to highlight a different aspect of the framework:

- The **ring attractor** and **Lorenz** demos demonstrate that CLEF's architecture is not specific to microscopy. Any system that produces sequential data samples and accepts perturbations can be wrapped as a backend. These demos also allow new users to explore the full closed-loop workflow on any machine, with no hardware dependencies.
- The **Brainalyzer** demo bridges the gap between synthetic and real data, enabling algorithm development against genuine neural recordings without tying up microscope time.
- The **physical hardware** demo shows the end-to-end production configuration, demonstrating CLEF's integration with Micro-Manager, pycro-manager, and real optical hardware.

## Example Implementation: Whole-Brain Closed-Loop Imaging in *C. elegans*

The first complete CLEF implementation was developed for closed-loop whole-brain imaging in *C. elegans*. This system integrates volumetric calcium imaging with optogenetic stimulation, allowing targeted perturbations based on observed neural dynamics.

### System Capabilities

The system captures neural activity across the entire *C. elegans* nervous system (302 neurons) at cellular resolution while the animal is restrained in a microfluidic chip. Real-time image processing identifies and tracks individual neurons, extracts fluorescence signals indicating neural activity, and detects specific patterns across neural populations. Based on these observations, optogenetic stimulation is delivered with spatial precision (targeting specific neurons via DMD) and millisecond temporal precision.

The system provides two operational modes:

- **Interactive mode:** A graphical interface displays real-time visualizations of imaging data, processed signals, and system state. Researchers can monitor experiments, manually trigger stimulation for testing, and adjust parameters on the fly. The interface includes panels showing raw images, segmented neurons, extracted activity traces, and low-dimensional projections of population dynamics.
- **Headless mode:** For automated experiments that do not require real-time monitoring, the system runs without a graphical interface, reducing computational overhead and enabling longer unattended experiments. All data are still logged and available for later review.

The configuration system allows the same experimental logic to run in either mode, making it easy to prototype interactively and then deploy for automated data collection.

### Data Pipeline

The closed-loop pipeline operates in several stages:

1. **Volumetric acquisition:** The microscope captures z-stacks at rates up to 5 volumes per second. Piezoelectric focus control synchronizes with camera exposures to capture optical sections through the depth of the animal.
2. **Image processing:** Each volume undergoes motion correction, background subtraction, and noise filtering using optimized implementations (Numba-compiled functions, OpenCV routines) to maintain throughput.
3. **Neuron identification:** Segmentation algorithms identify individual neurons in each volume based on fluorescence reporter expression. The system tracks neurons across volumes to maintain consistent identities despite animal movement.
4. **Activity extraction:** For each identified neuron, fluorescence time series are extracted and processed (baseline correction, detrending) to isolate activity signals from imaging artifacts.
5. **Feature analysis:** Dimensionality reduction and pattern detection extract relevant features from population activity, including principal component analysis, threshold-based event detection, and correlation analysis.
6. **Decision logic:** Based on extracted features and the current experimental context, the algorithm determines whether to trigger stimulation and with what parameters.
7. **Stimulus delivery:** If stimulation is triggered, spatial patterns are configured on the DMD and laser timing is controlled to illuminate specific neurons, synchronized with acquisition to avoid crosstalk.

The entire pipeline completes in under 100 milliseconds, allowing the system to respond to neural dynamics on timescales relevant to behavior.

### Applications

This implementation has enabled several types of experiments:

- **State-dependent perturbations:** Monitoring population activity to identify specific neural states, then triggering optogenetic stimulation when those states occur. This allows causal testing of how neural states influence behavior or future dynamics.
- **Pattern-triggered interventions:** Detecting specific patterns of co-activity across neurons to perturb the network at precise moments in its operation, revealing how temporal coordination contributes to circuit function.
- **Closed-loop behavioral control:** Maintaining animals in particular behavioral states by adjusting sensory stimuli based on neural activity, enabling study of how neural dynamics relate to behavior while controlling for state-dependent effects.

These experiments would be difficult or impossible with open-loop protocols because the relevant neural states occur at unpredictable times and last for short durations.

## Additional Use Cases

CLEF's modular design supports a range of closed-loop experimental paradigms beyond whole-brain imaging in *C. elegans*.

- **Multi-region recording with targeted stimulation:** The modular architecture allows different cameras to record different brain areas while a central algorithm analyzes across all streams to make stimulation decisions, supporting experiments testing inter-regional communication.
- **Adaptive behavioral testing:** CLEF can monitor both neural and behavioral readouts simultaneously and use both to guide experimental logic, adjusting sensory stimuli to maintain animals in specific behavioral states or to test state-dependent perturbations.
- **Algorithm development and testing:** Because experimental logic is encapsulated in pluggable algorithms, researchers can implement new approaches, test them in simulation using recorded data, then deploy the same code in live experiments. The configuration system makes it easy to compare different algorithms under identical experimental conditions.

## Related Projects

| Tool | Language | Domain | Closed-Loop | Config-Driven | Reference |
|------|----------|--------|-------------|---------------|-----------|
| **Bonsai** | C# | General experiment control | Yes | No (visual XML workflows) | Lopes et al., 2015 |
| **Open Ephys GUI** | C++ | Electrophysiology | Yes | No (GUI) | Siegle et al., 2017 |
| **CaImAn** | Python | Calcium imaging analysis | Partial (OnACID) | No | Giovannucci et al., 2019 |
| **Suite2p** | Python | Calcium imaging analysis | No (offline) | No | Pachitariu et al., 2017 |
| **Micro-Manager** | Java/C++ | Microscope control | No | Partial | Edelstein et al., 2014 |
| **pycro-manager** | Python/Java | Microscope control (Python) | No | No | Pinkard et al., 2021 |
| **ScanImage** | MATLAB | Two-photon microscopy | Partial | No | Pologruto et al., 2003 |
| **Stytra** | Python | Zebrafish behavior + light-sheet | Yes | Partial | Stih et al., 2019 |
| **RTXI** | C++ | Hard real-time electrophysiology | Yes | No | Patel et al., 2017 |
| **ACQ4** | Python | Patch-clamp + imaging | Partial | No | Campagnola et al., 2014 |
| **Autopilot** | Python | Distributed behavioral rigs | Yes | Yes | Saunders & Wehr, 2019 |
| **LabVIEW** | G (visual) | General instrument control | Partial | No | National Instruments |

### What Makes CLEF Unique

1. **Python-native closed-loop for volumetric calcium imaging.** No other Python framework combines real-time volumetric calcium imaging acquisition with online analysis and stimulus feedback. CaImAn provides algorithms but not orchestration; Stytra targets zebrafish light-sheet specifically; pycro-manager provides microscope control but not closed-loop logic.

2. **Declarative, validated configuration.** YAML files validated by Pydantic models mean experiments are fully specified in version-controllable text files. Invalid configurations are caught before hardware is initialized. This contrasts with Bonsai's visual XML workflows, ScanImage's MATLAB scripts, and the ad-hoc parameter passing common in lab code.

3. **Swappable hardware backends with a generic data interface.** The `DataInterface` abstraction decouples algorithms from data sources. The same algorithm code runs against a dummy backend in CI, a Lorenz attractor demo on a laptop, or a real microscope in the lab. This is made possible by building on Micro-Manager and pycro-manager for hardware communication while abstracting the interface one level higher.

4. **Algorithm registry pattern.** New online analysis methods are added by implementing `initialize_model()` and `process_sample()`, then registering a string name. No framework code changes are required.

5. **Full testability without hardware.** The dummy backend and demo backends (Lorenz, ring attractor) enable 377+ automated tests covering the complete pipeline from config loading through stimulus delivery.

6. **Accessibility through Python.** CLEF is written entirely in Python, making it immediately accessible to the large community of neuroscientists already using Python for data analysis. Researchers can leverage familiar tools (NumPy, SciPy, scikit-learn, PyTorch) directly within their experimental logic, lowering the barrier between offline analysis development and online deployment.

## Availability and Documentation

CLEF is open-source software released under the MIT license. The code is available on GitHub at [https://github.com/focolab/clef](https://github.com/focolab/clef). Documentation includes installation instructions, configuration guides, algorithm development tutorials, and example experiments.

We encourage contributions from the community. The modular architecture makes it straightforward to add support for new hardware devices or implement new algorithm types. Issues and pull requests are welcome on the GitHub repository.

## Acknowledgements

We thank members of the Kato lab for valuable discussion.

Data for this study were acquired at the UCSF Innovation Core at the Weill Institute for Neurosciences on a custom imaging system controlled by the open-source software Micro-Manager.

This work was supported by NIH grants NS115572 (R.L.D), R35GM124735 (S.K.), and the Weill Institute for Neurosciences (S.K.).

## References

- Borchardt, J., Dunn, R. & Kato, S. "minimo: a linked data and metadata storage system for small labs." *J. Open Source Softw.* 6, 2979 (2021). DOI: 10.21105/joss.02979
- Campagnola, L., Kratz, M.B. & Bhatt, D.B. "ACQ4: An open-source software platform for data acquisition and analysis in neurophysiology research." *Front. Neuroinform.* 8, 3 (2014). DOI: 10.3389/fninf.2014.00003
- Edelstein, A.D., Tsuchida, M.A., Amodaj, N., Pinkard, H., Vale, R.D. & Stuurman, N. "Advanced methods of microscope control using μManager software." *J. Biol. Methods* 1, e10 (2014). DOI: 10.14440/jbm.2014.36
- Giovannucci, A. et al. "CaImAn: An open source tool for scalable calcium imaging data analysis." *eLife* 8, e38173 (2019). DOI: 10.7554/eLife.38173
- Grosenick, L., Marshel, J.H. & Deisseroth, K. "Closed-loop and activity-guided optogenetic control." *Neuron* 86, 106-139 (2015). DOI: 10.1016/j.neuron.2015.03.034
- Lopes, G. et al. "Bonsai: An event-driven framework for processing and controlling data streams." *Front. Neuroinform.* 9, 7 (2015). DOI: 10.3389/fninf.2015.00007
- Lorenz, E.N. "Deterministic nonperiodic flow." *J. Atmos. Sci.* 20, 130-141 (1963). DOI: 10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2
- Newman, J.P. et al. "Optogenetic feedback control of neural activity." *eLife* 4, e07192 (2015). DOI: 10.7554/eLife.07192
- Pachitariu, M. et al. "Suite2p: beyond 10,000 neurons with standard two-photon microscopy." *bioRxiv* (2017). DOI: 10.1101/061507
- Packer, A.M., Russell, L.E., Dalgleish, H.W.P. & Hausser, M. "Simultaneous all-optical manipulation and recording of neural circuit activity with cellular resolution in vivo." *Nat. Methods* 12, 140-146 (2015). DOI: 10.1038/nmeth.3217
- Pinkard, H., Stuurman, N., Ivanov, I.E., Anthony, N.M., Ouyang, W., Li, B., Yang, B., Tsuchida, M.A., Chhun, B., Zhang, G., Mei, R., Anderson, M., Shepherd, D.P., Hunt-Isaak, I., Dunn, R.L., Jahr, W., Kato, S., Royer, L.A., Thiagarajah, J.R., . . . Waller, L. "Pycro-Manager: open-source software for customized and reproducible microscope control." *Nat. Methods* 18, 226-228 (2021). DOI: 10.1038/s41592-021-01087-6
- Pologruto, T.A., Sabatini, B.L. & Bhatt, D.B. "ScanImage: Flexible software for operating laser scanning microscopes." *BioMed. Eng. Online* 2, 13 (2003). DOI: 10.1186/1475-925X-2-13
- Saunders, J. & Wehr, M. "Autopilot: Automating behavioral experiments with lots of Raspberry Pis." *bioRxiv* (2019). DOI: 10.1101/807693
- Siegle, J.H. et al. "Open Ephys: An open-source, plugin-based platform for multichannel electrophysiology." *J. Neural Eng.* 14, 045003 (2017). DOI: 10.1088/1741-2552/aa5eea
- Stih, V., Petrucco, L., Kist, A.M. & Portugues, R. "Stytra: An open-source, integrated system for stimulation, tracking and closed-loop behavioral experiments." *PLOS Comput. Biol.* 15, e1006699 (2019). DOI: 10.1371/journal.pcbi.1006699
