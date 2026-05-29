---
title: "CLEF: A Python Framework for Closed-Loop Neuroscience Experiments"
tags:
  - Python
  - neuroscience
  - closed-loop
  - microscopy
  - calcium imaging
  - optogenetics
  - real-time
authors:
  - name: Raymond L. Dunn
    orcid: 0000-0003-4443-5519
    affiliation: 1
  - name: Saul Kato
    orcid: 0000-0003-2990-8306
    affiliation: 1
    corresponding: true
affiliations:
  - name: Department of Neurology, Weill Institute for Neurosciences, University of California, San Francisco, CA, USA
    index: 1
date: 29 May 2026
bibliography: paper.bib
---

# Summary

CLEF (Closed-Loop Experimental Framework) is an open-source Python platform for real-time closed-loop experiments with automated stimulus control. This first CLEF release targets microscopic imaging modalities for neuroscience researchers. CLEF enables researchers to observe neural activity or organism behavior, process real-time data according to pre-defined user-specified algorithms, and deliver stimuli in a single feedback loop governed entirely by declarative configuration files. CLEF also provides a GUI interface for live monitoring and controlling experiments.

The core philosophy behind CLEF is that a flexible framework will allow rapid design of custom experiments that naturally scale. CLEF replaces ad-hoc lab scripts and GUI-only workflows with a modular, config-driven architecture where hardware backends, real-time analysis algorithms, and stimulus controllers are interchangeable components selected at runtime via YAML. Think of it as an agentic toolkit for real-world experiments.

# Statement of Need

## Closed-Loop Experimentation for Advancing Neuroscience

Traditional neuroscience experiments typically follow a fixed, open-loop protocol: researchers design the experiment, set parameters, collect data subject to a fixed stimulus protocol, and analyze results afterward. This approach has yielded insight for many questions, but it is fundamentally impoverished approach to how dynamical biological systems operate. Brains are highly recurrent networks where activity patterns influence future states in complex ways. To understand causal relationships in these systems, we need experiments that can perturbatively and adaptively probe a system, responsive to the evolving system state itself.

Closed-loop experimental design allows real-time adaptation of stimulus protocols based on ongoing measurements. The value of this approach has been recognized across multiple areas of study and model systems [@grosenick2015]. As measurement technologies scale to capture hundreds or thousands of neurons simultaneously, closed-loop methods become increasingly important for understanding network-level mechanisms of brain function.

## The Need for Flexible Automation Tools

Closed-loop experiments require coordinated control of multiple hardware components (cameras, stages, stimulation devices) while performing low-latency, real-time computation on streaming data. The computational pipeline must extract features from raw measurements, decide what to do based on the current state of the interrogated system, and execute stimulus protocols with precise timing. Building this coordination layer from scratch for every new experiment is a barrier to wider adoption of closed-loop methods.

Existing tools cover individual parts of the problem. While Python is the predominant programming language in the biosciences, many established frameworks are written in non-Python languages: Bonsai in C# [@lopes2015], the Open Ephys GUI in C++ [@siegle2017], and ScanImage in MATLAB [@pologruto2003]. Several otherwise-Python tools target a different domain or stage of the workflow: Open Ephys focuses on electrophysiology rather than imaging [@siegle2017], while CaImAn [@giovannucci2019] and Suite2p [@pachitariu2017] provide offline calcium-imaging analysis without experiment orchestration. Tools such as Stytra [@stih2019], ACQ4 [@campagnola2014], and Autopilot [@saunders2019] provide closed-loop control but are specialized to particular rigs or behavioral paradigms. Commercial microscopy software exposes limited scripting that works for simple automated protocols but lacks the sophistication needed for responsive, state-dependent experiments. Pycro-Manager [@pinkard2021] established convenient Python-based microscope control on top of Micro-Manager [@edelstein2014] and enabled more complex acquisition sequences, but it does not, on its own, supply the architecture needed for closed-loop experimentation with multiple hardware components and real-time decision-making outside of the Micro-Manager ecosystem. The CLEF framework is written entirely in Python, so it is accessible to the large community of scientists already using Python. Researchers can use NumPy [@harris2020], SciPy [@virtanen2020], scikit-learn [@pedregosa2011], and PyTorch [@paszke2019] directly inside their experimental logic, as well as any other code libraries.

# Architecture

CLEF uses a layered architecture in which independent components communicate through simple interfaces. Researchers can extend parts of the system (adding new hardware, implementing new logic algorithms) for their individual application without needing to modify the rest of the codebase.

## Key Concepts

CLEF is organized around four concepts used throughout the rest of this manuscript:

| Concept   | Purpose                                                                                                                                                                                                                                                          |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `device`  | Components or endpoints that CLEF interacts with. A `device` is either an `input_device` or an `output_device`. Input devices provide data streams (e.g. camera images), and output devices are what the experimenter wants to control (e.g. a stage or a laser). |
| `logic`   | A control algorithm. It processes samples from the input devices and emits updates for the output devices.                                                                                                                                                       |
| `session` | Contextual metadata for an experiment. For example, information about the data subject (cell line, treatment condition), highly specific to the application.                                                                                                      |
| `engine`  | A discrete event loop that orchestrates iterations of data sampling, data processing, and actuation.                                                                                                                                                             |

An experiment can use one or more input devices and one or more output devices: multiple cameras, a camera plus a stage readout, or a DMD (digital micromirror device) plus a laser plus a stage. At runtime, the engine reads from every registered input device, passes the samples to the logic algorithm, and dispatches output commands to any subset of the registered output devices.

![System overview of CLEF. Three YAML configuration files (`io.yaml`, `logic.yaml`, `session.yaml`) are validated by Pydantic models and passed to the ClosedLoopEngine, which orchestrates the real-time acquisition loop. The engine reads samples from one or more registered `input_devices`, passes data samples to the active `logic` algorithms, and dispatches return values to one or more named `output_devices`. The `engine` collects per-device and per-algorithm metadata generated during a `session`.\label{fig:overview}](media/clef_architecture.svg)

## Code Architecture

The repository is split into two top-level trees. `core/` contains the framework internals: base classes, managers, the engine, configuration logic, and the CLI. `apps/` contains concrete implementations that register against the core base classes via plugin discovery, plus per-application YAML configs.

- **CLI entry point** (`core/utils/clef_cli.py`) accepts either a preset name (e.g. `clef <app_name>`) that resolves to a directory under `apps/config/`, or explicit `--session`, `--io`, and `--logic` paths.
- **ClosedLoopEngine** (`core/engine/closed_loop_engine.py`) owns an `IOManager` and a `LogicManager` and runs the sample loop: acquire from input devices, pass to logic, dispatch any returned commands to output devices.
- **ConfigManager** (`core/config/config_manager.py`) loads, validates, and merges the three configuration files against Pydantic models (`SessionConfig`, `IOConfig`, `ClosedLoopLogicConfig`), with defaults in `core/config/defaults/`. Validation catches invalid parameters before any hardware is initialized.
- **IOManager** (`core/io/`) instantiates `BaseInputDevice` and `BaseOutputDevice` subclasses listed in `io.yaml`. Each input device is paired with a `BaseDataInterface` that controls how the data stream is buffered, structured, and persisted at the end of the session.
- **LogicManager** (`core/logic/`) instantiates the `BaseClosedLoopLogic` subclass selected by `logic.yaml` and connects it to the input/output devices held by the IOManager.

At runtime, the data flow is symmetric and minimal:

```
input_devices → engine → closed-loop logic → engine → output_devices
```

# Customizing CLEF for your New Experiment

A new closed-loop experiment is built by writing (or reusing) one or more input devices, one or more output devices, one or more logic algorithms, and three configuration files. The four base abstractions (`BaseInputDevice`, `BaseOutputDevice`, `BaseDataInterface`, and `BaseClosedLoopLogic`) each define a small interface that researchers fill in for their own hardware and analysis. A single experiment can fan in samples from several cameras, electrode signals, or stage readouts, and fan out to multiple actuators (a DMD plus a light source plus a stage, for example). The customization workflow follows these steps:

1. **Create an input device.** Subclass `BaseInputDevice` to wrap any source of streaming data (a camera, an electrode amplifier, a stage readout, a TIFF stack), implementing `connect()`, `configure()`, `_get_input()`, and `close()`.
2. **Create an output device.** Subclass `BaseOutputDevice` to wrap any actuator (an LED, a DMD, a motorized stage), implementing `connect()`, `configure()`, `_update_output(**kwargs)`, and `close()`.
3. **Create a closed-loop logic algorithm.** Subclass `BaseClosedLoopLogic` to implement the experiment's online analysis and decision rules via `initialize_model()`, `process_sample(sample)`, `_check_logic()`, and `close()`.
4. **Pair input devices with a data interface (optional).** Subclass `BaseDataInterface` to separate *how a sample looks* from *how it is stored*, defining sample shape, dtype, and end-of-session serialization. New modalities can be added with no engine changes, and the per-session output bundle maps cleanly onto storage systems such as minimo [@borchardt2021].
5. **Write the three configuration files.** `io.yaml` lists the input and output devices and their parameters; `logic.yaml` selects the algorithm and its tunable parameters; `session.yaml` describes the run itself (operator, subject, conditions, output location, duration). All three are validated by Pydantic at startup, before any hardware is initialized.
6. **Run the experiment.** The CLI shorthand `clef <app_name>` resolves to `apps/config/<app_name>/` and loads all three files. At the end of every session, CLEF writes per-device data files plus a single JSON metadata file capturing the merged configuration, every output event with its timestamp, and per-device sample timestamps.

Full descriptions of each base class's interface, the registry mechanism, configuration schemas, and the CLI are provided in the online documentation.

# Demos

CLEF ships with several demo configurations covering a range of use cases, including synthetic dynamical systems, recorded data playback, and real microscope hardware. Each demo lives in its own directory under `apps/config/`.

## Limit Cycle Demo (`apps/config/limit_cycle/`)

The limit cycle demo captures the core conceptual motivation for CLEF. Closed-loop experimental frameworks exist to interrogate dynamical biological processes, and brains are perhaps the most compelling example. In this demo, a virtual live dynamical system (a bistable system with two concentric limit cycles) produces time-varying activity, and the experimenter delivers perturbations to understand how the system works. The setup is essentially a game: the system evolves according to its own hidden dynamics, and the experimenter must observe, hypothesize, and intervene in real time to discern how these dynamics are structured.

The `limit_cycle_input` device generates 2-D image frames showing a punctum orbiting on one of two concentric rings. The system state is defined by an angular position and a radial mode (inner or outer ring). The `limit_cycle_output` device delivers stimuli that perturb the angular position or toggle the system between rings. An auto-trigger mode is available, in which the logic fires stimuli when the system enters a specified angular region. The demo illustrates how CLEF separates data generation, online analysis, and stimulus control into independently configurable components.

![Screenshot of the limit cycle demo during a live session. The visualization shows the punctum orbiting on one of two concentric rings, with a fading trajectory trail indicating recent history. The experimenter can deliver angular perturbations or toggle the system between rings using the GUI controls, observing the effect of each intervention in real time.\label{fig:limitcycle}](media/screenshot_limit_cycle.png)

## Recording Playback Demo (`apps/config/recording_playback/`)

The recording playback demo provides the same GUI and analysis interface as the physical hardware configuration, but reads data from an existing volumetric calcium imaging dataset (a TIFF stack) via `recording_playback_input` rather than acquiring live from a microscope. Users can develop, test, and refine their analysis pipelines against real neural data without needing access to microscope hardware. The `brainalyzer_logic` algorithm performs real-time quantification of neural activity across z-planes (offloaded to `brainalyzer_worker.py` over shared memory) and supports stimulus parameter exploration through the interactive GUI. This demo illustrates how application-specific workflows can be integrated and tested with CLEF prior to deployment.

![Screenshot of the "brainalyzer" pipeline running over the recording-playback demo. The interface displays volumetric calcium imaging data read from a TIFF stack, with real-time quantification of neural activity across z-planes. This demo provides the same analysis and stimulus control interface as the physical hardware configuration, allowing algorithm development and parameter exploration without a connected microscope.\label{fig:brainalyzer}](media/screenshot_brainalyzer.png)

## Speech BCI Demo (`apps/config/speech_bci/`)

The speech BCI demo illustrates the network-service logic pattern and also is an example of CLEF for 1-D timeseries data (electrophysiology of spiking neurons). `speech_bci_input` provides neural spiking data, the logic algorithm streams it over WebSocket to a FastAPI decoder server (`apps/subprocess/speech_bci_server.py`) that runs a GRU + n-gram language model inside Docker (locally or on a GCP Compute Engine VM via `docker/deploy_decoder.sh`), and the decoded text is rendered through `speech_bci_output`. Alternatively, for minimizing latency introduced by inter-process communication on systems where model subprocesses are hosted on the same hardware as data streams, zero-copy data transfer via shared-memory buffers is also implemented. The demo shows how CLEF supports workflows with deployed, standalone, networked models.

## Physical Hardware Demo (`apps/config/physical_hardware/`)

The physical hardware demo runs a full volumetric calcium imaging experiment with real-time quantification and patterned optogenetic illumination using the Mightex Polygon1000 digital micromirror device. It can only be run with the appropriate microscope hardware configured (camera, stage, Micro-Manager device adapters, and Polygon1000, 89North LDI LS). This is the primary production use case for CLEF: acquiring volumetric calcium imaging data, processing it online via the brainalyzer logic, and delivering spatially patterned optogenetic stimuli in a closed loop. We include this demo as an illustration of CLEF being applied daily in our own lab [@dunn2025].

# Availability and Documentation

CLEF is open-source software released under the MIT license. The code is available on GitHub at <https://github.com/focolab/clef>. Documentation includes installation instructions, configuration guides, a vibe-coding quickstart for AI-assisted app development, and the demo configurations described above.

We welcome contributions from the community. The modular architecture supports adding new hardware devices or new logic types without modifying framework code. Issues and pull requests can be opened on the GitHub repository.

# AI Usage Statement

AI coding assistants were used during development to facilitate refactoring of parts of the codebase, improve test coverage, and to facilitate manuscript citation formatting. All scientific content, architectural decisions, and final code were authored and reviewed by the listed authors.

# Acknowledgements

We thank members of the Kato lab for valuable discussion.

Data for this study were acquired at the UCSF Innovation Core at the Weill Institute for Neurosciences on a custom imaging system controlled by the open-source software packages Micro-Manager and Pycro-Manager.

This work was supported by NIH grants NS115572 (R.L.D.), R35GM124735 (S.K.), and the Weill Institute for Neurosciences (S.K.).
