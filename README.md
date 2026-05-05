# CLEF (Closed-Loop Experimental Framework)

Meet `clef`: A lightweight, modular framework for custom closed-loop control.

You can use `clef` to specify and run "closed-loop" experiments in neuroscience, biology or other physical sciences. It provides tooling for:

- Input: Data acquisition, such as cameras or electrodes.
- Output: Control and perturbation, such as optogenetic lasers, stimulus delivery, or mechanical actuators.
- Hardware systems: Such as the microscopy ecosystem Micro-Manager.
- Design of on-the-fly computational analysis and logic that that executes during experimental sessions.
- GUI-based human-in-the-loop monitoring and control of live experimental sessions.

`clef` is great for:

**Discovery**: The world is full of complex, dynamic, interacting processes. Think neurons in a network, or proteins in a cell. If you want to understand causality in these kind of recurrent and interconnected systems, you need closed-loop experimental design. You can use `clef` to implement these experiments.

**Adaptation**: `clef` lets you adjust your acquisition system in response to real-time data. If your quality control metrics degrade, you can use `clef` to apply the appropriate adjustments, or resample the data under better conditions.

# Installation

```bash
git clone https://github.com/focolab/clef
cd clef
```

**Without extras** — installs only the base framework (core classes, config, engine):

```bash
pip install -e .
```

**`demos`** — adds everything needed to run the limit cycle, recording playback, screenshot, and speechbci demos:

```bash
pip install -e ".[demos]"
```

**`all`** — full install including hardware integrations and all extras:

```bash
pip install -e ".[all]"
```

# Quickstart: play with the demos

Install with the `demos` extra (see above), then run:

```bash
clef limit_cycle
```

`clef <name>` searches `apps/config/` for a directory named `limit_cycle`, finds the three config files inside it (`session_config.yaml`, `io_config.yaml`, `logic_config.yaml`), validates them, and prompts before starting the loop. Other available demos:

```bash
# Playback an existing .tiff recording you already have, for model workflow development
clef recording_playback

# For a physical hardware demo (requires Mightex Polygon 1000 configured in Micro-Manager)
clef hardware_physical

# Have a ML model from brain2text? visualize it here (requires setting up your own model first)
clef speech_bci
```

To validate configs without running:

```bash
clef limit_cycle --validate-config
```

To pass config files explicitly:

```bash
clef --session s.yaml --io io.yaml --logic l.yaml
```

# Setup: Customizing `clef` for your experiments

## Key Concepts

If you're thinking of applying `clef` to solve your problem, see if it decomposes into the following Key Concepts that `clef` is organized around:

| Keyword   | Purpose                                                                                                                                                                                                                                                                           |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `device`  | Components or endpoints that you want `clef` to interact with. They can be an `input_device` or an `output_device`. Input devices provide data streams (e.g. camera → images), and output devices are what you want to adapt/control (e.g. z-stage, light source, or a solenoid). |
| `logic`   | Your control algorithms, which process samples from your input device data streams and emit updates for your output devices.                                                                                                                                                      |
| `session` | Essential contextual metadata about a `clef` experiment. For example about your data subject (a cell line, treatment condition, etc), highly specific to your application.                                                                                                        |
| `engine`  | A discrete event loop that orchestrates iterations of data sampling, data processing, and actuation.                                                                                                                                                                              |

At runtime, the engine reads from input devices, passes samples to the logic algorithm, and dispatches output commands to output devices:

```
input_devices → engine → closed-loop logic → engine → output_devices
```

## Step 1: Create a module for your input device

Say you want to connect to a new camera. If it's compatible with Micro-Manager, you can use the existing `micromanager_camera_input` device. Otherwise, create a new class in `apps/io/input_device/` that extends `BaseInputDevice`:

```python
from core.io.input_device.BaseInputDevice import BaseInputDevice

class MyCameraInput(BaseInputDevice):
    device_class = "my_camera_input"

    def connect(self):
        # open connection to hardware

    def configure(self):
        # apply settings from self.config

    def _get_input(self):
        # return a frame / sample

    def close(self):
        # release hardware
```

Then reference `input_device_class: my_camera_input` in your `io_config.yaml`. Each input device gets linked with a matching `data_interface` — This controls how data is saved. The `DataInterface` is a useful abstraction for working with data that has a common structure (e.g. an XY uint16 matrix from a camera).

## Step 2: Create a module for your output device

Say you want to drive a stimulating LED or a motorized stage. Create a class in `apps/io/output_device/` that extends `BaseOutputDevice`:

```python
from core.io.output_device.BaseOutputDevice import BaseOutputDevice

class MyStageOutput(BaseOutputDevice):
    device_class = "my_stage_output"

    def connect(self):
        # open connection to hardware

    def configure(self):
        # apply settings from self.config

    def _update_output(self, **kwargs):
        # move stage / fire LED / etc.

    def close(self):
        # release hardware
```

Then reference `output_device_class: my_stage_output` in your `io_config.yaml`.

Output devices are driven by the return value of `_check_logic` (see below). The engine routes a dict of `{output_device_name: {kwargs}}` to each named device's `update_output`, which records a timestamp and calls `_update_output(**kwargs)`.

## Step 3: Create a module for your closed-loop logic algorithm

Create a class in `apps/logic/` that extends `BaseClosedLoopLogic`:

```python
from core.logic.BaseClosedLoopLogic import BaseClosedLoopLogic

class MyLogic(BaseClosedLoopLogic):
    logic_class = "my_logic"

    def initialize_model(self):
        # one-time setup before the loop starts

    def process_sample(self, sample):
        # receive data from input devices, update internal state

    def _check_logic(self):
        # Return None to do nothing this frame.
        # To drive an output device, return a dict of:
        #   {output_device_name: {kwargs for _update_output}}
        # The engine dispatches this to the named device and records a timestamp.
        if self.should_stimulate():
            return {"my_stage_output": {"x": 1.0, "y": 2.0}}
        return None

        # Note: if you don't need timestamped output events, you can also
        # call self.output_devices["my_stage_output"].update_output(x=1.0, y=2.0)
        # directly inside process_sample instead.

    def close(self):
        # cleanup
```

If you're new to this, a good first algorithm is an interactive GUI that displays your real-time data without driving any hardware — use `pyqtgraph` and update a plot inside `process_sample`. Once the visualization looks right, add the output logic.

## Step 4: Create configuration files

`clef` is entirely directed by three YAML configuration files, each corresponding to one of the core concepts listed above: `io.yaml`, `logic.yaml`, and `session.yaml`. YAML files are validated by Pydantic on initialization.

- `io.yaml` — Lists the input and output devices for the experiment. Each entry names a device_class (the registered Python plugin to load) along with any device-specific parameters you want to vary across experiments (camera exposure, ROI, illumination properties, serial port, etc.).
- `logic.yaml` — Selects the closed-loop algorithm via logic_class and supplies its tunable parameters (thresholds, gains, target regions, etc.).
- `session.yaml` — Describes the run itself rather than the hardware or algorithm. This is where you record the contextual metadata needed to interpret your data later: who ran the session, when, on what subject, under what conditions, where the outputs are written, and how long the run lasts.

## Step 5: Put your python files in the correct directories

Drop your application-specific code in the appropriate folder under `apps/`:

```
apps/
  io/
    input_device/    ← your input device
    output_device/   ← your output device
  logic/             ← your closed-loop algorithm
  config/
    your_app_name/   ← your three YAML configs
```

`clef` auto-discovers any class with a `device_class`, `data_interface_class`, or `logic_class` ClassVar at startup — no registration step needed.

## Step 6: Run your app with the `clef` CLI

```bash
clef your_app_name
```

`clef` ships with a command-line entry point that loads your three configs, runs the closed loop engine, and writes outputs to a timestamped session directory. At the end of every session, `clef` writes data for each `input_device` based on the `data_interface`, and also outputs a JSON metadata file containing all configuration, events, and per-device timestamps.

## Alternate setup: Vibe coding quickstart

Load `core/` and a few example apps from `apps/` into your AI assistant's context, then send a prompt like:

> I'm trying to make a new CLEF app. I need an input device for [ABC], and an output device for [XYZ]. Here are some scripts where I demonstrate control of the devices:
>
> [paste your existing device control scripts]
>
> Format them to work with CLEF. For the closed-loop logic algorithm, make me [describe the feedback rule]. Finally, make me config files in a new app directory named `your_app_name`.

That should get you 90% of the way there, but complex control will require testing.

# Hosting a model as a network service

For computationally heavy models (e.g. a speech BCI decoder with a GRU + n-gram language model), you can offload inference to a separate process or machine and have `clef` connect to it over a network. The `speech_bci` demo uses this pattern — the decoder runs as a FastAPI WebSocket server inside Docker, and the `clef` logic algorithm streams neural data to it over WebSocket and receives decoded text back.

The relevant files are in `docker/` and `apps/subprocess/`:

```
docker/
  Dockerfile.decoder     ← multi-stage build: compile C++ extension, then runtime image
  deploy_decoder.sh      ← build → push to Artifact Registry → deploy to GCP Compute Engine
apps/subprocess/
  speech_bci_server.py   ← FastAPI WebSocket server (the model endpoint)
  speech_bci_inference.py
  speech_bci_decoder.py
```

High-level steps to follow this pattern for your own model:

1. **Wrap your model in a server** — create a FastAPI (or equivalent) server in `apps/subprocess/` that loads your model once at startup and exposes a WebSocket or HTTP endpoint.
2. **Write a Dockerfile** — copy your server code and model weights into the image. Expose the port. Set a healthcheck. See `docker/Dockerfile.decoder` for a multi-stage example that compiles a C++ extension before building the runtime image.
3. **Build and test locally** — `docker build -f docker/Dockerfile.decoder -t my-model .` and verify the healthcheck passes before pushing anywhere.
4. **Push and deploy** — `deploy_decoder.sh` shows the full GCP Compute Engine workflow (Artifact Registry push → `create-with-container`). Adapt it for your cloud provider or run the container on-prem.
5. **Point your CLEF logic at the server** — set the `decoder_url` (or equivalent) in your `io_config.yaml` or `logic_config.yaml` to the server's WebSocket address (e.g. `ws://<IP>:8765/ws`). Your logic algorithm handles the client-side connection.

# Contributing

Contributions are welcome. Fork the repo, create a branch off `main`, and open a pull request — the [PR template](.github/pull_request_template.md) will guide you through summary, changes, and testing notes. For non-trivial changes, please open an issue first to discuss scope.

# License

MIT
