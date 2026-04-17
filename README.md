# CLEF - what is it?

Closed-Loop Experimental Framework

CLEF is a customizable software platform to specify and run "closed-loop" experiments in neuroscience, biology or other physical sciences. It provides tooling for interfacing with input (data acquisition, such as microscopes or electrodes) and output (control and perturbation, such as optogenetic lasers or stimulus delivery) hardware systems, for design of on-the-fly computational analysis and logic that that executes during experimental sessions, and for GUI-based interactive human monitoring and control of live experimental sessions. 

## Basic concepts/terminology



## Installation

```bash
git clone https://github.com/focolab/clef.git
cd clef
```

**Without extras** — installs only the base framework (core classes, config, engine):

```bash
pip install -e .
```

**`core`** — adds pytest for running the core test suite:

```bash
pip install -e ".[core]"
```

**`demos`** — adds everything needed to run the ring attractor, recording playback, screenshot, and speechbci demos:

```bash
pip install -e ".[demos]"
```

**`all`** — full install including hardware integrations and all extras:

```bash
pip install -e ".[all]"
```

---

## Quickstart: demos

Install with the `demos` extra (see above), then run:

```bash
clef ring_attractor
```

`clef <name>` searches `apps/config/` for a directory named `ring_attractor`, finds the three config files inside it (`session_config.yaml`, `io_config.yaml`, `logic_config.yaml`), validates them, and prompts before starting the loop. Other available demos:

```bash
clef recording_playback # playback an existing .tiff recording you already have, for model workflow development
clef speech_bci # have a ML model from brain2text? visualize it here (requires setting up your own model first)
```

To validate configs without running:

```bash
clef ring_attractor --validate-config
```

To pass config files explicitly:

```bash
clef --session s.yaml --io io.yaml --logic l.yaml
```

---

## How do I configure/customize clef?

CLEF is directed by YAML configuration files. There are three:

| File | Purpose |
|---|---|
| `session_config.yaml` | Session metadata: name, user, output directory, number of samples, save flags. Also a good place for any experiment-specific metadata you want saved with the session — subject ID, genotype, treatment condition, etc. — structured however you like (see [`apps/config/physical_hardware/session_config.yaml`](apps/config/physical_hardware/session_config.yaml) for an example). |
| `io_config.yaml` | Which input devices and output devices to use, and their parameters |
| `logic_config.yaml` | Which closed-loop logic algorithm(s) to run, their parameters, and which devices they read from / write to |

At runtime, the engine reads from input devices, passes samples to the logic algorithm, and dispatches output commands to output devices:

```
input_devices → engine → closed-loop logic → engine → output_devices
```

The loop runs for `num_samples` iterations (or indefinitely if set to `-1`).

---

## How do I use CLEF for my experiments?

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

CLEF auto-discovers any class with a `device_class`, `data_interface_class`, or `logic_class` ClassVar at startup — no registration step needed.

### Create an input device

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

Then reference `input_device_class: my_camera_input` in your `io_config.yaml`.

### Create an output device

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

### Create a closed-loop logic algorithm

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

---

## Extending CLEF: vibe coding quickstart

Load `core/` and a few example apps from `apps/` into your AI assistant's context, then send a prompt like:

> I'm trying to make a new CLEF app. I need an input device for [ABC], and an output device for [XYZ]. Here are some scripts where I demonstrate control of the devices:
>
> [paste your existing device control scripts]
>
> Format them to work with CLEF. For the closed-loop logic algorithm, make me [describe the feedback rule]. Finally, make me config files in a new app directory named `your_app_name`.

That should get you 90% of the way there.

---

## Output data

Each input device controls how its data is saved — see the `save_data` method on your input device and its associated `DataInterface`. At the end of every session, CLEF writes a JSON metadata file containing all configuration, events, and per-device timestamps.

---

## Hosting a model as a network service

For computationally heavy models (e.g. a speech BCI decoder with a GRU + n-gram language model), you can offload inference to a separate process or machine and have CLEF connect to it over a network. The `speech_bci` demo uses this pattern — the decoder runs as a FastAPI WebSocket server inside Docker, and the CLEF logic algorithm streams neural data to it over WebSocket and receives decoded text back.

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

---

## Development

```bash
pip install -e ".[core]"
pytest tests/ -v
```

---

## License

MIT
