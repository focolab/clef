# CLEF Class Hierarchy

# All boxes are **classes**. Arrows show **inheritance** or **composition** (labeled with the held reference). No data flow is shown here — see `system_overview.md`.

```mermaid
classDiagram
    direction TB

    class BaseInputDevice {
        +initialize()
        +close()
        +get_metadata() dict
    }

    class BaseDataInterface {
        +sample_data() ndarray
        +get_sample_shape() tuple
        +get_sample_dtype() dtype
        +configure_sampling()
        +save_data()
    }

    class BaseOutputDevice {
        +initialize()
        +trigger(params)
        +close()
        +get_metadata() dict
    }

    class BaseClosedLoopLogic {
        +initialize_model()
        +process_sample(sample) dict
        +get_metadata() dict
        +close()
    }

    class IOManager {
        +initialize()
        +close()
        +get_metadata() dict
    }

    class LogicManager {
        +initialize()
        +close()
        +get_metadata() dict
    }

    class ClosedLoopEngine {
        +run()
        +initialize()
        +run_acquisition_loop()
        +close()
    }

    class ConfigManager {
        +load(paths) Configs
    }

    class IOConfig {
    }

    class ClosedLoopLogicConfig {
    }

    class SessionConfig {
    }

    BaseInputDevice <|-- UserInputDevice
    BaseOutputDevice <|-- UserOutputDevice
    BaseClosedLoopLogic <|-- UserClosedLoopLogic

    BaseInputDevice --> BaseDataInterface : data

    IOManager --> BaseInputDevice : input_device
    IOManager --> BaseOutputDevice : output_device
    LogicManager --> BaseClosedLoopLogic : logic

    ClosedLoopEngine --> IOManager : io
    ClosedLoopEngine --> LogicManager : logic

    ConfigManager --> IOConfig : produces
    ConfigManager --> ClosedLoopLogicConfig : produces
    ConfigManager --> SessionConfig : produces
```

**Legend**

| Symbol | Meaning |
|---|---|
| All boxes | Classes |
| `BaseXxx` | Abstract base class (defined in `core/`) |
| `UserXxx` | User-supplied subclass (lives in `apps/`) |
| `XxxConfig` | Pydantic model for config validation |
| `──▷` (open arrow) | Inheritance: subclass extends base |
| `-->` labeled | Composition: left class holds a reference to right |
