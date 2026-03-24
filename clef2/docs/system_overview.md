# CLEF2 System Overview

All boxes are **runtime instances**. Arrows show **data flow** (labeled) or **control flow** (unlabeled). Inheritance is not shown here — see `class_hierarchy.md`.

```mermaid
flowchart TD
    User([Researcher])

    subgraph UserFiles ["apps/ - user-editable"]
        Configs[YAML configs\nsession · io · logic]
        LogicImpl[ClosedLoopLogic subclasses]
        InputImpl[InputDevice subclasses]
        OutputImpl[OutputDevice subclasses]
    end

    subgraph CoreRuntime ["core/ - framework"]
        CM[ConfigManager]
        Engine[ClosedLoopEngine]
        IOM[IOManager\nInputDevice dict · OutputDevice dict]
        LM[LogicManager\nClosedLoopLogic dict]
    end

    Storage[(Output\n.tiff · metadata.json)]

    User -->|edits| UserFiles
    Configs -->|validated config| CM
    CM --> Engine
    InputImpl --> IOM
    OutputImpl --> IOM
    LogicImpl --> LM
    Engine --> IOM
    Engine --> LM
    IOM -->|sample| Engine
    Engine -->|trigger decision| LM
    LM -->|trigger| IOM
    Engine --> Storage
```

**Legend**

| Symbol | Meaning |
|---|---|
| `([...])` rounded | External actor |
| `[...]` rectangle | Runtime instance |
| Subgraph | Directory / module grouping |
| `-->` unlabeled | Control flow (initialize, orchestrate) |
| `-->` labeled | Data flow — label names what is passed |
| `[(...)]` cylinder | Persistent storage |
