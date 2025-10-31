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

HERE ARE A FEW THINGS THAT I NEED TO INTEGRATE INTO THIS SECTIONS ORGANIZATION:

1. Current State Analysis (Section 2)

Pain Points Identified: 7 major categories

Hardware-specific references (35+ locations)
Worm/neuron terminology throughout
Scattered configuration management
Legacy code (~4,250 LOC to refactor or deprecate)
No testing infrastructure
Absolute file paths everywhere
No algorithm architecture consistency



2. Refactoring Objectives (Section 3)

Primary Goals: 3 must-have deliverables

Hardware abstraction (0 hardcoded device strings)
Distributable package (pip install clef)
Demo mode functional (reviewers can validate)


Success Metrics Defined: 6 measurable targets

Hardcoded refs: ~35 → 0
Test coverage: 0% → 60%
Demo runtime: N/A → <2 min

3. Architecture & Design Decisions (Section 4)

Target Architecture: Component diagram with clear responsibilities

ConfigManager (NEW): YAML loading/validation
HardwareManager (REFACTORED): Backend abstraction
ClosedLoopEngine (REFACTORED): Config-driven orchestration

Design Patterns: 4 patterns selected

Configuration Object (replace args dict)
Adapter (hardware backends)
Factory (stimulus/algorithm creation)
Strategy (future algorithm base class)


Key Decisions: 5 major technical choices

YAML + Pydantic validation
Keep Gooey as config generator
Backend abstraction (pycromanager/pymmcore/dummy)
Package defaults + user overrides
Incremental migration strategy

4. Configuration Templates Created

Full Templates: 3 comprehensive YAML files

hardware.yaml: Device mappings, MM configs, illumination channels
experiment.yaml: Acquisition params, subject metadata, output settings
algorithm.yaml: CL algorithm selection, stimulus parameters


Minimal Templates: 3 bare-bones test configs

hardware_minimal.yaml: Dummy backend, no real devices
experiment_minimal.yaml: 100 frames, single Z, no saves
algorithm_minimal.yaml: DummyAlg, no GUI, no randomization

END INFORMATION WE STILL NEED TO INTEGRATE

### Target Architecture
```
[High-level architecture description or diagram]

Key components:
- Component A: [Responsibility]
- Component B: [Responsibility]
- Component C: [Responsibility]

Data flow:
[Describe how data flows through the system]
```

### Design Patterns to Implement
1. **[Pattern Name]**
   - Where: [Which modules]
   - Why: [Rationale]
   - Example: [Brief example]

### Key Technical Decisions

#### Decision 1: [Decision Title]
- **Context:** [What problem are we solving?]
- **Options Considered:**
  - Option A: [Pros/Cons]
  - Option B: [Pros/Cons]
- **Decision:** [Chosen option]
- **Rationale:** [Why we chose this]
- **Trade-offs:** [What we're giving up]

#### Decision 2: [Decision Title]
- **Context:** [What problem are we solving?]
- **Options Considered:**
  - Option A: [Pros/Cons]
  - Option B: [Pros/Cons]
- **Decision:** [Chosen option]
- **Rationale:** [Why we chose this]
- **Trade-offs:** [What we're giving up]

### Migration Strategy
- **Approach:** [Incremental / Big-bang / Hybrid]
- **Backward Compatibility:** [Required / Not required]
- **Feature Flags:** [Yes / No - describe strategy]
- **Deployment Strategy:** [Blue-green / Rolling / Canary / etc.]

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