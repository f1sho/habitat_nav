# Habitat-Sim Indoor Robot Navigation

This repository contains the implementation and evaluation code for an indoor robot navigation project developed in Habitat-Sim.

The navigation pipeline is primarily based on:

**RGB Camera → YOLO Instance Segmentation → IPM Distance Estimation → Global Planning → Discrete DWA Local Planning**

The project also includes experiments for comparing different model deployment formats and evaluating perception performance, IPM distance accuracy, and navigation performance.

## Navigation Demo

To run the visualised navigation simulation:

```bash
cd scripts
python ipm_main.py
```

`ipm_main.py` displays the Habitat-Sim navigation process together with perception and navigation information.

Before running the script, update the following paths in `ipm_main.py` to match the local Replica dataset installation:

```python
scene_path = "/path/to/replica/apartment_2/habitat/mesh_semantic.ply"
navmesh_path = "/path/to/replica/apartment_2/habitat/mesh_semantic.navmesh"
```

The current demo uses `yolo26n-seg.onnx` by default. The model path can also be changed inside the script.

## Main Files

* `scripts/ipm_main.py` — visualised navigation demo
* `scripts/run_navigation.py` — headless navigation runner
* `scripts/run_evaluation.py` — model and navigation evaluation
* `scripts/run_ipm_accuracy.py` — IPM distance accuracy evaluation
* `scripts/core/` — perception, Habitat environment, and planning modules
* `scripts/evaluation/` — evaluation metrics and logging
* `scripts/results/` and `results/` — experimental outputs
* `scripts/legacytests/` — earlier development and experimental scripts

## Legacy Tests

The files under `scripts/legacytests/` are retained mainly as development records and reference code.

Some of these scripts were originally designed to run directly from the `scripts/` directory. To test them, the relevant Python file may need to be copied or moved back into `scripts/` before execution.

**Note:** the main project interfaces were modified during later development, so some legacy scripts may no longer be compatible with the current codebase and are not guaranteed to run without modification.

## Requirements

The project requires Habitat-Sim, the Replica dataset, Ultralytics YOLO, and the relevant Python/CUDA dependencies.

Model files used during the experiments, including PyTorch, ONNX, and TensorRT variants, are included under `scripts/`.

This repository represents the final research/development state of the project rather than a fully packaged standalone application.
