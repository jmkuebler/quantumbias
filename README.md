
# The Inductive Bias of Quantum Kernels



## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Reproduce Results
- All the methods that define the quantum kernels, target functions etc. are contained in the module `quantum_methods.py`.
- To reproduce the figures of the paper: 
  - first run `theorem_2.py`, `train_test_error.py` (to speed up, you might want to
  set runs=1 to only run the experiment for a single random seed), 
  `alignment.py`, or `tm_alignment.py`. 
This runs the experiments and stores the outcome data under `data/...`. 
  - Run the corresponding `plot_XYZ.py` 
  to reproduce the figures (stored in `evauations/...`).
  
  

