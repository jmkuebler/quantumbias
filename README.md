
# The Inductive Bias of Quantum Kernels



## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Structure
- All the methods that define the quantum kernels, target functions etc. are contained in the module `quantum_methods.py`.
- To reproduce the figures of the paper first run `theorem_2.py`, `train_test_error.py`, `alignment.py`, or `tm_alignment.py`. 
This runs the experiments and stores the outcome data under `data/...`. Then simpy run the corresponding `plot_XYZ.py` 
  to reproduce the figures (stored in `evauations/...`).
  
  

