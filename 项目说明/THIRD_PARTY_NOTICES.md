# Third-party notices

## Python Optimal Transport (POT)

- Repository: <https://github.com/PythonOT/POT>
- Package: `POT==0.9.7.post1` from PyPI
- License: MIT License
- Use in this repository: Question 3 directly calls
  `ot.da.SinkhornLpl1Transport` for Sinkhorn transport with LpL1
  class regularization. This repository does not copy or reimplement POT's
  Sinkhorn or class-regularized optimal-transport solver.
