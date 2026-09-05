# Third-party notices

## Transfer-Learning-Library

The optional feature-level DANN baseline in `src/dann_feature_baseline.py`
uses an independently written PyTorch implementation of the gradient-reversal
mechanism and warm-start schedule described by:

- Repository: <https://github.com/thuml/Transfer-Learning-Library>
- Relevant files inspected: `tllib/modules/grl.py` and `tllib/alignment/dann.py`
- Revision inspected: `fb07e9150c014455f2fd8fa6225930488c2454f0` (GRL file)
- License: MIT, copyright notice in that repository's `LICENSE`.

`src/q3_pipeline.py` likewise independently implements the GRL warm-start,
balanced source/target batch organisation and DANN loss from their mathematical
definitions. It does not copy any source files or their target-label evaluation
protocol.

This repository does not copy that library's source files.  The notice is kept
to attribute the algorithmic implementation and retain the applicable license
context for any future reuse or distribution.
