# Signed-noise-residual physical contract

For a signed zero-background observation `O`, exact decomposition into only
nonnegative terms is impossible wherever `O < 0`. The corrected contract
distinguishes astronomical source layers from the observational noise closure.

Given normalized head coordinates `L_req` and `L_comp` and positive g/r/z
scales `S`:

- `P_req = S ReLU(L_req)`;
- `P_comp = S ReLU(L_comp)`;
- `P_noise = O - P_req - P_comp`.

`P_req` and `P_comp` are the only physical source layers. `P_noise` is a
signed residual/noise layer. The construction has exact zero representation,
no forced positive source floor, and algebraic observation closure.

The full Family-E1 training-free preflight passed on 14,000 frozen
source-group-safe episodes. Truth was used only to form inverse-coordinate
representability witnesses `T/S`; clean targets are not inference inputs.
ReLU is the sole in-forward source mapping and may not be replaced by
evaluation-only clipping.

Passing this contract does not imply that a neural model can learn the
coordinates or pass catalog-safety gates.
