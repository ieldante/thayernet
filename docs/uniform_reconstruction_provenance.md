# Uniform reconstruction provenance

Every empirical outcome in the prospective hierarchical-feasibility campaign
uses the Phase-I Condition-C prompted checkpoint:

- path: `outputs/runs/thayer_select_prompt_ablation_20260711_164329/checkpoints/c_randomized_coordinate_prompt_best.pth`;
- SHA-256: `e9176dc5d5fe91a07bc72f9eb811c9692c2af9315f2c367135cbd84d3bffe382`;
- architecture: compact 119,091-parameter prompted U-Net with a three-band
  linear reconstruction head;
- input: frozen training-normalized `g,r,z` plus a two-pixel-sigma Gaussian
  coordinate prompt;
- output scaling: multiply each predicted band by the same frozen per-band
  training scale used to normalize the input;
- execution: MPS, `eval()` mode, `requires_grad=False` for every parameter,
  and `torch.no_grad()`;
- deterministic audit: exact repeated outputs before bulk extraction;
- checkpoint audit: byte-identical before and after the campaign.

The same checkpoint hash appears in every training, validation, and calibration
outcome row. R1 reconstruction outputs were not used. Lightweight CPU heads do
not modify or fine-tune the reconstructor. Development and lockbox inference
counts are zero.
