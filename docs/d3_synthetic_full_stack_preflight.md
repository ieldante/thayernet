# D3 Synthetic Full-Stack Preflight

The Thayer-D3E synthetic preflight used deterministic analytic tensors with
production feature, target, output, and batch shapes. The fixtures contain no
scientific values and use no random sampled data.

The strict MPS consumer passed:

- exact two-expert model construction and strict initial-state loading;
- production-shape forward and exact square mapping;
- batch-size, order, layout, and larger-batch invariance checks;
- production/reference hard assignment and pair-cost comparison;
- production/reference target-loss comparison;
- production/reference forward and truth-coverage evaluation with zero file
  I/O from the evaluator;
- exact AdamW construction with learning rate `0.001`, weight decay `0`, no
  scheduler, and gradient clipping at `5.0`;
- one backward and optimizer step with finite nonzero gradients and updates in
  both experts, including final and nonfinal blocks;
- checkpoint save, strict reload, and fresh-process MPS replay with matching
  state, gradient, assignment, loss, and output hashes; and
- equality of all 180 declared requirements with all 180 accessed or validated
  requirements.

All 25 preregistered capsule corruptions failed closed before model execution.
The final actual-consumer and future-launcher preflights emitted
`ALL_D3_REQUIREMENTS_CONSUMER_VALIDATED` and
`READY_FOR_AUTHORITATIVE_D3_EXECUTION`.

No scientific array value was loaded and no scientific D3 step was executed.
Synthetic execution establishes software-path executability, not scientific
decoder sufficiency.
