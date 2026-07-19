# Output-space optimization audit

Thayer-LG optimized detached two-expert six-channel tensors directly. No model
was loaded into the graph, no neural parameter received a gradient, and no
model optimizer step occurred. The protocol used CPU float32 Adam for 40 fixed
steps with bounds [-8, 8].

The full frozen objective moved away from exact truth. Starting from the exact
approved outputs lowered the objective from 0.029377 to 0.029000, increased
mean primary scientific distance from numerical zero to 8.265, reduced
ordinary coverage to 0.03125, and reduced every ambiguous coverage rate to
zero. Starting from trained outputs also converged to zero coverage.

Diagnostic source-only objectives behaved differently. Source reconstruction
plus ordinary concentration reached mean primary scientific distance 2.295,
ordinary coverage 0.5625, ambiguous own and alternate coverage 0.625, and
both-mode coverage 0.5625 after the same 40 steps. Excluding the forward term
was the strongest leave-one-term-out scientific result. These runs diagnose
the frozen geometry; they do not select or authorize a replacement loss.
