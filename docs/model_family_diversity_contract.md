# Model-family diversity contract

A reconstruction is admitted as a candidate family only after it passes three
ordered gates.

1. It must implement the common requested-source contract: normalized g/r/z
   blend plus a Gaussian coordinate prompt, three g/r/z source-layer outputs,
   frozen inverse normalization, detected-electron units, no clipping, and zero
   residual background.
2. It must be promptable and scientifically usable on a fresh non-Atlas,
   source-group-isolated validation manifest. Architecture novelty alone is not
   family admission.
3. Only then may a frozen one-pass Atlas evaluation test whether cross-family
   distance exceeds same-family seed distance, both decompositions remain
   forward-consistent, and additional ambiguity witnesses are not output
   artifacts.

The prompted ResUNet passed structural compatibility and reconstruction-factor
checks but failed promptability. A batch-geometry-sensitive candidate hash also
prevented a complete contract pass, although identical-batch replay passed.
Because gate 2 failed, it is not admitted as a useful second family and no Atlas
or auditor inference is authorized.

