# D3 Stop-Event Precedence

Simultaneous events are logged in this exact order:

1. access-guard violation;
2. historical-write attempt;
3. protected-path access;
4. cache, bytecode, or delete event;
5. target or hash mismatch;
6. nonfinite value;
7. MPS fallback;
8. physical negative output;
9. cached-feature mutation;
10. optimizer-contract violation;
11. expert death;
12. prompt collapse;
13. scientific success;
14. budget exhaustion.

The first present event selects terminal status and exit code. Safety failures
always outrank success. Success is valid only when no higher-priority failure
is present at the same evaluation. Safety failures persist the event record;
post-state numerical and policy failures additionally persist a terminal-
failure state when a valid state exists.
