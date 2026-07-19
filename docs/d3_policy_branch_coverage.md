# D3 Policy Branch Coverage

The canonical registry contains 16 policies. The actual launcher delegates
policy preflight to the pure engine and does not reproduce its thresholds or
conditions.

The synthetic fixture inventory contains 76 fixtures and covers all 106 stable
branch IDs. Coverage requires execution plus asserted output; importing a
function does not count. Terminal, nonterminal, success, failure, diagnostic,
semantic-state, outcome, authorization, artifact, error-rejection, and launcher
readiness branches all passed.

Declared, executable, accessed, branch-tested, persisted-artifact, and launcher
policy sets each contain the same 16 IDs with set SHA-256
`ab7e9cdd39e2ae5c952f6374b606d389007b0ae1a434e090b8d779f1b201393c`.
