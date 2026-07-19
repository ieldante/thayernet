# Permutation-invariant decomposition loss

For two hypotheses and two approved targets, Thayer-MH computes requested,
companion, and source-sum loss under both assignments and uses the smaller one
for that scene. The assignment exists only for loss calculation; no global slot
identity is imposed. Ordinary scenes supervise both hypotheses to one target
and add concentration.

The frozen objective also includes observation recomposition, prompt-swap set
consistency, and a small unordered pair-equivalence term. There is no generic
diversity reward: separation is useful only when slots match different approved
truths while remaining prompt-faithful and forward-consistent.

Tests cover both assignments, slot permutation, prompt swap, full six-channel
decomposition, source summation, ordinary concentration, and unordered pair-set
distance. Passing these tests did not imply scientific truth coverage.
