# Course notation and conventions

Maintained by hand. Two jobs:

1. **Stage 1 vocabulary biasing.** The headword list below is flattened into
   the ASR prompt. Whisper's prompt window is ~224 tokens, so keep the
   *Vocabulary* section tight and put the terms that actually get misheard
   first. This is the highest-leverage knob in the pipeline.
2. **Stage 2/3 context.** The whole file is handed to the LLM on every prompt.

> **Only list terms this course actually uses.** Priming a term that never
> occurs makes the recogniser reach for it. Observed on lecture 11: with
> "Sylow" in this list, large-v3 turned "**zero** is equal to one" into
> "**Sylow** is equal to 1". A wrong entry here actively creates errors.

## Vocabulary

Terms the recogniser gets wrong. Most-misheard first.

- Gronwall, Gronwall's lemma, epsilon, delta, Lipschitz
- ODE, initial value problem, uniqueness, existence
- factorial, exponential, integrand, derivative, differentiable
- supremum, infimum, monotone, continuous, bounded

## Conventions

Describe how this lecturer speaks, so the math pass can disambiguate.

- "eff of ex" -> `f(x)`
- "a sub n" -> `a_n`
- "m prime of s" -> `m'(s)`
- "g of t one dee t one" -> `g(t_1)\,dt_1`
- "x squared plus one over x" is ambiguous; the lecturer usually means
  `x^2 + \frac{1}{x}` and says "all over" for `\frac{x^2+1}{x}`.

## Do not correct

Things that look like errors but are not.

- (add entries here as you find them)
