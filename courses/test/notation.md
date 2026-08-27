# test — notation and conventions

Edit this from the course page, or straight on disk. Two jobs:

1. **Stage 1 vocabulary biasing.** The list below is fed to the recogniser.
   Whisper honours only ~224 tokens of prompt, so keep it tight and put the
   most-misheard terms first. This is the highest-leverage knob in the
   pipeline.
2. **Stage 2/3 context.** The whole file goes to the model on every prompt.

> **Only list terms this course actually uses.** Priming a term that never
> occurs makes the recogniser reach for it: with "Sylow" in this list, a
> lecture that said "zero is equal to one" came back as "Sylow is equal to 1".

## Vocabulary

Terms the recogniser gets wrong. Most-misheard first.

- (add the names, theorems and symbols this course keeps mangling)

## Conventions

How this lecturer speaks, so the maths pass can disambiguate.

- "eff of ex" -> `f(x)`
- "a sub n" -> `a_n`

## Do not correct

Things that look like errors but are not.

- (add entries here as you find them)
