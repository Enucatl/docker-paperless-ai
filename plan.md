# Paperless AI Evaluation Migration Plan

- [x] Use TypeSafe Jev/System One as the sole metadata evaluator.
- [x] Keep the evaluation corpus input-only with no expected metadata.
- [x] Send one Jev request per document containing date, correspondent, and title judgments.
- [x] Expose Phoenix metadata metrics as continuous Jev probabilities rather than binary `0`/`1` values.
