# Data dictionary

The observation unit is a model-benchmark-source-protocol cell. The canonical regression uses percent-score cells only. Non-percent score scales are retained for provenance and excluded from the primary target unless explicitly normalized in a future extension.

Important columns:

- `model`: model name used in the analysis.
- `family`: model-family grouping used for grouped validation.
- `benchmark`: versioned benchmark identifier, such as `AIME_2024` or `AIME_2025`.
- `score`: reported score. Canonical analysis expects 0-100 percent scores.
- `score_scale`: scale of the reported score, for example `percent` or `elo`.
- `source`: source class, such as `official`, `cross_report`, `leaderboard`, `aggregator`, or `estimated`.
- `source_weight`: pragmatic reliability weight used in weighted regressions.
- `source_detail`: human-readable provenance note.
- `exact_protocol_match`: whether the row appears to match the intended benchmark/version/protocol definition.
- `thinking`: reported reasoning/thinking mode where known.
- `tools_allowed`: tool-use flag where known.
- `reasoning_effort`: coarse reasoning-effort category where known.

Caution: closed-source parameter counts, training tokens, and training-recipe proxies are not direct measurements. They are public metadata proxies and should be revised as better sources become available.
