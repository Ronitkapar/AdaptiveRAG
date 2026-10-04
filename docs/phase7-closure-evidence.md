# Phase 7 Closure — Evidence Snapshot

This file exists to make the Phase 7/8 evidence **auditable and reproducible**.
Most of it is deliberately *not* in git:

* `experiments/*/` and `storage/` are gitignored, so every Phase 8 artifact —
  the oracle, the gold-label audit/repair, the arm comparison, and the per-arm
  `rows.jsonl` inputs — is unversioned.
* `data/processed_phase8_{before,after}/` are derived corpora (reproducible from
  the raw PDFs by the ingestion pipeline) and are now ignored like `data/processed/*`.

The snapshot therefore records the **source commit** plus a **SHA-256 of every
input and artifact** the Phase 7 closure will rely on. Any later run can be shown
to have used the same evidence, or shown to have drifted.

## 1. Provenance

| Field | Value |
|---|---|
| Branch (freeze) | `phase7-closure` |
| Tag (freeze) | `phase7-closure-freeze-01` |
| Parent commit (Phase 7 close) | `ae64d676b8fcf4f1c6d51d7ca3d631a63b769936` |
| Environment | Python 3.12.14 (`.python-version` = 3.12), numpy 2.5.3 |

The freeze commit captures the 20 modified tracked files plus the new Phase 8
scripts/tests/docs that were uncommitted on `phase-5-reranking`. Code and docs,
not bulk data, are the versioned part; the data/artifacts are pinned by hash below.

## 2. Corpora

Both Phase 8 arms report the **same** `corpus_version` string
(`corpus_c5996129918d7e44`), which does *not* distinguish the column-extraction
before/after pair. The distinguishing key is the **namespace directory** and the
manifest hashes below. Do not conflate the two corpora by `corpus_version` alone.

| Corpus | Namespace | Ingestion | Chunking | Chunks / Docs | Embedding |
|---|---|---|---|---|---|
| shipping (after) | `data/processed_phase8_after` | ingestion_v3 | structure_aware_v1 | 613 / 14 | text-embedding-3-large, dim 3072, normalized |
| pre-fix (before) | `data/processed_phase8_before` | ingestion_v3 | structure_aware_v1 | see manifest | text-embedding-3-large, dim 3072, normalized |

Manifest SHA-256:

| File | after | before |
|---|---|---|
| `chunks_manifest.json` | `bf0b8b2f7836c689ef79177bf5f2b3d01599fa3095a7321ee6bc436ddbfe8800` | `23f7b453ac9cefe7ca3bd29d87cd5634eb5790d6064dbabd767f8905fcfa51f0` |
| `documents_manifest.json` | `3f8b4a6b255c8fb7f3547b89f9d60f34642483db621f01643a3a4cb0e58cbcfc` | `4959b1daa175188282d266b4ce81308f7bf96e77d747833d5550e7ee11b68dab` |
| `embeddings_manifest.json` | `c413a3039211dac5c1c9d3988851edf5500b1f95588284f7585fd891bd806aa2` | `76341bc411af2d7700df85a3190113b51db48fe7eb40f8950fd6b7553c7349fe` |
| `chunk_stats.json` | `bf2c696225f96527cf4f722ed9f9ffb545bcfaaa1dbf59c2b12377e55a113c9a` | `c647df1c3d792d4eacab164eba82291af2949872c6c108d5f455e34ac36cdcbc` |

## 3. Frozen evaluation dataset

| File | SHA-256 |
|---|---|
| `data/evaluation/phase7_eval_v1.jsonl` | `f0695189903c8ef8ea7185ac96a78b80f4f96d836c55214bf025176363e295cb` |
| `data/evaluation/phase7_eval_v1.ledger.jsonl` | `b6219e40df37d0a77aa0484fb05df6e91f66db33d9a6d65eddb9c9b0b49b0edb` |
| `data/evaluation/README.md` | `9bf23bc26de48cc8d24d5738f840e6410760bc610f3c3c4a0d7c6646a15006e9` |

## 4. Frozen routing configuration (all arms)

From `.../E1_baseline_comparison__adaptive/config.json`:

```
available_strategies    = [bm25, dense, hybrid, hybrid_rerank]   <-- NOTE: "adaptive" is NOT selectable
escalation_ladder       = [bm25, dense, hybrid, hybrid_rerank]
max_escalation_steps    = 1
sufficiency_threshold   = 0.5
coverage_threshold      = 0.5
top1_coverage_threshold = 0.3
cost_weight             = 0.25
router_version          = rule_based_v1
strategy_cost_ms        = {bm25: 2.25, dense: 451.78, hybrid: 455.97, hybrid_rerank: 3854.41}
```

**Why this matters for Gate 2:** the shipped router can only ever choose among
`[bm25, dense, hybrid, hybrid_rerank]`. The arm named `adaptive` is the router's
*output* over 107 queries, not a strategy a router could select. The current
`oracle_ceiling.json` nevertheless includes `adaptive` in its selectable set —
the correction the closure plan requires.

Per-arm `config_hash` (p8a = after, p8b = before):

| Strategy | p8a hash | p8b hash |
|---|---|---|
| bm25 | `a264056d19049edd` | `42bda3317b56a047` |
| dense | `31bad5d6f496eff5` | `884e57e21efa6de4` |
| hybrid | `92d8bb4920011209` | `e1b3a7df7b9c978d` |
| hybrid_rerank | `4825cc94f4c1b6d0` | `aca7277eaa10363f` |
| adaptive | `ba9d8bb6f867edba` | `773b9759cf0926bb` |

## 5. Arm input SHA-256 (`experiments/phase8/combined/**/rows.jsonl`)

| Strategy | p8a (after) | p8b (before) |
|---|---|---|
| bm25 | `c3b66d36c793499ff562c4c2c113b800dda2c7c8b825f32ee17676b7a6cb9d01` | `53b506dc005202c0b60edd1d423b28d86f053dd501bad50bdfe207bc187d46a1` |
| dense | `4e7ac1aa3714408877cc2a05d01215b722cca68d43e7124aa356714f6829d4b1` | `73427ed9cc74b7e8b41f2ba3240cccb5b34f7d5cc1627c531d88177dd820a131` |
| hybrid | `1554a69ffc4ed4896c5f2083727989cbf35e78c7b735e65d702fcfa200a9a81b` | `4ada977598f96aa8520d2102e6a9a54329149326287775e9fdbe96724cd525d4` |
| hybrid_rerank | `a8fe1470f78d3fe4b7ab51250621ce7db952a53ca707c25cdee727d9a70b0614` | `9ff6a3e23d7e36fda69aaea8879d353aad342e517afccc00ebb968e719c40ae2` |
| adaptive | `c24295ba36e050c8348407cb641c8a06e7368937027374f1a3c06aa7e7925890` | `7cd9e390181475e2905f7bea94d02cfad65719c10d85d08cc84c570062ac0faa` |

## 6. Artifact SHA-256 (`experiments/phase8/*.json`)

| Artifact | SHA-256 |
|---|---|
| `oracle_ceiling.json` | `bb155994a07a975266666bd64e5ece752ce76a392fdaa2f81c06af3d7e31ef00` |
| `gold_label_audit.json` | `56a7608ac9019089fc331091a86c6ab05f98d9709774e42671ed0a70a9c34fd7` |
| `gold_label_repair.json` | `2c452940a841f2db22a0f9cad94b96e3f8f7f74fe2dea47c05543b4a44c9646f` |
| `run_integrity.json` | `a22031d8b076a32e32cf20f58dc2dcb093c9fa6e024c323a5f522578a7d3630d` |
| `arm_comparison.json` | `1a4b679d64207b87211779d6ba61b742174a3317e19f80aae671ad9d8be11157` |
| `arm_comparison_clean4.json` | `71b63af3ceefcc7a573afc923943abcd2a2c240f7274e3abf6cafd6537cb3b68` |
| `arm_comparison_full5.json` | `ff03ab67b942adad9996996bfcb76c7a9b604127479a75804015663a4362851d` |
| `column_validation.json` | `3741ee032d37f0023b51be089fd18f4c014bbfb7c08f17dc8ce110d69804ffab` |
| `strategy_cost_ms_phase8_after.json` | `a1388af19ebb358ec3e6c8ce4161d5a622eea884ee0b91a691ffbb4ee09027ac` |
| `strategy_cost_ms_phase8_before.json` | `fa8b07320fa884945409a8d07434e0d239deaa42bbfba7a1ee2265d2dee35846` |

## 7. Verify the snapshot

```bash
# artifacts + arm inputs + corpora + dataset (from repo root)
sha256sum experiments/phase8/*.json \
  $(find experiments/phase8/combined -name rows.jsonl | sort) \
  data/processed_phase8_after/*manifest.json data/processed_phase8_before/*manifest.json \
  data/evaluation/phase7_eval_v1.jsonl data/evaluation/phase7_eval_v1.ledger.jsonl
```

Regeneration (documented, not run here): the corpora come from
`scripts/build_index.py`/`ingestion` over the raw PDFs; the E1 arms from
`scripts/run_phase7_suite.py`; the oracle from `scripts/oracle_routing_ceiling.py`.
