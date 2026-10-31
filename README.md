# PCC-SQL: Policy-Conditioned Constrained Decoding for Column-Level Access Control in Text-to-SQL

[![arXiv](https://img.shields.io/badge/arXiv-2607.12341-b31b1b.svg)](https://arxiv.org/abs/2607.12341)

Ryoto Miyamoto, Xin Fan, Hayato Yamana

AACL-IJCNLP 2026

## Overview

- Whether a Text-to-SQL query complies with a column-level access policy
  depends not only on which columns it mentions but on how it uses them.
  We formalize this as a **column-use policy**: each column is Public,
  ConditionOnly, AggregateOnly, or Hidden, and is checked against its role
  in the query (output, join / filter condition, or aggregation argument).
- **PCC-SQL** (Policy-Conditioned Constrained SQL Generation) tracks the
  role implied by the current SQL grammar position during decoding and
  applies a per-token logits mask that excludes columns disallowed in that
  role. This deterministically eliminates single-query column-use
  violations on the supported SQL fragment in a single decoding pass.
- Across Spider-CU, BIRD-CU, and Spider-ACL with three open-source models,
  PCC-SQL achieves 0% Leakage Rate and Coverage up to 88.7% on Spider-CU,
  while staying within +10% tokens of direct prompting.

## Repository Layout

```
pcc-sql/
├── data/dev.json          # Spider-CU benchmark
├── src/
│   ├── constrained/       # PCC-SQL decoder
│   └── sqlchecks.py       # policy / syntax checks
├── eval/
│   ├── run_eval.py        # entry point
│   ├── baselines.py       # direct / retry / 2-step verifier
│   ├── prompt.py
│   └── scorer.py
├── benchmark/             # Spider-CU construction pipeline + Spider-ACL converter
├── setup_spider.sh        # fetch Spider 1.0 into spider_data/
└── setup_spider_acl.sh    # fetch and convert Spider-ACL into data_spider_acl/
```

## Installation

```bash
git clone https://github.com/yamanalab/pcc-sql.git
cd pcc-sql
uv sync
bash setup_spider.sh
```

## Usage

```bash
# PCC-SQL on Spider-CU
uv run python -m eval.run_eval \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --data-dir data \
  --schema-dir spider_data/spider_data

# Spider-ACL
bash setup_spider_acl.sh
uv run python -m eval.run_eval \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --data-dir data_spider_acl \
  --schema-dir data_spider_acl
```

### Baselines

`--method` selects PCC-SQL (default) or one of the baselines from the paper:

| `--method` | Baseline |
|---|---|
| `pcc-sql` | PCC-SQL (this work) |
| `direct` | direct prompting |
| `retry` | regenerate with violation feedback, up to `--retries` times (default 3) |
| `verifier` | 2-step verifier (Klisura et al., 2026) |

The baselines also accept Claude models via the Anthropic API:

```bash
uv sync --extra claude
export ANTHROPIC_API_KEY=...
uv run python -m eval.run_eval --method retry --model claude-haiku-4-5
```

BIRD-CU will be added later.

## Cite as

```bibtex
@misc{miyamoto2026policyconditionedconstraineddecodingcolumnlevel,
      title={Policy-Conditioned Constrained Decoding for Column-Level Access Control in Text-to-SQL},
      author={Ryoto Miyamoto and Xin Fan and Hayato Yamana},
      year={2026},
      eprint={2607.12341},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.12341},
}
```

## License

Code (`src/`, `eval/`, `benchmark/`) is released under the [MIT License](LICENSE).

`data/dev.json` is derived from Spider 1.0 (Yu et al., 2018) and is
distributed under CC BY-SA 4.0, the same license as Spider. This
repository does not redistribute the Spider SQLite databases;
`setup_spider.sh` fetches them from the official release.
