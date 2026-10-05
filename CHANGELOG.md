# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- Ollama adapter (`ollama:<model>`) for free local open-weights models; no key or extra package needed
- `mdfaith run --resume <run-id>` to continue interrupted runs; `--temperature`; image-only condition dropped automatically for text-only models
- Grounded claim extraction: model-extracted claims are kept only if their numbers appear in their source sentence
- Extractor defaults: a local extractor for local runs; `--extractor-model` to choose another

### Changed
- The verifier returns *unverifiable* for malformed claim payloads instead of raising
- A stuck claim extraction yields no claims instead of aborting a long run; requests are retried

## [0.1.0] - 2026-10-04

First research-prototype release, delivered as stacked pull requests F01-F12.

### Added
- Core engine: ground truth (RMSD, RMSF, radius of gyration, contacts), planted artifacts, claim schema and programmatic verifier
- Trajectory QC engine and a QC-gated experiment condition (F02)
- Hydrogen-bond ground truth with distance and angle criteria (F03)
- SQLite store (F04), evidence-linked report annotation and a rule-based claim extractor (F05)
- Simulated demo explainers and an Anthropic adapter (F06), experiment runner with bootstrap aggregation (F07)
- FastAPI REST API (F08), web app (F09) and landing website (F10)
- Figures pipeline, architecture docs, poster template and findings registry (F11)
- Docker packaging, README and changelog (F12)

### Known limitations
- No experiments with real models have been run; all results shown are from simulated explainers and labelled as demo data
- The Anthropic adapter has not been exercised against the live API
- One public system (AdK) is bundled
