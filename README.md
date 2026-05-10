# NeuroAgent

A disease-agnostic multi-agent LLM framework for interpretable 3D neuro-imaging classification on ADHD-200, BraTS HGG vs LGG, and ATLAS-v2 stroke lateralization.

It contains the agent code, evaluation harnesses, classifier-fitting scripts, and an ablation pipeline.

## What's in here

```
agents/         5 tool agents + orchestrator + JSON schemas
data/           Per-task classifier weights (LOSO logreg, 5-fold CV logreg) and norm tables
ablations/      6 x 2 x 3 ablation pipeline (run_one_ablation.sh + scripts/ablate_agents.py)
baselines/      Direct-LLM and 3D CNN baselines, vLLM serving harness
eval/           Comprehensive evaluation grid (5 LLMs x 4 modes x 3 tasks)
streamlit_demo/ Interactive walkthrough on bundled sample subjects
run_neuroagent.py  Single-subject driver
```

## Setup

```bash
conda env create -f environment.yml
conda activate neuroagent
```

`vLLM` is needed only if you want to serve open-weight LLMs locally for the agent runs; it is not required for the classifier-fitting scripts. Install separately on a CUDA host:

```bash
pip install vllm
```

## Environment variables

All scripts read paths from environment variables so the repo is portable across machines and clusters.

| Variable | Required? | What it points to | Default |
|---|---|---|---|
| `PROJECT` | No (auto-detected) | Absolute path of this repo | Resolved from script location |
| `VENV_PATH` | **Yes** for SLURM scripts | Conda env or virtualenv root (the dir containing `bin/activate`) | None, scripts fail loudly if unset |
| `MODEL_ROOT` | No | Directory containing local LLM weights (used by vLLM) | `/common/data/models` |
| `FS_LICENSE` | **Yes** for FreeSurfer/fMRIPrep | Path to your `license.txt` for FreeSurfer | None, scripts fail loudly if unset |
| `CUDA_HOME` | No | CUDA toolkit root | `/cm/shared/apps/cuda12.6/toolkit/12.6.2` |
| `OPENAI_API_KEY` | Only if using a hosted LLM | API key | None |
| `OPENAI_BASE_URL` | Only if pointing at vLLM or another OpenAI-compatible endpoint | Base URL | OpenAI default |

Typical setup before running anything on a cluster:

```bash
export PROJECT=/abs/path/to/this/repo
export VENV_PATH=/abs/path/to/your/venv
export FS_LICENSE=/abs/path/to/license.txt
export MODEL_ROOT=/abs/path/to/models       # only if your models live elsewhere
```

The SLURM `--partition`, `--qos`, and `--account` directives are still hard-coded inside each `*.sh`/`*.sbatch` script (because SBATCH directives cannot read shell variables). Edit them to match your cluster before submitting.

## Reproducing the experiments

### Apples-to-apples evaluation grid (5 LLMs x 4 modes x 3 tasks)

The full grid is driven by `eval/comprehensive_study.py`. After you have a vLLM endpoint running for each model, submit one SLURM job per (model, mode) pair:

```bash
sbatch baselines/submit_v2_run.sh \
    --export=ALL,SERVED_NAME=Llama-3.1-8B-Instruct,MODEL_PATH=Llama-3.1-8B-Instruct
```

Or run a single cell directly:

```bash
python eval/comprehensive_study.py run \
    --models Llama-3.1-8B-Instruct \
    --tasks adhd_binary tumor_grade stroke_lat \
    --modes neuroagent neuroagent_llm direct_matched direct \
    --max-subjects 9999 \
    --resume
```

### Per-component ablation grid (6 ablations x 2 LLMs x 3 tasks)

```bash
sbatch ablations/submit_ablations.sh
```

The harness sets `NEUROAGENT_DISABLE_{STRUCTURAL,FUNCTIONAL,RAG,ATLAS}` per cell. Results land in `ablations/results/<ablation>/<model>__neuroagent/`.

### Re-fitting the classifiers from scratch

```bash
python data/fit_adhd_loso_classifier.py
python data/fit_tumor_grade_classifier.py
```

These produce `data/adhd200_loso_logreg.json` and `data/brats_grade_logreg.json` which the agent reads at runtime.

## Single-subject inference

```bash
python run_neuroagent.py \
    --subject-id sub-0010001 \
    --task adhd_binary \
    --model Llama-3.1-8B-Instruct
```

The agent emits a structured evidence record and a per-subject diagnostic report.

## Datasets

The three datasets are public. We do not redistribute the raw imaging here; the FastSurfer-derived per-subject feature CSVs and the per-subject reasoning traces produced by our agent runs are at `eval/results/`.

- ADHD-200: http://fcon_1000.projects.nitrc.org/indi/adhd200/
- BraTS: https://www.synapse.org/Synapse:syn53708249
- ATLAS-v2: https://atlas.grand-challenge.org/

## License

Released under the MIT License; see `LICENSE` at the repository root.
