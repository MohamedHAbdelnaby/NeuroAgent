# NeuroAgent Ablations & Auxiliary Experiments

Each subdirectory / script corresponds to one experimental axis from the
NeurIPS proposal Week 7 ablation plan plus the comparison-baseline plan.

## Layout

| Path | Purpose | Status |
|---|---|---|
| `scripts/ablate_agents.py` | Disable each agent (structural/functional/RAG/atlas) and re-run | implemented |
| `scripts/failure_modes.py` | Analyze worst predictions per condition; cluster error patterns | implemented |
| `scripts/hallucination_eval.py` | Atlas-grounded vs free-form spatial reasoning, count ungrounded region claims in LLM text | implemented |
| `scripts/cross_condition.py` | Use one condition's prompt/strategy on another (e.g. tumor -> stroke) | implemented |
| `scripts/early_stopping_eval.py` | Active tool-use cost minimization, measure tokens vs accuracy with stop-when-confident | planned |
| `scripts/multimodal_vlm.py` | Feed 2D mid-axial slice + prompt to a VLM (BiomedCLIP / LLaVA-Med) | planned |
| `scripts/run_3d_cnn_baseline.sh` | Run the existing baselines/cnn_3d.py for ADHD/tumor | planned |
| `scripts/run_nnunet_baseline.sh` | nnU-Net tumor segmentation; lesion-symptom mapping for stroke | planned |
| `scripts/medagent_pro_adapter.py` | Adapt MedAgent-Pro hierarchical workflow to brain MRI | planned |
| `scripts/llm_judge_reasoning.py` | LLM-as-judge for evidence-chain quality | planned |
| `submit_ablations.sh` | One SLURM job per ablation cell | implemented |
| `results/` | All output `summary.csv` / `per_subject.jsonl` / `*.json` | |
| `logs/` | SLURM stdout/stderr | |

## Convention

- Each ablation writes to `results/<ablation_name>/<model>/<task>_per_subject.jsonl`
  and `<task>_results.json` so we can use the same metric-aggregation code as
  the main study.
- Disable flags are env-var driven so the orchestrator stays a single
  codepath:
  - `NEUROAGENT_DISABLE_STRUCTURAL=1`
  - `NEUROAGENT_DISABLE_FUNCTIONAL=1`
  - `NEUROAGENT_DISABLE_RAG=1`
  - `NEUROAGENT_DISABLE_ATLAS=1`
  - `NEUROAGENT_FORCE_FREEFORM_REGIONS=1` (no atlas grounding in finding text)

## Running

Compute heavy ablations all go via SLURM:
```
sbatch ablations/submit_ablations.sh
```

Light analysis (failure modes, hallucination eval, cross-condition with
existing artifacts) runs fast on a compute node:
```
srun -A aml -p t4_normal_q -q fal_t4_normal_base --gres=gpu:1 \
     -c 4 --mem=16G -t 1:00:00 --pty bash
python ablations/scripts/failure_modes.py --study study_postfix2
```
