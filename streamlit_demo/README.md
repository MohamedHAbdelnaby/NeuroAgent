# NeuroAgent: Streamlit Demo

Self-contained, portable demo of the NeuroAgent multi-agent neuroimaging framework.
All sample data is bundled, no GPUs, no API keys, no model downloads required.

## What you get

Three demo subjects walking through the full pipeline:

| Subject | Condition | Modalities |
|---|---|---|
| `demo_ADHD_001` | ADHD (combined type), 11 y/o | T1w, fMRI |
| `demo_TUMOR_001` | Right hemisphere glioma, 56 y/o | T1w, T1c, segmentation |
| `demo_STROKE_001` | Left MCA chronic infarct, 62 y/o | T1w |

Each subject has 5 tabs:
1. **Imaging viewer**: pre-rendered axial / coronal / sagittal slices
2. **FastSurfer output**: sample structural-volumetrics text
3. **Agent pipeline**: every tool call (input args + JSON output)
4. **Final report**: synthesized diagnostic markdown + downloads
5. **About**: architecture explanation

## Quick start

### 1. Set up a Python environment (any version >= 3.10)

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the demo

```bash
streamlit run app.py
```

This opens a browser at `http://localhost:8501` automatically.

### Or use the helper script (Linux / macOS):

```bash
bash run.sh
```

## Folder layout

```
streamlit_demo/
├── app.py                # Main Streamlit app (single file, ~400 lines)
├── requirements.txt      # 3 deps: streamlit, numpy, Pillow
├── run.sh                # Helper launcher
├── README.md             # This file
└── data/
    └── slices/           # 18 pre-rendered PNG brain slices (~250 KB total)
        ├── adhd_T1w_axial.png
        ├── adhd_T1w_coronal.png
        ├── adhd_T1w_sagittal.png
        ├── adhd_fMRI_*.png
        ├── tumor_T1w_*.png
        ├── tumor_T1c_*.png
        ├── tumor_Segmentation_*.png      # T1c with colored seg overlay
        └── stroke_T1w_*.png
```

## Notes

- All agent outputs (FastSurfer stats, structural findings, fMRI connectivity,
  atlas mappings, RAG citations, final reports) are **hardcoded sample data**
  designed to look realistic for a presentation
- No live LLM calls, runs offline
- Imaging slices are pre-rendered from real BraTS-2024, ADHD-200, and ATLAS-v2 data
- Segmentation overlay uses BraTS color convention: green=NCR, yellow=ED, red=ET

## Customize

- Edit `SAMPLE_*` dicts in `app.py` to change the demo content
- Drop new PNGs into `data/slices/` named `{subject}_{modality}_{axis}.png`
- The synthetic-brain fallback runs if a slice PNG is missing

## License

Educational / demo use.
