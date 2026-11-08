# Analyzing esports matches using deep learning

Code and experiment artefacts accompanying my master's thesis (full text: [`MAGISTERKA.pdf`](MAGISTERKA.pdf)).

The experiments compare several model families (DeepSets, Set Transformer, GCN, GAT, temporal GNN, LightGBM)
on per-map Counter-Strike 2 round data, evaluating predictive log-likelihood and calibration (ECE),
including ablations on training-set size and on hidden-information/edge features.

## Repository layout

| Path | Contents |
|---|---|
| `dataset.py` | PyTorch / PyG `Dataset` classes and data-loading helpers shared by all notebooks |
| `training_notebook.ipynb` | Training notebook (template used for the per-map runs in `results/`) |
| `ablation/` | Training-set-size ablation on Dust2 (notebook, CSV, plot) |
| `visualisation_notebook.ipynb` | Figures |
| `comparison/` | Cross-model evaluation notebook (hidden-32 phase/ECE) |
| `results/trained64/`, `results/trained32/` | Executed notebooks (with outputs) per map for the 64- and 32-dim models |
| `results/32eval/` | Per-map evaluation notebooks for the 32-dim models |
| `checkpoints/<map>/` | Trained weights (`.pt`, `lightgbm.txt`), normalisers and figures per map |

Maps: ancient, anubis, dust2, inferno, mirage, nuke, overpass, and `train` (pooled training set).

## Data (not published)

The processed match data used for training and evaluation is not part of this repository and is not publicly
distributed. `dataset.py` documents the expected per-match JSON structure (keys `df`, `X`, `Y`, `GNN`), so the
code can be read and adapted, but the experiments cannot be re-run without equivalent data. The executed notebooks
in `results/` contain all outputs, so the reported results can be inspected without re-running anything.

## Running

```bash
pip install -r requirements.txt   # pick the torch build matching your CUDA setup
jupyter lab
```

The notebooks were run on Google Colab and set `BASE = Path('/content/drive/MyDrive/magisterka')`.
Change `BASE` to your local checkout before running (requires your own data in the layout `dataset.py` expects).

## Licence

- Code, notebooks and trained weights: [MIT](LICENSE)
- Thesis text and figures (`MAGISTERKA.pdf`): [CC BY 4.0](LICENSE-THESIS.md)

## Citation

Tramś, B. (2026). *Analyzing esports matches using deep learning*. Master's thesis, Wrocław University of Science and Technology.

```bibtex
@mastersthesis{tramsh2026esports,
  author = {Tramś, Bartosz},
  title  = {Analyzing esports matches using deep learning},
  school = {Wrocław University of Science and Technology},
  year   = {2026}
}
```
