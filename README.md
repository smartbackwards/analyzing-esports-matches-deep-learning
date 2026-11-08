# Master's thesis – code, notebooks and results

Code and experiment artefacts accompanying my master's thesis (full text: [`MAGISTERKA.pdf`](MAGISTERKA.pdf)).

The experiments compare several model families (DeepSets, Set Transformer, GCN, GAT, temporal GNN, LightGBM)
on per-map Counter-Strike 2 round data, evaluating predictive log-likelihood and calibration (ECE),
including ablations on training-set size and on hidden-information/edge features.

## Repository layout

| Path | Contents |
|---|---|
| `dataset.py` | PyTorch / PyG `Dataset` classes and data-loading helpers shared by all notebooks |
| `train_map.ipynb`, `train_map_v2.ipynb`, `train_dust2.ipynb`, `training_notebook.ipynb` | Training notebooks (per-map training) |
| `ablation_training_size_dust2.ipynb` / `.csv` / `.png` | Training-set-size ablation on Dust2 |
| `visualisation_notebook.ipynb` | Figures |
| `comparison/` | Cross-model evaluation notebook (hidden-32 phase/ECE) |
| `results/trained64/`, `results/trained32/` | Executed notebooks (with outputs) per map for the 64- and 32-dim models |
| `results/32eval/` | Per-map evaluation notebooks for the 32-dim models |
| `checkpoints/<map>/` | Trained weights (`.pt`, `lightgbm.txt`), normalisers and figures per map |

Maps: ancient, anubis, dust2, inferno, mirage, nuke, overpass, and `train` (pooled training set).

## Data (not included)

The processed match files (`data/<map>/season*/…json`, ≈ 49 GB) are too large for GitHub and are not part of this
repository. `dataset.py` expects per-match JSON files with keys `df`, `X`, `Y`, `GNN`, as produced by the
preprocessing step (`preprocess.py`, which is not included here). Expected layout:

```
data/<map>/{season1,season2,current}/<event>_<team>-vs-<team>-<map>.json
data/train/{season1,season2}/…-train.json
```

> TODO (author): add a link to where the data can be obtained (e.g. Zenodo / institutional storage) and
> state the source and licence of the underlying demo data.

## Running

```bash
pip install -r requirements.txt   # pick the torch build matching your CUDA setup
jupyter lab
```

The notebooks were run on Google Colab and set `BASE = Path('/content/drive/MyDrive/magisterka')`.
Change `BASE` to your local checkout (and point `data/` at the downloaded data) before running.
The notebooks in `results/` already contain their outputs, so the results can be inspected without re-running anything.

## Citation / licence

> TODO (author): add the thesis citation (author, title, university, year) and choose a licence.
