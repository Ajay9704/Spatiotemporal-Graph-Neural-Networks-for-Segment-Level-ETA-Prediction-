# Spatiotemporal Graph Neural Networks for Segment-Level ETA Prediction

**Repository:** https://github.com/Ajay9704/Spatiotemporal-Graph-Neural-Networks-for-Segment-Level-ETA-Prediction-.git


---

## Project overview

I built and evaluated Graph Neural Network (GNN) models to predict short-term travel times (STT / AccSTT) on the Dublin road network.  
I use a **line-graph transformation** so that road **links (TCS1–TCS2)** become the graph **nodes**, and link-level travel-times (STT) become node features and targets. I implemented two model families (TGC‑RN and TGC‑CN), trained them on the Dublin trips dataset, and saved training/evaluation artifacts and figures.

**Research question:**  
> Can the Estimated Time of Arrival on a Dublin Road Network be accurately predicted using Graph Neural Networks?

**Aims**
- Investigate how GNNs model spatial and temporal dynamics for ETA prediction.
- Compare model variants and analyze strengths / weaknesses.
- Visualize and interpret factors that impact ETA predictions.

---

## Files in this repository

```
.
├─ data/
│  ├─ trips-1.csv          # (subset or 1-day sample used in experiments)
│  ├─ trips.csv            # (full trips dataset, if available)
│  └─ routes.kml           # KML file with TCS coordinates
├─ notebooks/
│  └─ ETA_GNN_notebook.ipynb   # end-to-end Colab-friendly notebook
├─ src/
│  ├─ preprocess.py       # data loading & line-graph conversion
│  ├─ dataloader.py       # builds torch_geometric Data objects and DataLoaders
│  ├─ models.py           # TGCRN, TGCCN model classes
│  └─ train_eval.py       # training, evaluation, metrics logging, saving
├─ results/
│  ├─ metrics.csv
│  └─ figures/
│     ├─ train_loss.png
│     ├─ eval_plot_timestamp0.png
│     └─ metric_summary.png
├─ requirements.txt
└─ README.md
```

> I included `trips-1.csv`, `trips.csv` and `routes.kml` in the `data/` directory — these are the exact files I used.

---

## Data description & preprocessing

**Key definitions**
- **Route**: a group of two or more Control Sites along a common roadway (one or more links).
- **Link**: two adjacent junctions (TCS) and the road segment between them.
- **STT**: Smoothed Travel Time (link-level travel time).
- **AccSTT**: Accumulated Smoothed Travel Time.
- **TCS**: Traffic Control Site (SCATS site ID, unique city-wide).

**Loading & initial cleaning**
- I load `trips-1.csv` (or `trips.csv` for larger runs) and `routes.kml`.
- Parse `Timestamp` (format `YYYYMMDD-HHMM`) into datetime and filter the time-window used in experiments.
- Drop rows with missing values (or optionally impute/mask them).

**Line-graph transformation**
- Each directed link is represented as a node named `TCS1-TCS2` (direction-aware).
- Two nodes are connected if the original links share a TCS (i.e., common junction).
- Node features: STT values for both directions (columns created via pivoting).
- Target `y`: STT values at `T+1` (next timestamp). This is a supervised one-step-ahead setup.

**Pivoting & normalization**
- Pivot by `['Timestamp', 'node']` to create a table of node features per timestamp.
- Fill missing direction values with `0` (or use masking).
- Normalize target values using training-set mean and std; denormalize for final reporting.

---

## Model implementations

I implemented the following models in `src/models.py`:

1. **TGC‑RN** (Temporal Graph Convolutional Recurrent Network)  
   - `GCNConv` → `GRU` → `Linear`. Predicts STT for both directions.
2. **TGC‑CN** (Temporal Graph Chebyshev Convolution Network)  
   - `ChebConv` → `LSTM` → `Linear`. Another temporal‑spatial baseline.

Models accept node features `x` and `edge_index` and return node-wise predictions `y_pred`.

---

## How to run (Colab-friendly)

### 1) Requirements
Install dependencies:
```bash
pip install -r requirements.txt
# minimal:
pip install pandas numpy matplotlib seaborn scikit-learn folium torch torch-geometric torch-geometric-temporal
```

### 2) Run in Google Colab (recommended)
1. Upload `data/trips-1.csv`, `data/trips.csv`, and `data/routes.kml` to your Google Drive (e.g. `My Drive/TRIPS/`).
2. Open `notebooks/ETA_GNN_notebook.ipynb` in Colab.
3. Mount Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
DATA_CSV = "/content/drive/My Drive/TRIPS/trips-1.csv"
KML_PATH = "/content/drive/My Drive/TRIPS/routes.kml"
```
4. Install packages (run once in Colab):
```bash
!pip install --quiet pandas numpy matplotlib seaborn scikit-learn folium
!pip install --quiet torch torchvision torchaudio
!pip install --quiet torch-geometric torch-geometric-temporal
```
5. Run cells in sequence: preprocessing → graph construction → dataloaders → train → evaluate → visualize.

**RAM tips:** If Colab crashes, set `DATA_SUBSET = 1500` (or 500) in the notebook and reduce batch size / hidden dims.

### 3) Run locally (script mode)
Preprocess:
```bash
python src/preprocess.py --input data/trips-1.csv --kml data/routes.kml --out data/processed.pt --subset 1500
```
Train:
```bash
python src/train_eval.py --mode train --data data/processed.pt --model TGCRN --epochs 50 --batch_size 32 --lr 0.001
```
Evaluate:
```bash
python src/train_eval.py --mode eval --model_path models/tgcrn.pt --data data/processed.pt
```

---

## Training & evaluation details

- **Loss:** MSE (Mean Squared Error).  
- **Optimizer:** Adam.  
- **Metrics reported:** Test Loss, MAE, MSE, RMSE, R².  
- **Data split:** 80% train timestamps / 20% test timestamps.


---


## Reproducibility & tips

- Ensure timestamps are sorted before building graphs.
- Keep a consistent node ordering across timestamps (I sort node keys lexicographically).
- If you change prediction horizon or timesteps, rebuild the processed dataset.
- Use `torch.save` to persist processed datasets and model checkpoints for fast reload.

---

## Acknowledgements & references

- Dublin City Open Data — Trips dataset (source of `trips-1.csv` and `routes.kml`).  
- PyTorch Geometric & torch-geometric-temporal.  
- Standard graph theory references for line-graph transformations.

---


**Repository remote:**  
`git remote add origin https://github.com/Ajay9704/Spatiotemporal-Graph-Neural-Networks-for-Segment-Level-ETA-Prediction-.git`
