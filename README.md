# Deep Learning for Computer Vision: Bildklassifikation mit ViT und CNN

Uni-Projekt im Modul *Deep Learning for Computer Vision*. Bildklassifikation mit Vision Transformern und CNNs in PyTorch, aufgeteilt in drei Abgaben (Station 1 bis 3).

## Setup

Voraussetzung ist Python 3.13 oder neuer. Installation entweder mit [uv](https://docs.astral.sh/uv/) oder mit pip.

**uv** (nutzt `pyproject.toml` und `uv.lock`):

```bash
uv sync
```

**pip** (nutzt `requirements.txt`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Neue Abhängigkeiten in beiden Dateien eintragen: mit `uv add <paket>` in `pyproject.toml` und von Hand in `requirements.txt`.

## Skripte ausführen

Aus dem Projektordner:

```bash
uv run python -m scripts.<name>   # mit uv
python -m scripts.<name>          # mit pip, bei aktivierter .venv
```

## ML-Server

`ml2-ifm-min.ad.fh-bielefeld.de` (Ubuntu 24.04)

- GPU: 8× NVIDIA A16 mit je 15 GB VRAM (Treiber 580)
- CPU: AMD EPYC 9124, 16 Kerne / 32 Threads
- RAM: 188 GB
- Speicher: ca. 6,4 TB frei unter `/home`

Aktuelle GPU-Auslastung prüfen: `nvidia-smi`

## Ordner

- `data/`: Datensätze (nicht im Repo)
- `src/data/`: Dataset-Klassen und Transforms
- `src/models/`: Modell-Code
- `src/utils/`: Hilfsfunktionen, z. B. `get_device()`
- `scripts/`: ein Skript pro Experiment
- `notebooks/`: Exploration
- `results/station1/` bis `results/station3/`: Metriken und Plots (Checkpoints werden nicht committet)
- `report/station1/`: LaTeX-Ausarbeitung
