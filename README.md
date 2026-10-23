# Deep Learning for Computer Vision: Bildklassifikation mit ViT und CNN

Uni-Projekt im Modul *Deep Learning for Computer Vision*. Bildklassifikation mit Vision Transformern und CNNs in PyTorch, aufgeteilt in drei Abgaben (Station 1 bis 3).

## Setup

Voraussetzung ist [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Skripte ausführen

```bash
uv run python scripts/<name>.py
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
