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

## Ordner

- `data/`: Datensätze (nicht im Repo)
- `src/data/`: Dataset-Klassen und Transforms
- `src/models/`: Modell-Code
- `src/utils/`: Hilfsfunktionen, z. B. `get_device()`
- `scripts/`: ein Skript pro Experiment
- `notebooks/`: Exploration
- `results/station1/` bis `results/station3/`: Metriken und Plots (Checkpoints werden nicht committet)
- `report/station1/`: LaTeX-Ausarbeitung
