# Deep Learning for Computer Vision: Bildklassifikation mit ViT und CNN

Uni-Projekt im Modul *Deep Learning for Computer Vision*. Bildklassifikation mit Vision Transformern und CNNs in PyTorch.

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

## Ordner-Struktur

- `data/`: Datensätze (nicht im Repo)
- `src/data/`: Dataset-Klassen und Transforms
- `src/models/`: Modell-Code
- `src/training/`: Trainer und Evaluation
- `src/utils/`: Hilfsfunktionen
- `scripts/`: ein Skript pro Experiment, direkt ausführen per Bash
- `notebooks/`: Exploration
- `results/station1/` bis `results/station3/`: Metriken und Plots (Checkpoints werden nicht committet)
- `report/station1/`: LaTeX-Ausarbeitung

## Architektur

### Generische Files

Die folgenden Dateien sind generisch und enthalten wiederverwendbare Funktionen die von allen Modell-Trainingsläufen benutzt werden, um vergleichbare Ergebnisse sicherzustellen. Sie abstrahieren die konkrete implementation:

- `loading.py`: Lädt den gewählten Datensatz (tanganke/sun397) von Huggingface und bereitet DataLoaders für das Training vor
- `trainer.py`: Enthält die Trainingsloop
- `evaluation.py`: Evaluierung eines Modells
- `experiment.py`: Abstrakte Klasse von der jedes Modell erbt, damit die Struktur gleich bleibt unabhängig vom Modell

### Modelle

Die folgenden Dateien enthalten die eigentlichen Modelle, sie erben alle von `experiment.py`:

- `vit_lora.py`
- `cnn.py`

Die Trainingsparameter werden in dem Ausführungs-Skript eines  entsprechenden Modells gesetzt. Um sie zu verändern müssen diese Dateien direkt bearbeitet werden.

## Skripte ausführen

Aus dem Projektordner:

```bash
uv run python -m scripts.<name>   # mit uv
python -m scripts.<name>          # mit pip, bei aktivierter .venv
```

Beispiel für den CNN-Lauf:

```bash
uv run python -m scripts.cnn_finetune.py
python -m scripts.cnn_finetune.py
```

Für die Datenanalyse und -bereinigung das Jupyter Notebook unter `notebooks/dataset_analysis.ipynb` ausführen. Es erezugt die Datei `drop.json`, die beschreibt welche Datenpunkte beim Training ausgelassen werden. Wenn der Seed verändert wird muss das Skript auf jeden Fall erneut gestartet werden. Ansonsten liegt die Datei für den Seed 67 im Repo vor.

## ML-Server

Wir trainieren auf dem ML2 Server der HSBI.

`ml2-ifm-min.ad.fh-bielefeld.de` (Ubuntu 24.04)

- GPU: 8× NVIDIA A16 15 GB VRAM
- CPU: AMD EPYC 9124, 16 Kerne / 32 Threads
- RAM: 188 GB
- Speicher: 6,4 TB frei

## Quellen

Der Quellcode dieses Projektes ist inspiriert von den folgenden Quellen, sowie den offiziellen Dokus der verwendeten Bibliotheken.

### PyTorch- und Huggingface Dokumentationen

#### Generische Architektur

- [Python abc](https://docs.python.org/3/library/abc.html)
- [Build the Neural Network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)

#### Trainings- und Evaluationsloop

- [Training with PyTorch](https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html)
- [Transfer Learning for Computer Vision](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Optimizing Model Parameter](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)

#### Optimizer, Scheduler, etc

- [torch.optim](https://docs.pytorch.org/docs/2.15/optim.html)
- [AdamW](https://docs.pytorch.org/docs/2.15/generated/torch.optim.AdamW.html)
- [CosineAnnealingLR](https://docs.pytorch.org/docs/2.15/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html)

#### Daten und Augmentierung

- [Datasets](https://huggingface.co/docs/datasets/main/en/index)
- [DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [Transforms v2](https://docs.pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html)

#### ViT

- [timm](https://huggingface.co/docs/timm/index)

#### LoRA

- [PEFT-LoRA](https://huggingface.co/docs/peft/package_reference/lora)
- [Apply LoRA on timm-Model](https://huggingface.co/docs/peft/developer_guides/custom_models)

## AI-Disclaimer

In diesem Projekt wurde an einigen Stellen generative KI genutzt, unter anderem um das Projekt-Setup (Ordner- und Klassenstruktur, Hilfsklassen wie `device.py` und `uv`/`pip` Setup), sowie eine saubere und einheitliche Dokumentation des Codes zu erstellen. Das gesamte Projekt stammen aus den eigenen Händen der Autoren und orientiert sich an den offiziellen Dokumentationen.
