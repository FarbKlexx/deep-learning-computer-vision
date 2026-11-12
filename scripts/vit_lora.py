from src.models.vit_lora import ViTLoRA

ViTLoRA(
    rank=8,                            # Rang der LoRA-Matrizen
    targets=("qkv", "proj"),           # Schichten, die LoRA bekommen
    lr=1e-3,                           # Lernrate für LoRA-Matrizen und Kopf
).run(out="results/station1", epochs=10)
