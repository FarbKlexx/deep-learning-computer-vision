from src.models.linear_probing import ViTLinearProbing

if __name__ == "__main__":
    ViTLinearProbing().run(out="results/station1/linear_probe", epochs=10)