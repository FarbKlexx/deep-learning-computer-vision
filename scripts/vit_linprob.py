from src.models.linear_probing import ViTLinearProbing
import sys

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        ViTLinearProbing(lr=float(arg)).run(out="results/station1/linear_probe", epochs=10)