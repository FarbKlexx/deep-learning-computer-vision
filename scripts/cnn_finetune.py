from src.models.cnn import CNNFinetune

# main-guard for macOS
if __name__ == "__main__":
    CNNFinetune(
        lr=1e-4,  # Lernrate für das ganze Netz
    ).run(out="results/station1", epochs=10)
