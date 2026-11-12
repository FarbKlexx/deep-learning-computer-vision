import json
from pathlib import Path

import pandas as pd

OUT = Path("results/station1")
rows = [json.loads(p.read_text()) for p in OUT.glob("*.json")
        if not p.name.endswith("_history.json") and p.name != "drop.json"]

df = pd.DataFrame(rows).sort_values("trainable")
df["trainable_pct"] = 100 * df["trainable"] / df["total"]
cols = ["name", "test_acc", "trainable", "trainable_pct", "sec_per_epoch", "peak_vram_gb"]
print(df[cols].to_string(index=False))
df[cols].to_csv(OUT / "comparison.csv", index=False)
