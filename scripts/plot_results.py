"""Render figures exclusively from saved measured experiment files.

Optional dependency: matplotlib. Run from the repository root.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs" / "figures"
DEST.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})


def save(fig, name):
    fig.savefig(DEST / (name + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(DEST / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


names = ["oracle", "static", "identify", "refresh", "legacy", "schema",
         "random", "linear_set_17", "linear_set_29", "linear_set_43", "linear_single_17"]
labels = ["Finite oracle", "Static", "Identify version", "Full refresh", "Legacy",
          "Explicit scope", "Random", "Set linear / 17", "Set linear / 29",
          "Set linear / 43", "Single linear / 17"]
data = [json.loads((ROOT / "experiments/cpu" / n / "summary.json").read_text())["summary"]
        for n in names]
fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.5, 1]})
y = list(range(len(names)))
axes[0].barh([i - .18 for i in y], [d["success_rate"] * 100 for d in data],
             height=.34, label="Realized success", color="#b8d8e8")
axes[0].barh([i + .18 for i in y], [d["robust_success_rate"] * 100 for d in data],
             height=.34, label="Audited robust success", color="#225b78")
axes[0].set_yticks(y, labels)
axes[0].invert_yaxis()
axes[0].set_xlim(0, 104)
axes[0].set_xlabel("All-test episodes (%)")
axes[0].legend(loc="lower right", fontsize=9)
axes[1].barh(y, [d["mean_query_cost"] for d in data], color="#cf8f42")
axes[1].set_yticks(y, [])
axes[1].invert_yaxis()
axes[1].set_xlabel("Mean query cost (benchmark units)")
fig.suptitle("ToolShift v1: 1,258 held-out episodes / 74 source groups", fontsize=13)
fig.text(.5, -.01, "148 impossible episodes remain in denominators. Lower cost alone is not better when success differs.",
         ha="center", fontsize=9)
fig.tight_layout()
save(fig, "cpu_results")

model_files = [ROOT / "experiments/lora_main" / name / "summary.json"
               for name in ("base", "adapter")]
if all(path.exists() for path in model_files):
    values = [json.loads(p.read_text())["summary"] for p in model_files]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar([-.18, .82], [d["success_rate"] * 100 for d in values], width=.34,
           label="Realized", color="#b8d8e8")
    ax.bar([.18, 1.18], [d["robust_success_rate"] * 100 for d in values], width=.34,
           label="Audited robust", color="#225b78")
    ax.set_xticks([0, 1], ["SmolLM2 135M base", "200-step LoRA"])
    ax.set_ylabel("Success (%)")
    ax.set_ylim(0, 104)
    ax.set_title("Paired 51-episode pilot / 13 source groups")
    ax.legend()
    fig.tight_layout()
    save(fig, "lora_results")
