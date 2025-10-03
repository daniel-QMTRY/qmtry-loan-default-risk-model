# src/update_readme.py
import os

README_PATH = "README.md"

def update_readme():
    """
    Appends Phase 3 training plots to README.md automatically.
    Overwrites old plot section if present.
    """
    plots = {
        "Training & Validation Loss": "models/loss_curve.png",
        "ROC-AUC over Epochs": "models/roc_auc_curve.png",
        "ROC Curve": "models/roc_curve.png",
        "Precision-Recall Curve": "models/pr_curve.png"
    }

    # Read existing README
    with open(README_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove any old plots section
    marker_start = "<!-- PLOTS_START -->"
    marker_end = "<!-- PLOTS_END -->"
    if marker_start in content and marker_end in content:
        before = content.split(marker_start)[0]
        after = content.split(marker_end)[-1]
        content = before + after

    # Build new plots section
    plots_md = [f"### 📊 Model Training Results\n{marker_start}\n"]
    for title, path in plots.items():
        if os.path.exists(path):
            plots_md.append(f"**{title}:**\n\n![]({path})\n")
    plots_md.append(marker_end)

    # Merge and overwrite README
    new_content = content.strip() + "\n\n" + "\n".join(plots_md)
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)

    print("✅ README.md updated with plots.")


if __name__ == "__main__":
    update_readme()
