import json
import os
from copy import deepcopy
import mpld3   


class NotebookOptimizer:
    def __init__(self, notebook_path):
        if not os.path.exists(notebook_path):
            raise FileNotFoundError(f"Notebook '{notebook_path}' not found.")
        
        self.notebook_path = notebook_path
        
        with open(notebook_path, "r", encoding="utf-8") as f:
            self.nb = json.load(f)

    # ---------------------------------------------------------
    # Remove ALL outputs (Plotly, Matplotlib, logs, etc.)
    # ---------------------------------------------------------
    def strip_outputs(self):
        """Remove all cell outputs and execution counts."""
        for cell in self.nb.get("cells", []):
            if "outputs" in cell:
                cell["outputs"] = []
            cell["execution_count"] = None
        print("✔ All cell outputs stripped.")

    # ---------------------------------------------------------
    # Remove metadata (sometimes huge)
    # ---------------------------------------------------------
    def clean_metadata(self):
        """Remove heavy metadata fields."""
        if "metadata" in self.nb:
            self.nb["metadata"] = {}
        print("✔ Notebook metadata cleaned.")

    # ---------------------------------------------------------
    # Detect large cell outputs, warn user
    # ---------------------------------------------------------
    def detect_large_cells(self, threshold_kb=500):
        """Warn about large output cells."""
        warnings = []
        for i, cell in enumerate(self.nb.get("cells", [])):
            cell_str = json.dumps(cell)
            size_kb = len(cell_str) / 1024
            
            if size_kb > threshold_kb:
                warnings.append((i, round(size_kb, 2)))
        
        if warnings:
            print("⚠ Large cells detected:")
            for idx, size in warnings:
                print(f" - Cell {idx} → {size} KB")
        else:
            print("✔ No oversized cells found.")

    # ---------------------------------------------------------
    # Save cleaned version
    # ---------------------------------------------------------
    def save_clean_version(self, output_path=None):
        """Save optimized notebook."""
        if output_path is None:
            output_path = self.notebook_path.replace(".ipynb", "_CLEAN.ipynb")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.nb, f, indent=1)

        print(f"✔ Optimized notebook saved as: {output_path}")
        
    def save_matplotlib_figure(self, fig, filename, folder="Visualizations"):
        """
        Save a matplotlib figure as interactive HTML.
        
        Parameters:
        - fig: matplotlib figure object
        - filename: name of HTML file (without extension)
        - folder: folder to save HTML into (default "Visualizations")
        """
        # Ensure folder exists
        os.makedirs(folder, exist_ok=True)
        
        # Full path to save
        filepath = os.path.join(folder, f"{filename}.html")
        
        # Convert figure to HTML
        html_str = mpld3.fig_to_html(fig)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_str)
        
        print(f"✔ Matplotlib figure saved as HTML: {filepath}")
        return filepath

