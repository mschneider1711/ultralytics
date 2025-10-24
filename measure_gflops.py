import torch
from ultralytics import YOLO
from pathlib import Path
import pandas as pd
from thop import profile
import time

# ======================================
# KONFIGURATION
# ======================================
ROOT_DIR = Path("/Users/marcschneider/sciebo/Modelle_transfer")
OUTPUT_XLSX = Path("gflops_results.xlsx")
IMGSZ = 640
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================
# FUNKTIONEN
# ======================================

def find_best_pt_files(root_dir: Path):
    """Findet rekursiv alle 'best.pt'-Dateien."""
    return list(root_dir.rglob("best.pt"))


def get_model_name(best_pt_path: Path):
    """Extrahiert den Modellnamen aus dem Pfad."""
    return best_pt_path.parent.parent.name


def measure_gflops_thop(model_path: Path):
    """Berechnet GFLOPs mit THOP."""
    try:
        model = YOLO(model_path).model.to(DEVICE)
        model.eval()
        dummy_input = torch.randn(1, 3, IMGSZ, IMGSZ).to(DEVICE)
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        gflops = macs / 1e9
        return gflops
    except Exception as e:
        print(f"[THOP] Fehler bei {model_path}: {e}")
        return None


def measure_gflops_profiler(model_path: Path):
    """Schätzt GFLOPs basierend auf torch.profiler-Auswertung."""
    try:
        model = YOLO(model_path).model.to(DEVICE)
        model.eval()
        dummy_input = torch.randn(1, 3, IMGSZ, IMGSZ).to(DEVICE)

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            if torch.cuda.is_available()
            else [torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            with_flops=True
        ) as prof:
            with torch.no_grad():
                model(dummy_input)

        # Aggregiere alle FLOPs aus dem Profil
        flops_total = sum([e.flops for e in prof.key_averages() if hasattr(e, "flops")])
        gflops = flops_total / 1e9 if flops_total > 0 else None
        return gflops
    except Exception as e:
        print(f"[Profiler] Fehler bei {model_path}: {e}")
        return None


def main():
    models = find_best_pt_files(ROOT_DIR)
    if not models:
        print("Keine 'best.pt'-Dateien gefunden.")
        return

    results = []
    for model_path in models:
        model_name = get_model_name(model_path)
        print(f"\n=== Messen der GFLOPs für: {model_name} ===")

        flops_thop = measure_gflops_thop(model_path)
        flops_profiler = measure_gflops_profiler(model_path)

        results.append({
            "Model": model_name,
            "Path": str(model_path),
            "GFLOPs (torch.profiler)": flops_profiler
        })

    df = pd.DataFrame(results)
    df.to_excel(OUTPUT_XLSX, index=False)
    print(f"\nErgebnisse gespeichert unter: {OUTPUT_XLSX.resolve()}")
    print(df)


if __name__ == "__main__":
    main()
