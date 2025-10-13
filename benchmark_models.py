import torch
from pathlib import Path
from ultralytics import YOLO

# ==============================
# 1) Konfiguration
# ==============================
ROOT_DIR  = Path("/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/backbone_experiments")  # Wurzelordner mit Unterordnern
DATA_PATH = Path("/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/PlantDoc-3/splits/split0/data.yaml")  # Pfad zu data.yaml
FORMAT    = "engine"   # TensorRT
DEVICE    = 0          # GPU-ID oder "cpu"
IMGSZ     = 640        # Input-Größe
VERBOSE   = True        # Detaillierte Benchmark-Ausgabe

# ==============================
# 2) Funktionen
# ==============================
def find_best_pt_files(root_dir: Path):
    """Sucht rekursiv alle 'best.pt'-Dateien."""
    return list(root_dir.rglob("best.pt"))

def get_model_name(best_pt_path: Path):
    """Zwei Ebenen über der Datei."""
    return best_pt_path.parent.parent.name

def run_benchmark(model_path: Path):
    """Führt Benchmark für FP32 und FP16 aus + misst GPU Speicher."""
    model_name = get_model_name(model_path)
    print(f"\n{'=' * 60}")
    print(f"Benchmark für Modell: {model_name}")
    print(f"Pfad: {model_path}")
    print(f"{'=' * 60}")

    model = YOLO(model_path)

    # === FP32 Benchmark ===
    print("\n--- TensorRT FP32 ---")
    torch.cuda.reset_peak_memory_stats()
    model.benchmark(
        data=DATA_PATH,
        format=FORMAT,
        half=False,
        device=DEVICE,
        imgsz=IMGSZ,
        verbose=VERBOSE
    )
    max_mem_fp32 = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Max GPU-Speicher (FP32): {max_mem_fp32:.1f} MB")

    # === FP16 Benchmark ===
    print("\n--- TensorRT FP16 ---")
    torch.cuda.reset_peak_memory_stats()
    model.benchmark(
        data=DATA_PATH,
        format=FORMAT,
        half=True,
        device=DEVICE,
        imgsz=IMGSZ,
        verbose=VERBOSE
    )
    max_mem_fp16 = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Max GPU-Speicher (FP16): {max_mem_fp16:.1f} MB")

# ==============================
# 3) Hauptprogramm
# ==============================
def main():
    if not torch.cuda.is_available():
        print("⚠️ Keine GPU erkannt – TensorRT Benchmark wird stark verlangsamt.")
    best_models = find_best_pt_files(ROOT_DIR)

    if not best_models:
        print("⚠️ Keine 'best.pt'-Dateien gefunden.")
        return

    for best_pt in best_models:
        run_benchmark(best_pt)

if __name__ == "__main__":
    main()
