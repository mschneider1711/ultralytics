import torch
from pathlib import Path
from ultralytics import YOLO

ROOT_DIR  = Path("/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/backbone_experiments")
DATA_PATH = Path("/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/PlantDoc-3/splits/split0/data.yaml")
FORMATS   = ["-", "torchscript", "engine"]
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
IMGSZ     = 640
VERBOSE   = True


def find_best_pt_files(root_dir: Path):
    return list(root_dir.rglob("best.pt"))


def get_model_name(best_pt_path: Path):
    return best_pt_path.parent.parent.name


def run_single_benchmark(model: YOLO, model_name: str, fmt: str, half: bool):
    print(f"\n--- Benchmark: {fmt.upper()} (half={half}) ---")
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        model.benchmark(
            data=DATA_PATH,
            format=fmt,
            half=half,
            device=DEVICE,
            imgsz=IMGSZ,
            verbose=VERBOSE
        )

        if torch.cuda.is_available():
            max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"Max GPU memory: {max_mem:.1f} MB")

    except Exception as e:
        print(f"Fehler beim Benchmark ({fmt}, half={half}) für {model_name}: {e}")


def run_benchmarks_for_precision(best_models, half: bool):
    mode = "FP16" if half else "FP32"
    print(f"\n{'=' * 70}")
    print(f"Starte {mode}-Benchmarks für alle Modelle")
    print(f"{'=' * 70}")

    for model_path in best_models:
        model_name = get_model_name(model_path)
        print(f"\n{'=' * 70}")
        print(f"Benchmark für Modell: {model_name} ({mode})")
        print(f"Pfad: {model_path}")
        print(f"{'=' * 70}")

        model = YOLO(model_path)
        for fmt in FORMATS:
            run_single_benchmark(model, model_name, fmt, half)


def main():
    if not torch.cuda.is_available():
        print("Keine GPU erkannt – Benchmarks laufen auf CPU.")

    best_models = find_best_pt_files(ROOT_DIR)
    if not best_models:
        print("Keine 'best.pt'-Dateien gefunden.")
        return

    run_benchmarks_for_precision(best_models, half=False)

    print("\n\n" + "#" * 80)
    print("###########################  FP16 BENCHMARKS  ###########################")
    print("#" * 80 + "\n\n")

    run_benchmarks_for_precision(best_models, half=True)


if __name__ == "__main__":
    main()
