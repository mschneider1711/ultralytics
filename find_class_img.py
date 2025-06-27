import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# === Pfad zum Dataset (z. B. run2 oder run3)
dataset_path = Path("/Users/marcschneider/Documents/PlantDoc.v4i.yolov8")
splits = ["train", "valid", "test"]
target_class = 16  # Soybean leaf

# === Bild finden
for split in splits:
    labels_path = dataset_path / split / "labels"
    images_path = dataset_path / split / "images"
    
    for label_file in labels_path.glob("*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()
            if any(line.strip().startswith(f"{target_class} ") for line in lines):
                image_file = images_path / (label_file.stem + ".jpg")
                if not image_file.exists():
                    image_file = images_path / (label_file.stem + ".png")
                if image_file.exists():
                    print(f"✅ Gefunden in: {image_file}")
                    img = Image.open(image_file)
                    plt.imshow(img)
                    plt.title(f"{split.upper()} – {image_file.name}")
                    plt.axis('off')
                    plt.show()
                    break
    else:
        continue  # gehe zum nächsten Split
    break  # Bild gefunden → abbrechen
else:
    print("❌ Kein Bild mit Klasse 16 gefunden.")
