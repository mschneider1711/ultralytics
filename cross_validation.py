from pathlib import Path

dataset_path = Path(r"/Users/marcschneider/Documents/PlantDoc.v4i.yolov8")  # replace with 'path/to/dataset' for your custom data
labels = sorted(dataset_path.rglob("*labels/*.txt"))  # all data in 'labels'

import yaml

yaml_file = r"/Users/marcschneider/Documents/PlantDoc.v4i.yolov8/data.yaml"  # your data YAML with data directories and names dictionary
with open(yaml_file, encoding="utf8") as y:
    classes = yaml.safe_load(y)["names"]
cls_idx = sorted(classes.keys())