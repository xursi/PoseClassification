import json
import os
import sys

# Novavision SDK'lerinin çözülmesi için sunucu yolunu ekle
sys.path.append("C:/Users/gslix/.novavision/Server/08EA28/B6DD01/diginova-pytorch")
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.PackageModel import PackageModel as Package


with open("data.json", "w", encoding="utf-8") as file:
    file.write(Package.schema_json(indent=2))
