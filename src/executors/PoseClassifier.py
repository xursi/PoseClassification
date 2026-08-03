import os
import sys

# Proje kök dizinini ekle (Novavision sunucu yapısı için)
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PoseActionClassifier.src.models.PackageModel import PackageModel
from components.PoseActionClassifier.src.utils.response import build_response_pose
from components.PoseActionClassifier.src.utils.image_logic import classify_pose_model, classify_pose_geometry


class PoseClassifier(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.detections = self.request.get_param("inputDetections")
        
        # Konfigürasyonları oku
        self.method = self.request.get_param("PoseMethod")
        if isinstance(self.method, dict):
            self.method = self.method.get("value", "Geometry-Based")

        self.model_path = self.request.get_param("PoseModelPath")
        if isinstance(self.model_path, dict):
            self.model_path = self.model_path.get("value", "")

        self.knee_threshold = self.request.get_param("KneeAngleThreshold")
        if isinstance(self.knee_threshold, dict):
            self.knee_threshold = self.knee_threshold.get("value", 130.0)

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        classified_detections = []

        for det in self.detections:
            # Sınıflandırma metodunu seç ve çalıştır
            if self.method == "Model-Based":
                pose_class = classify_pose_model(
                    det.keyPoints, 
                    det.boundingBox, 
                    self.model_path, 
                    self.knee_threshold
                )
            else:
                pose_class = classify_pose_geometry(
                    det.keyPoints, 
                    det.boundingBox, 
                    self.knee_threshold
                )

            # classLabel alanını güncelle (Örn: "person" -> "person_standing")
            det.classLabel = f"{det.classLabel}_{pose_class}"
            classified_detections.append(det)

        self.detections = classified_detections
        return build_response_pose(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
