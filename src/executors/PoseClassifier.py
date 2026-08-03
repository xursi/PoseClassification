import os
import sys

# Proje kök dizinini ekle (Novavision sunucu yapısı için)
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.base.capsule import Capsule  # Capsule sınıfından miras alındı
from sdks.novavision.src.helper.executor import Executor
from capsules.PoseClassification.src.models.PackageModel import PackageModel
from capsules.PoseClassification.src.utils.response import build_response_pose
from capsules.PoseClassification.src.utils.image_logic import classify_pose_geometry


class PoseClassifier(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.detections = self.request.get_param("inputDetections")
        
        # Konfigürasyonları oku
        self.knee_threshold = self.request.get_param("kneeAngleThreshold")
        if isinstance(self.knee_threshold, dict):
            self.knee_threshold = self.knee_threshold.get("value", 130.0)

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        classified_detections = []

        for det in self.detections:
            # Geometrik sınıflandırmayı çalıştır (standing, sitting, climbing)
            pose_class = classify_pose_geometry(
                det.keyPoints, 
                det.boundingBox, 
                self.knee_threshold
            )

            # classLabel alanını güncelle (Örn: "person" -> "person_standing")
            det.classLabel =  "Naber Mudur" #f"{det.classLabel}_{pose_class}"
            classified_detections.append(det)

        self.detections = classified_detections
        return build_response_pose(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
