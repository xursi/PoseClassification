import os
import sys

# Proje kök dizinini ekle (Novavision sunucu yapısı için)
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from capsules.PoseClassification.src.models.PackageModel import PackageModel
from capsules.PoseClassification.src.utils.response import build_response_pose

# Ortak geometri ve sınıflandırma fonksiyonlarını geometry modülünden çekiyoruz
from capsules.PoseClassification.src.utils.geometry import (
    classify_pose_geometry,
    classify_pose_ergonomics
)


class PoseClassifier(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.detections = self.request.get_param("inputDetections")
        
        # Arayüzden seçilen geometri modunu al
        self.mode = "Standard Pose Classification"
        pose_geom_mode = self.request.get_param("poseGeometryMode")
        if isinstance(pose_geom_mode, dict):
            self.mode = pose_geom_mode.get("name", "Standard Pose Classification")
            
        # Parametrelerin varsayılan değerleri
        self.knee_threshold = 130.0
        self.back_tilt_threshold = 20.0
        
        # Mod parametrelerini al
        knee = self.request.get_param("kneeAngleThreshold")
        if knee is not None:
            self.knee_threshold = knee
            
        back = self.request.get_param("backTiltThreshold")
        if back is not None:
            self.back_tilt_threshold = back

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        classified_detections = []

        if self.detections:
            for det in self.detections:
                is_dict = isinstance(det, dict)
                
                keypoints = det.get("keyPoints") if is_dict else getattr(det, "keyPoints", None)
                bbox = det.get("boundingBox") if is_dict else getattr(det, "boundingBox", None)

                # Seçilen moda göre sınıflandırmayı çalıştır
                if self.mode == "Ergonomic Safety Assessment":
                    pose_class = classify_pose_ergonomics(
                        keypoints, 
                        bbox, 
                        self.back_tilt_threshold
                    )
                else:
                    pose_class = classify_pose_geometry(
                        keypoints, 
                        bbox, 
                        self.knee_threshold
                    )

                # classPosition alanına sonucu yazdır
                if is_dict:
                    det["classPosition"] = pose_class
                else:
                    det.classPosition = pose_class

                classified_detections.append(det)

        self.detections = classified_detections
        return build_response_pose(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
