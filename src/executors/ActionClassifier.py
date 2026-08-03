import os
import sys

# Proje kök dizinini ekle (Novavision sunucu yapısı için)
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.PoseActionClassifier.src.models.PackageModel import PackageModel
from components.PoseActionClassifier.src.utils.response import build_response_action
from components.PoseActionClassifier.src.utils.tracker import CentroidTracker
from components.PoseActionClassifier.src.utils.video_logic import classify_action_model, classify_action_geometry


class ActionClassifier(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        # inputDetections, Buffer paketinden gelen düzleştirilmiş (flattened) tespit listesidir.
        self.input_detections = self.request.get_param("inputDetections")
        
        # Konfigürasyonları oku
        self.method = self.request.get_param("ActionMethod")
        if isinstance(self.method, dict):
            self.method = self.method.get("value", "Geometry-Based")

        self.model_path = self.request.get_param("ActionModelPath")
        if isinstance(self.model_path, dict):
            self.model_path = self.model_path.get("value", "")

        self.velocity_threshold = self.request.get_param("VelocityThreshold")
        if isinstance(self.velocity_threshold, dict):
            self.velocity_threshold = self.velocity_threshold.get("value", 3.0)

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        if not self.input_detections:
            self.detections = []
            return build_response_action(context=self)

        # 1. Gelen tespitleri imgUID'ye göre grupla (Frame bazında ayır)
        frames_grouped = []
        seen_uids = {}
        for det in self.input_detections:
            uid = det.imgUID
            if uid not in seen_uids:
                seen_uids[uid] = len(frames_grouped)
                frames_grouped.append([])
            frames_grouped[seen_uids[uid]].append(det)

        # Buffer en yeni kareden en eski kareye doğru sıralıdır (insert 0 yapıldığı için).
        # Kronolojik analiz için listeyi ters çevirip en eskiden en yeniye doğru sıralayalım.
        frames_grouped.reverse()

        if len(frames_grouped) == 0:
            self.detections = []
            return build_response_action(context=self)

        # 2. Centroid Tracker ile kişi takibini kronolojik olarak çalıştır
        tracker = CentroidTracker()
        for frame in frames_grouped:
            tracker.update(frame)

        # 3. Sadece en yeni (güncel) karenin tespitlerini sınıflandırıp çıktı olarak verelim
        # Böylece ekranda eski kareler mükerrer olarak çizilmez, sadece güncel kare gösterilir.
        newest_frame_detections = frames_grouped[-1]
        final_detections = []

        for det in newest_frame_detections:
            # Bu tespite ait hareket geçmişini (history) tracker'dan bul
            det_history = []
            for obj_id, hist in tracker.history.items():
                if hist and hist[-1] == det: # Nesne referansı eşleşiyorsa
                    det_history = hist
                    break

            # Eğer geçmiş bulunduysa hareket sınıflandırıcısını çalıştır
            if det_history:
                if self.method == "Model-Based":
                    action_class = classify_action_model(
                        det_history,
                        det.boundingBox,
                        self.model_path,
                        self.velocity_threshold
                    )
                else:
                    action_class = classify_action_geometry(
                        det_history,
                        det.boundingBox,
                        self.velocity_threshold
                    )
            else:
                # Geçmiş bulunamazsa varsayılan durum
                action_class = "standing"

            # classLabel alanını güncelle (Örn: "person" -> "person_walking")
            det.classLabel = f"{det.classLabel}_{action_class}"
            final_detections.append(det)

        self.detections = final_detections
        return build_response_action(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
