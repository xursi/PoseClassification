import os
import sys
import time
import math

# Proje kök dizinini ekle (Novavision sunucu yapısı için)
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from capsules.PoseClassification.src.models.PackageModel import PackageModel
from capsules.PoseClassification.src.utils.response import build_response_pose

# Ortak geometri fonksiyonlarını geometry modülünden çekiyoruz
from capsules.PoseClassification.src.utils.geometry import (
    classify_pose_geometry,
    classify_pose_ergonomics
)


# Kareler arası insan takibi ve hareket analizi için global takip hafızası (Süreç bazında kalıcı)
TRACKER_HISTORY = {}  # {track_id: {"centroid": (cx, cy), "first_unsafe_time": float or None, "last_seen_time": float, "history": [(t, (x, y))]}}
NEXT_TRACK_ID = 1


def get_centroid(bbox):
    """
    Bounding box'ın merkez noktasını hesaplar.
    """
    if not bbox:
        return (0.0, 0.0)
    if isinstance(bbox, dict):
        left = bbox.get("left", 0.0)
        top = bbox.get("top", 0.0)
        width = bbox.get("width", 0.0)
        height = bbox.get("height", 0.0)
    else:
        left = getattr(bbox, "left", 0.0)
        top = getattr(bbox, "top", 0.0)
        width = getattr(bbox, "width", 0.0)
        height = getattr(bbox, "height", 0.0)
    
    cx = left + width / 2.0
    cy = top + height / 2.0
    return (cx, cy)


def update_tracks(detections, max_distance=0.15, max_age=1.5):
    """
    Centroid tracking kullanarak tespitleri kareler boyu takip eder ve geçmişi hafızada tutar.
    """
    global TRACKER_HISTORY, NEXT_TRACK_ID
    current_time = time.time()
    
    # 1. Eski takipleri temizle
    stale_ids = [tid for tid, track in TRACKER_HISTORY.items() if current_time - track["last_seen_time"] > max_age]
    for tid in stale_ids:
        del TRACKER_HISTORY[tid]
        
    matched_tracks = {}  # {detection_index: track_id}
    
    # 2. Merkez noktalarını çıkar
    current_centroids = []
    for det in detections:
        is_dict = isinstance(det, dict)
        bbox = det.get("boundingBox") if is_dict else getattr(det, "boundingBox", None)
        current_centroids.append(get_centroid(bbox))
        
    # 3. Mesafe tabanlı eşleştirme yap
    for idx, (cx, cy) in enumerate(current_centroids):
        best_tid = None
        min_dist = float("inf")
        
        for tid, track in TRACKER_HISTORY.items():
            if tid in matched_tracks.values():
                continue
                
            tcx, tcy = track["centroid"]
            dist = math.sqrt((cx - tcx)**2 + (cy - tcy)**2)
            
            if dist < min_dist and dist < max_distance:
                min_dist = dist
                best_tid = tid
                
        if best_tid is not None:
            matched_tracks[idx] = best_tid
            TRACKER_HISTORY[best_tid]["centroid"] = (cx, cy)
            TRACKER_HISTORY[best_tid]["last_seen_time"] = current_time
            TRACKER_HISTORY[best_tid]["history"].append((current_time, (cx, cy)))
            if len(TRACKER_HISTORY[best_tid]["history"]) > 10:
                TRACKER_HISTORY[best_tid]["history"].pop(0)
        else:
            tid = NEXT_TRACK_ID
            NEXT_TRACK_ID += 1
            TRACKER_HISTORY[tid] = {
                "centroid": (cx, cy),
                "first_unsafe_time": None,
                "last_seen_time": current_time,
                "history": [(current_time, (cx, cy))]
            }
            matched_tracks[idx] = tid
            
    return matched_tracks


def calculate_velocity(history):
    """
    Kayıtlı merkez noktaları geçmişini kullanarak hızı hesaplar.
    """
    if len(history) < 2:
        return 0.0
    
    t_old, (cx_old, cy_old) = history[0]
    t_curr, (cx_curr, cy_curr) = history[-1]
    
    dt = t_curr - t_old
    if dt < 0.1:
        return 0.0
        
    dx = cx_curr - cx_old
    dy = cy_curr - cy_old
    
    return math.sqrt(dx**2 + dy**2) / dt


def classify_standard_action(track, keypoints, bbox, knee_threshold=130.0, velocity_threshold=0.15):
    """
    Hız ve duruş kurallarına göre dinamik aksiyonu sınıflandırır.
    """
    history = track.get("history", [])
    v = calculate_velocity(history)
    
    vy = 0.0
    if len(history) >= 2:
        t_old, (_, cy_old) = history[0]
        t_curr, (_, cy_curr) = history[-1]
        dt = t_curr - t_old
        if dt >= 0.1:
            vy = (cy_curr - cy_old) / dt
            
    # Düşme tespiti
    if vy > velocity_threshold * 2.0:
        return f"falling (vy: {vy:.2f})"
        
    # Koşma / Yürüme tespiti
    if v > velocity_threshold * 2.0:
        return f"running (v: {v:.2f})"
    elif v > velocity_threshold:
        return f"walking (v: {v:.2f})"
    else:
        # Statik geometriye geri düş
        static_pose = classify_pose_geometry(keypoints, bbox, knee_threshold)
        return f"{static_pose} (static)"


class ActionClassifier(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.detections = self.request.get_param("inputDetections")
        
        # Arayüzden seçilen aksiyon modunu al
        self.mode = "Standard Action Classification"
        action_geom_mode = self.request.get_param("actionGeometryMode")
        if isinstance(action_geom_mode, dict):
            self.mode = action_geom_mode.get("name", "Standard Action Classification")
            
        # Varsayılan değerler
        self.knee_threshold = 130.0
        self.back_tilt_threshold = 20.0
        self.duration_threshold = 3.0
        self.velocity_threshold = 0.15
        
        # Parametreleri oku
        duration = self.request.get_param("durationThreshold")
        if duration is not None:
            self.duration_threshold = duration
            
        back = self.request.get_param("backTiltThreshold")
        if back is not None:
            self.back_tilt_threshold = back

        velocity = self.request.get_param("velocityThreshold")
        if velocity is not None:
            self.velocity_threshold = velocity

        knee = self.request.get_param("kneeAngleThreshold")
        if knee is not None:
            self.knee_threshold = knee

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        classified_detections = []
        current_time = time.time()

        if self.detections:
            # Nesne takipçisini çalıştır
            matched_tracks = update_tracks(self.detections, max_distance=0.15, max_age=1.5)

            for idx, det in enumerate(self.detections):
                is_dict = isinstance(det, dict)
                
                keypoints = det.get("keyPoints") if is_dict else getattr(det, "keyPoints", None)
                bbox = det.get("boundingBox") if is_dict else getattr(det, "boundingBox", None)

                track_id = matched_tracks[idx]
                track = TRACKER_HISTORY[track_id]

                # 1. Mod: Dinamik Aksiyon Sınıflandırma
                if self.mode == "Standard Action Classification":
                    pose_class = classify_standard_action(
                        track,
                        keypoints,
                        bbox,
                        self.knee_threshold,
                        self.velocity_threshold
                    )
                
                # 2. Mod: Dinamik İSG Süre Analizi
                else:
                    instant_risk = classify_pose_ergonomics(
                        keypoints, 
                        bbox, 
                        self.back_tilt_threshold
                    )
                    is_unsafe = instant_risk.startswith("Unsafe")
                    
                    if is_unsafe:
                        reasons = instant_risk.replace("Unsafe (", "").replace(")", "")
                        if track["first_unsafe_time"] is None:
                            track["first_unsafe_time"] = current_time
                            elapsed = 0.0
                        else:
                            elapsed = current_time - track["first_unsafe_time"]
                            
                        if elapsed >= self.duration_threshold:
                            pose_class = f"Unsafe Sustained ({reasons})"
                        else:
                            pose_class = f"Warning ({reasons} - Bending {elapsed:.1f}s / {self.duration_threshold}s)"
                    else:
                        track["first_unsafe_time"] = None
                        pose_class = instant_risk

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
