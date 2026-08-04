import os
import sys
import math

# Proje kök dizinini ekle (Novavision sunucu yapısı için)
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.base.capsule import Capsule  # Capsule sınıfından miras alındı
from sdks.novavision.src.helper.executor import Executor
from capsules.PoseClassification.src.models.PackageModel import PackageModel
from capsules.PoseClassification.src.utils.response import build_response_pose


# COCO Anahtar Noktaları (Keypoint Indexes)
NOSE = 0
L_SHOULDER = 5
R_SHOULDER = 6
L_ELBOW = 7
R_ELBOW = 8
L_WRIST = 9
R_WRIST = 10
L_HIP = 11
R_HIP = 12
L_KNEE = 13
R_KNEE = 14
L_ANKLE = 15
R_ANKLE = 16


def calculate_angle(p1, p2, p3):
    """
    Üç nokta arasındaki açıyı hesaplar. p2 köşedir (vertex).
    """
    if not p1 or not p2 or not p3:
        return None

    def get_coords(p):
        if isinstance(p, dict):
            return [p['cx'], p['cy']]
        return [p[0], p[1]]

    a = get_coords(p1)
    b = get_coords(p2)
    c = get_coords(p3)

    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]

    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    if magnitude_ba == 0 or magnitude_bc == 0:
        return None

    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
    cosine_angle = max(-1.0, min(1.0, cosine_angle))
    return round(math.degrees(math.acos(cosine_angle)), 1)


def is_hand_raised(wrist, shoulder, head=None):
    """
    Elin omuz çizgisinden veya baş seviyesinden yukarıda olup olmadığını kontrol eder.
    """
    if not wrist or not shoulder:
        return False

    def get_y(p):
        if isinstance(p, dict):
            return p['cy']
        return p[1]

    wrist_y = get_y(wrist)
    shoulder_y = get_y(shoulder)

    raised_above_shoulder = wrist_y < shoulder_y

    if head:
        head_y = get_y(head)
        raised_above_head = wrist_y < head_y
        return raised_above_shoulder or raised_above_head

    return raised_above_shoulder


def extract_pose_features(keypoints, bbox):
    """
    17 keypoint'ten omuz, dirsek, diz açılarını ve el pozisyonlarını çıkartır.
    """
    features = {}
    
    nose = keypoints[NOSE]
    l_shoulder = keypoints[L_SHOULDER]
    r_shoulder = keypoints[R_SHOULDER]
    l_wrist = keypoints[L_WRIST]
    r_wrist = keypoints[R_WRIST]
    l_hip = keypoints[L_HIP]
    r_hip = keypoints[R_HIP]
    l_knee = keypoints[L_KNEE]
    r_knee = keypoints[R_KNEE]
    l_ankle = keypoints[L_ANKLE]
    r_ankle = keypoints[R_ANKLE]

    # Eklem Açıları
    features["left_knee_angle"] = calculate_angle(l_hip, l_knee, l_ankle)
    features["right_knee_angle"] = calculate_angle(r_hip, r_knee, r_ankle)
    
    # Omuz ve El/Baş Referansı
    ref_shoulder = l_shoulder if l_shoulder else r_shoulder
    
    features["left_hand_raised"] = is_hand_raised(l_wrist, ref_shoulder, nose)
    features["right_hand_raised"] = is_hand_raised(r_wrist, ref_shoulder, nose)
    
    # Bacak Sıkışma Oranı (Leg-to-Torso Ratio)
    active_hip = l_hip if l_hip else r_hip
    active_ankle = l_ankle if l_ankle else r_ankle
    
    ratio = None
    if ref_shoulder and active_hip and active_ankle:
        def get_y(p):
            return p['cy'] if isinstance(p, dict) else p[1]
        
        torso_height = abs(get_y(active_hip) - get_y(ref_shoulder))
        leg_height = abs(get_y(active_ankle) - get_y(active_hip))
        if torso_height > 0:
            ratio = leg_height / torso_height
            
    features["leg_torso_ratio"] = ratio
    return features


def classify_pose_geometry(keypoints, bbox, knee_threshold=130.0):
    """
    Matematiksel/geometrik kurallara göre statik poz sınıflandırması yapar.
    """
    kpts = []
    if keypoints:
        for kp in keypoints:
            if kp:
                if isinstance(kp, dict):
                    kpts.append({
                        'cx': kp.get('cx', 0.0), 
                        'cy': kp.get('cy', 0.0), 
                        'confidence': kp.get('confidence', 1.0)
                    })
                else:
                    kpts.append({
                        'cx': getattr(kp, 'cx', 0.0), 
                        'cy': getattr(kp, 'cy', 0.0), 
                        'confidence': getattr(kp, 'confidence', 1.0)
                    })
            else:
                kpts.append(None)
    else:
        kpts = [None] * 17

    features = extract_pose_features(kpts, bbox)

    # 1. Kural: Tırmanma (climbing) tespiti
    l_hand = features.get("left_hand_raised", False)
    r_hand = features.get("right_hand_raised", False)
    hands_raised = l_hand or r_hand

    l_knee = features.get("left_knee_angle")
    r_knee = features.get("right_knee_angle")
    knee_bent = False
    if l_knee is not None and l_knee < knee_threshold:
        knee_bent = True
    elif r_knee is not None and r_knee < knee_threshold:
        knee_bent = True

    ratio = features.get("leg_torso_ratio")
    vertical_compression = ratio is not None and ratio < 1.05

    if hands_raised and (knee_bent or vertical_compression):
        return "climbing"

    # 2. Kural: Oturma (sitting) tespiti
    if l_knee is not None and r_knee is not None:
        if l_knee < 120.0 and r_knee < 120.0:
            return "sitting"

    # 3. Varsayılan Durum: Ayakta (standing)
    return "standing"


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

        if self.detections:
            for det in self.detections:
                # Sözlük veya nesne olma durumunu kontrol et
                is_dict = isinstance(det, dict)
                
                # keyPoints ve boundingBox al
                keypoints = det.get("keyPoints") if is_dict else getattr(det, "keyPoints", None)
                bbox = det.get("boundingBox") if is_dict else getattr(det, "boundingBox", None)

                # Geometrik sınıflandırmayı çalıştır (standing, sitting, climbing)
                pose_class = classify_pose_geometry(
                    keypoints, 
                    bbox, 
                    self.knee_threshold
                )

                # classLabel alanını güncelle
                if is_dict:
                    det["classLabel"] = f"{det.get('classLabel', 'person')}_{pose_class}"
                else:
                    det.classLabel = f"{det.classLabel}_{pose_class}"

                classified_detections.append(det)

        self.detections = classified_detections
        return build_response_pose(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
