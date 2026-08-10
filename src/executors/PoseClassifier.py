import os
import sys
import math

# Proje kök dizinini ekle (Novavision sunucu yapısı için)
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.base.capsule import Capsule
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


# Sözlüklerden derinlemesine güvenli veri çekmek için yardımcı fonksiyon (NoneType çökmelerini önler)
def safe_get(d, *keys, default=None):
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key)
        else:
            return default
    return d if d is not None else default


# Açı hesaplama fonksiyonları
def calculate_angle(p1, p2, p3):
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


def calculate_trunk_tilt(l_shoulder, r_shoulder, l_hip, r_hip):
    """
    Gövdenin (sırtın) dikey eksenle yaptığı açıyı hesaplar.
    """
    if not l_shoulder and not r_shoulder:
        return 0.0
    if not l_hip and not r_hip:
        return 0.0

    # Omuz ve kalça merkezlerini bul
    if l_shoulder and r_shoulder:
        sh_x = (l_shoulder[0] + r_shoulder[0]) / 2.0
        sh_y = (l_shoulder[1] + r_shoulder[1]) / 2.0
    else:
        sh = l_shoulder if l_shoulder else r_shoulder
        sh_x, sh_y = sh[0], sh[1]

    if l_hip and r_hip:
        hp_x = (l_hip[0] + r_hip[0]) / 2.0
        hp_y = (l_hip[1] + r_hip[1]) / 2.0
    else:
        hp = l_hip if l_hip else r_hip
        hp_x, hp_y = hp[0], hp[1]

    dx = sh_x - hp_x
    dy = sh_y - hp_y

    magnitude = math.sqrt(dx**2 + dy**2)
    if magnitude == 0:
        return 0.0

    # Resim koordinatlarında dikey eksen (0, -1)'dir (yukarı doğru).
    # Bu vektör ile omuz-kalça vektörünün açısı:
    cos_angle = -dy / magnitude
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def calculate_neck_tilt(nose, l_shoulder, r_shoulder):
    """
    Boynun dikey eksenle yaptığı eğilme açısını hesaplar.
    """
    if not nose or (not l_shoulder and not r_shoulder):
        return 0.0

    if l_shoulder and r_shoulder:
        sh_x = (l_shoulder[0] + r_shoulder[0]) / 2.0
        sh_y = (l_shoulder[1] + r_shoulder[1]) / 2.0
    else:
        sh = l_shoulder if l_shoulder else r_shoulder
        sh_x, sh_y = sh[0], sh[1]

    dx = nose[0] - sh_x
    dy = nose[1] - sh_y

    magnitude = math.sqrt(dx**2 + dy**2)
    if magnitude == 0:
        return 0.0

    cos_angle = -dy / magnitude
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def is_hand_raised(wrist, shoulder, head=None):
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

    features["left_knee_angle"] = calculate_angle(l_hip, l_knee, l_ankle)
    features["right_knee_angle"] = calculate_angle(r_hip, r_knee, r_ankle)
    
    ref_shoulder = l_shoulder if l_shoulder else r_shoulder
    
    features["left_hand_raised"] = is_hand_raised(l_wrist, ref_shoulder, nose)
    features["right_hand_raised"] = is_hand_raised(r_wrist, ref_shoulder, nose)
    
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


# 1. Mod: Standart Duruş Sınıflandırma
def classify_pose_geometry(keypoints, bbox, knee_threshold=130.0):
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

    if l_knee is not None and r_knee is not None:
        if l_knee < 120.0 and r_knee < 120.0:
            return "sitting"

    return "standing"


# 2. Mod: Ergonomik Davranış / İSG Analizi (RULA/REBA Uyumlu)
def classify_pose_ergonomics(keypoints, bbox, back_tilt_threshold=20.0):
    kpts = []
    if keypoints:
        for kp in keypoints:
            if kp:
                if isinstance(kp, dict):
                    kpts.append([kp.get('cx', 0.0), kp.get('cy', 0.0)])
                else:
                    kpts.append([getattr(kp, 'cx', 0.0), getattr(kp, 'cy', 0.0)])
            else:
                kpts.append([0.0, 0.0])
    else:
        kpts = [[0.0, 0.0]] * 17

    # Noktaları çek
    nose = kpts[NOSE] if kpts[NOSE] != [0.0, 0.0] else None
    l_shoulder = kpts[L_SHOULDER] if kpts[L_SHOULDER] != [0.0, 0.0] else None
    r_shoulder = kpts[R_SHOULDER] if kpts[R_SHOULDER] != [0.0, 0.0] else None
    l_wrist = kpts[L_WRIST] if kpts[L_WRIST] != [0.0, 0.0] else None
    r_wrist = kpts[R_WRIST] if kpts[R_WRIST] != [0.0, 0.0] else None
    l_elbow = kpts[L_ELBOW] if kpts[L_ELBOW] != [0.0, 0.0] else None
    r_elbow = kpts[R_ELBOW] if kpts[R_ELBOW] != [0.0, 0.0] else None
    l_hip = kpts[L_HIP] if kpts[L_HIP] != [0.0, 0.0] else None
    r_hip = kpts[R_HIP] if kpts[R_HIP] != [0.0, 0.0] else None

    # Açıları ve kol yükselme durumlarını hesapla
    back_tilt = calculate_trunk_tilt(l_shoulder, r_shoulder, l_hip, r_hip)
    neck_tilt = calculate_neck_tilt(nose, l_shoulder, r_shoulder)
    
    l_arm_raised = False
    if l_shoulder:
        if l_wrist and l_wrist[1] < l_shoulder[1]:
            l_arm_raised = True
        if l_elbow and l_elbow[1] < l_shoulder[1]:
            l_arm_raised = True
            
    r_arm_raised = False
    if r_shoulder:
        if r_wrist and r_wrist[1] < r_shoulder[1]:
            r_arm_raised = True
        if r_elbow and r_elbow[1] < r_shoulder[1]:
            r_arm_raised = True
            
    arms_raised = l_arm_raised or r_arm_raised

    # RULA / REBA Ergonomik Sınır Değerleri
    # Yüksek Risk (Unsafe): Gövde eğimi > 45° VEYA Boyun eğimi > 35° VEYA Eller omuz hizası üstündeyse (Overhead Work)
    if back_tilt > 45.0 or neck_tilt > 35.0 or arms_raised:
        reasons = []
        if back_tilt > 45.0:
            reasons.append("Stooping")
        if neck_tilt > 35.0:
            reasons.append("Neck Bend")
        if arms_raised:
            reasons.append("Overhead Work")
        return f"Unsafe ({', '.join(reasons)})"
        
    # Context (Warning): Gövde eğimi > back_tilt_threshold VEYA Boyun eğimi > 20°
    elif back_tilt > back_tilt_threshold or neck_tilt > 20.0:
        reasons = []
        if back_tilt > back_tilt_threshold:
            reasons.append("Mild Stooping")
        if neck_tilt > 20.0:
            reasons.append("Mild Neck Bend")
        return f"Warning ({', '.join(reasons)})"
        
    else:
        return "Safe"


# Veritabanında kayıtlı olan orijinal "PoseClassifier" sınıf ismine sadık kalındı
class PoseClassifier(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.detections = self.request.get_param("inputDetections")
        
        # Konfigürasyonları oku
        self.pose_geom_mode = self.request.get_param("poseGeometryMode")
        self.mode = "Standard Pose Classification"
        
        if isinstance(self.pose_geom_mode, dict):
            self.mode = self.pose_geom_mode.get("name", "Standard Pose Classification")
            
        self.knee_threshold = 130.0
        self.back_tilt_threshold = 20.0
        
        # Mod parametrelerini oku
        if self.mode == "Standard Pose Classification":
            val = safe_get(self.pose_geom_mode, "value")
            if isinstance(val, dict):
                self.knee_threshold = safe_get(val, "kneeAngleThreshold", "value", default=130.0)
        elif self.mode == "Ergonomic Safety Assessment":
            val = safe_get(self.pose_geom_mode, "value")
            if isinstance(val, dict):
                self.back_tilt_threshold = safe_get(val, "backTiltThreshold", "value", default=20.0)

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

                # Seçilen geometri moduna göre çalıştır
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
