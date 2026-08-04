import os
import sys
import math
import torch
import torch.nn as nn
import urllib.request
from pathlib import Path

# Proje kök dizinini ekle (Novavision sunucu yapısı için)
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.base.capsule import Capsule  # Capsule sınıfından miras alındı
from sdks.novavision.src.helper.executor import Executor
from sdks.novavision.src.helper.package import PackageHelper
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


# 1. PyTorch MLP model yapısı (Eklem noktalarından duruş pozisyonu tahmin eder)
class PoseMLP(nn.Module):
    def __init__(self, input_dim=34, hidden_dim=64, num_classes=3):
        super(PoseMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 2. Keypoint Normalizasyonu (Shift ve Scale Invariance)
def normalize_keypoints(keypoints, bbox):
    """
    Eklem noktası koordinatlarını gövde yüksekliğine ve kalça merkezine göre normalize eder.
    Girdi: 17 eklem noktası
    Çıktı: 34 boyutlu (17*2) normalize liste.
    """
    coords = []
    for kp in keypoints:
        if kp:
            if isinstance(kp, dict):
                coords.append([kp.get('cx', 0.0), kp.get('cy', 0.0)])
            else:
                coords.append([getattr(kp, 'cx', 0.0), getattr(kp, 'cy', 0.0)])
        else:
            coords.append([0.0, 0.0])
    
    coords = np_coords = []
    # numpy importu yerine standart python matrisi kullanarak ek kütüphane yükünü azaltalım
    # coords listesini numpy yerine düz matematik ile merkezleyip ölçekliyoruz
    
    # 1. Merkez Kaydırma: Kalça orta noktasını merkez yap
    l_hip = coords[L_HIP]
    r_hip = coords[R_HIP]
    
    if isinstance(bbox, dict):
        left = bbox.get('left', 0.0)
        top = bbox.get('top', 0.0)
        width = bbox.get('width', 1.0)
        height = bbox.get('height', 1.0)
    else:
        left = getattr(bbox, 'left', 0.0)
        top = getattr(bbox, 'top', 0.0)
        width = getattr(bbox, 'width', 1.0)
        height = getattr(bbox, 'height', 1.0)

    if l_hip != [0.0, 0.0] and r_hip != [0.0, 0.0]:
        center_x = (l_hip[0] + r_hip[0]) / 2.0
        center_y = (l_hip[1] + r_hip[1]) / 2.0
    else:
        l_shoulder = coords[L_SHOULDER]
        r_shoulder = coords[R_SHOULDER]
        if l_shoulder != [0.0, 0.0] and r_shoulder != [0.0, 0.0]:
            center_x = (l_shoulder[0] + r_shoulder[0]) / 2.0
            center_y = (l_shoulder[1] + r_shoulder[1]) / 2.0
        else:
            center_x = left + width / 2.0
            center_y = top + height / 2.0
            
    # 2. Ölçekleme: Gövde yüksekliğine böl (omuz - kalça arası dikey fark)
    l_shoulder = coords[L_SHOULDER]
    l_hip = coords[L_HIP]
    
    scale = 1.0
    if l_shoulder != [0.0, 0.0] and l_hip != [0.0, 0.0]:
        scale = abs(l_hip[1] - l_shoulder[1])
        
    if scale == 0:
        scale = height if height > 0 else 1.0
        
    # Tüm koordinatları merkezleyip ölçekle
    normalized = []
    for c in coords:
        norm_x = (c[0] - center_x) / scale
        norm_y = (c[1] - center_y) / scale
        normalized.extend([norm_x, norm_y])
        
    return normalized


# 3. Model Ağırlıklarını İndirme Metotları (Yolo tarzı)
def download_weights(name_weight, storage_dir="/storage"):
    """
    Önceden eğitilmiş varsayılan modeli uzak sunucudan /storage klasörüne indirir.
    """
    try:
        storage_path = Path(storage_dir) / name_weight
        if storage_path.exists():
            return str(storage_path)
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Modeli genel bir GitHub reposundan indiriyoruz
        download_url = f"https://raw.githubusercontent.com/NovavisionAI/assets/main/models/{name_weight}"
        urllib.request.urlretrieve(download_url, str(storage_path))
        return str(storage_path)
    except Exception as e:
        print(f"Default model download failed: {e}")
        return None


def load_storage(storageID, storage_dir="/storage"):
    """
    Kullanıcının Novavision arayüzünden yüklediği modeli storageID ile indirir.
    """
    try:
        result = PackageHelper.get_storage_details(storageID)
        data = result["data"]
        url_path = result["data_url"]
        name = data["name"]
        file_path = os.path.join(storage_dir, name)
        
        os.makedirs(storage_dir, exist_ok=True)
        urllib.request.urlretrieve(url_path, file_path)
        return file_path
    except Exception as e:
        print(f"Custom model download failed: {e}")
        return None


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


class PoseClassifier(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))
        self.detections = self.request.get_param("inputDetections")
        
        # Konfigürasyonları oku
        self.pose_method_obj = self.request.get_param("poseMethod")
        self.method = "Geometry-Based"
        
        if isinstance(self.pose_method_obj, dict):
            self.method = self.pose_method_obj.get("name", "Geometry-Based")
            
        # Geometri parametresini oku
        self.knee_threshold = 130.0
        if self.method == "Geometry-Based":
            val = self.pose_method_obj.get("value", {})
            self.knee_threshold = val.get("kneeAngleThreshold", {}).get("value", 130.0)
            
        # Yüklenen modeli al
        self.model = self.bootstrap.get("model")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        bootstrap_data = {}
        try:
            # 1. Hangi executor çalışıyor bul
            executor_cfg = config.get("configs", {}).get("executor", {})
            executor_name = executor_cfg.get("name")
            
            # Eğer PoseClassifier çalışıyorsa
            if executor_name == "PoseClassifier":
                pose_val = executor_cfg.get("value", {}).get("value", {})
                pose_configs = pose_val.get("configs", {})
                
                # PoseMethod seçeneğini al
                pose_method_obj = pose_configs.get("poseMethod", {})
                method_name = pose_method_obj.get("name")
                
                # Eğer Model-Based seçildiyse
                if method_name == "Model-Based":
                    model_selection = pose_method_obj.get("value", {}).get("poseModelSelection", {})
                    source_name = model_selection.get("name")
                    
                    weight_path = None
                    if source_name == "CustomWeight":
                        # Custom model ID'sini al ve Novavision'dan indir
                        storage_id = model_selection.get("value", {}).get("customFieldStorage", {}).get("value", {}).get("storageID", {}).get("value", {}).get("Id", {}).get("value")
                        if storage_id:
                            weight_path = load_storage(storage_id)
                    else:
                        # Pre-trained model adını al ve Github'dan indir
                        model_name = model_selection.get("value", {}).get("defaultModelName", {}).get("value", "pose_mlp_v1.pth")
                        weight_path = download_weights(model_name)
                    
                    if weight_path and os.path.exists(weight_path):
                        # PyTorch modelini yükle
                        model = PoseMLP()
                        model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
                        model.eval()
                        bootstrap_data["model"] = model
                        bootstrap_data["model_path"] = weight_path
        except Exception as e:
            print(f"Bootstrap failed: {e}")
        
        return bootstrap_data

    def run(self):
        classified_detections = []

        if self.detections:
            for det in self.detections:
                is_dict = isinstance(det, dict)
                
                keypoints = det.get("keyPoints") if is_dict else getattr(det, "keyPoints", None)
                bbox = det.get("boundingBox") if is_dict else getattr(det, "boundingBox", None)

                pose_class = "standing"  # Varsayılan sınıf

                # Sınıflandırma metodunu seç ve çalıştır
                if self.method == "Model-Based" and self.model is not None:
                    try:
                        # Keypoint'leri normalize et
                        norm_kpts = normalize_keypoints(keypoints, bbox)
                        input_tensor = torch.tensor([norm_kpts], dtype=torch.float32)
                        
                        # Model çıkarımını yap
                        with torch.no_grad():
                            outputs = self.model(input_tensor)
                            _, predicted = torch.max(outputs, 1)
                            class_idx = predicted.item()
                            pose_class = ["standing", "sitting", "climbing"][class_idx]
                    except Exception as e:
                        print(f"Model inference failed, falling back to geometry: {e}")
                        # Model hatası durumunda kural tabanlı geometriye güvenli fallback
                        pose_class = classify_pose_geometry(keypoints, bbox, self.knee_threshold)
                else:
                    # Kural tabanlı geometriyi çalıştır
                    pose_class = classify_pose_geometry(keypoints, bbox, self.knee_threshold)

                # classPosition alanına yazdır (classLabel değerine dokunmadan)
                if is_dict:
                    det["classPosition"] = pose_class
                else:
                    det.classPosition = pose_class

                classified_detections.append(det)

        self.detections = classified_detections
        return build_response_pose(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()
