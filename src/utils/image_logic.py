import os
import pickle
import numpy as np
from components.PoseClassification.src.utils.geometry import extract_pose_features

# Eklem indeksleri referansı
L_SHOULDER = 5
R_SHOULDER = 6
L_HIP = 11
R_HIP = 12


def normalize_keypoints(keypoints, bbox):
    """
    Keypoint koordinatlarını model eğitimine/tahminine uygun şekilde 
    konum ve ölçekten bağımsız hale getirmek için normalize eder.
    Girdi: 17 eklem noktası listesi (cx, cy)
    Çıktı: Düzleştirilmiş 34 boyutlu (17*2) normalize koordinat dizisi.
    """
    coords = []
    for kp in keypoints:
        if kp:
            coords.append([kp.cx, kp.cy])
        else:
            coords.append([0.0, 0.0])
    
    coords = np.array(coords)
    
    # 1. Merkez Kaydırma (Shift Invariance):
    # Kalça orta noktasını (eğer yoksa omuz orta noktasını) (0,0) merkezi yapar.
    l_hip = coords[L_HIP]
    r_hip = coords[R_HIP]
    
    if not np.array_equal(l_hip, [0.0, 0.0]) and not np.array_equal(r_hip, [0.0, 0.0]):
        center = (l_hip + r_hip) / 2.0
    else:
        l_shoulder = coords[L_SHOULDER]
        r_shoulder = coords[R_SHOULDER]
        if not np.array_equal(l_shoulder, [0.0, 0.0]) and not np.array_equal(r_shoulder, [0.0, 0.0]):
            center = (l_shoulder + r_shoulder) / 2.0
        else:
            center = np.array([bbox.left + bbox.width/2.0, bbox.top + bbox.height/2.0])
            
    coords_shifted = coords - center
    
    # 2. Ölçekleme (Scale Invariance):
    # Gövde yüksekliğine (omuz-kalça arası dikey mesafe) böler.
    l_shoulder = coords[L_SHOULDER]
    l_hip = coords[L_HIP]
    
    scale = 1.0
    if not np.array_equal(l_shoulder, [0.0, 0.0]) and not np.array_equal(l_hip, [0.0, 0.0]):
        scale = abs(l_hip[1] - l_shoulder[1])
        
    if scale == 0:
        scale = bbox.height if bbox.height > 0 else 1.0
        
    coords_normalized = coords_shifted / scale
    
    # 34 boyutlu vektör olarak düzleştir (flatten)
    return coords_normalized.flatten().tolist()


def classify_pose_geometry(keypoints, bbox, knee_threshold=130.0):
    """
    Matematiksel/geometrik kurallara göre statik poz sınıflandırması yapar.
    Sınıflar: standing (ayakta), sitting (oturuyor), climbing (tırmanıyor)
    """
    kpts = []
    for kp in keypoints:
        if kp:
            kpts.append({'cx': kp.cx, 'cy': kp.cy, 'confidence': kp.confidence})
        else:
            kpts.append(None)

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
