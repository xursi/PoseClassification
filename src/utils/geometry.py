import numpy as np
import math

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
    Noktalar {'cx': x, 'cy': y} veya [x, y] formatında olmalıdır.
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
    Görüntü koordinatlarında y ekseni aşağı doğru arttığından, 'küçük y' yukarıyı ifade eder.
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
