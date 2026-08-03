import os
import pickle
import math
import numpy as np
from components.PoseActionClassifier.src.utils.image_logic import normalize_keypoints

# Eklem indeksleri
L_HIP = 11
R_HIP = 12


def classify_action_geometry(history, current_bbox, velocity_threshold=3.0):
    """
    Kişinin son N karelik hareket geçmişi (history) üzerinden aksiyon sınıflandırması yapar.
    Sınıflar: standing/idle (duruyor), walking (yürüyor), running (koşuyor), jumping (zıplıyor)
    
    history: List of Detection objects for the same person (oldest to newest).
    """
    if len(history) < 2:
        return "standing"

    # 1. Merkez noktaların (kalça veya bbox merkezleri) zaman içindeki yörüngesini çıkar
    centroids = []
    hip_ys = []
    
    for det in history:
        bbox = det.boundingBox
        cx = bbox.left + (bbox.width / 2.0)
        cy = bbox.top + (bbox.height / 2.0)
        centroids.append((cx, cy))
        
        # Kalça yüksekliğini izle (zıplama tespiti için)
        kpts = det.keyPoints
        l_hip = kpts[L_HIP] if len(kpts) > L_HIP else None
        r_hip = kpts[R_HIP] if len(kpts) > R_HIP else None
        
        if l_hip and r_hip:
            hip_ys.append((l_hip.cy + r_hip.cy) / 2.0)
        else:
            hip_ys.append(cy)

    # 2. Hız ve Değişimleri Hesapla
    num_frames = len(history) - 1
    first_c = centroids[0]
    last_c = centroids[-1]
    
    # Piksel bazında yer değiştirme
    dx = last_c[0] - first_c[0]
    dy = last_c[1] - first_c[1]
    
    # Ortalama kare başı hız (piksel/kare)
    v_x = dx / num_frames
    v_y = dy / num_frames
    speed = math.sqrt(v_x**2 + v_y**2)
    
    # 3. Zıplama (Jumping) Analizi:
    # Zıplayan bir kişinin kalça y-koordinatı (dikey) kısa süre içinde tepe yapar.
    # Kalça yörüngesindeki maksimum ve minimum y-değerleri arasındaki farkı inceleyelim.
    # y ekseni ters olduğu için min_y en tepe noktayı, max_y en alt noktayı gösterir.
    max_hip_y = max(hip_ys)
    min_hip_y = min(hip_ys)
    vertical_displacement = max_hip_y - min_hip_y
    
    # Bounding box yüksekliğine göre oranla (kamera uzaklığından bağımsız hale getirmek için)
    bbox_height = current_bbox.height if current_bbox.height > 0 else 1.0
    vert_ratio = vertical_displacement / bbox_height
    
    # Eğer dikey salınım boyun %12'sinden fazlaysa ve dikey hareket yatay hareketten baskınsa
    # (Koşarken de dikey hareket olur ama yatay hareket çok daha fazladır)
    horizontal_displacement = abs(dx)
    is_jumping = vert_ratio > 0.12 and vertical_displacement > horizontal_displacement

    if is_jumping:
        return "jumping"

    # 4. Yürüme/Koşma Analizi (Hız eşiklerine göre):
    # Ölçekten bağımsız olması için hızı vücut boyuna oranlayalım
    normalized_speed = speed / bbox_height # Vücut boyu / kare cinsinden hız
    
    # Eşik değerler (Vücut boyu oranlı)
    # 0.005: Durma/Yürüme sınırı
    # 0.025: Yürüme/Koşma sınırı
    if normalized_speed < 0.004:
        return "standing"
    elif normalized_speed < 0.022:
        return "walking"
    else:
        return "running"


def classify_action_model(history, current_bbox, model_path, velocity_threshold=3.0):
    """
    Eğitilmiş bir modeli (.pkl veya .pt) yükleyerek sıralı keypoint tahmini yapar.
    Hata durumunda geometrik kurallara fallback yapar.
    """
    if model_path and os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Son N karelik keypoint serisini normalize edip tek bir öznitelik vektörü yapalım
            # Örn: Her kare için 34 öznitelik * 10 kare = 340 girdili bir vektör
            sequence_features = []
            for det in history[-10:]: # Son 10 kareyi alalım
                feat = normalize_keypoints(det.keyPoints, det.boundingBox)
                sequence_features.extend(feat)
                
            # Eğer buffer uzunluğu 10'dan azsa sıfırlarla doldur (padding)
            while len(sequence_features) < 340:
                sequence_features.extend([0.0] * 34)
                
            # Tahmin çalıştır
            predicted_action = model.predict([sequence_features])[0]
            
            if predicted_action in ["standing", "walking", "running", "jumping"]:
                return predicted_action
        except Exception as e:
            pass

    # Model yoksa kural tabanlıya geri dön
    return classify_action_geometry(history, current_bbox, velocity_threshold)
