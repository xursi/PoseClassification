import math

class CentroidTracker:
    def __init__(self, max_distance=150.0):
        self.next_object_id = 0
        self.objects = {}       # id -> last centroid (x, y)
        self.history = {}       # id -> list of detections (from oldest to newest)
        self.max_distance = max_distance

    def _get_centroid(self, bbox):
        cx = bbox.left + (bbox.width / 2.0)
        cy = bbox.top + (bbox.height / 2.0)
        return (cx, cy)

    def register(self, centroid, detection):
        self.objects[self.next_object_id] = centroid
        self.history[self.next_object_id] = [detection]
        self.next_object_id += 1

    def deregister(self, object_id):
        # We don't delete history since we want the full window's history at the end
        if object_id in self.objects:
            del self.objects[object_id]

    def update(self, detections):
        """
        Gelen frame'deki tespitleri mevcut nesnelerle eşleştirir.
        detections: List of Detection objects in a single frame.
        """
        if len(detections) == 0:
            # Match nothing, deregister all active tracking centroids
            active_ids = list(self.objects.keys())
            for object_id in active_ids:
                self.deregister(object_id)
            return

        # Yeni tespitlerin merkez noktalarını hesapla
        input_centroids = []
        for det in detections:
            input_centroids.append(self._get_centroid(det.boundingBox))

        # Eğer şu an takip edilen nesne yoksa hepsini kaydet
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], detections[i])
            return

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # Uzaklık matrisini hesapla
        distances = []
        for o_c in object_centroids:
            row = []
            for i_c in input_centroids:
                dist = math.sqrt((o_c[0] - i_c[0])**2 + (o_c[1] - i_c[1])**2)
                row.append(dist)
            distances.append(row)

        # Basit en yakın komşu eşleştirmesi
        # (Daha gelişmiş Hungarian Algorithm yerine pratik ve hızlı mesafe eşleştirmesi)
        used_rows = set()
        used_cols = set()

        # Eşleştirmeleri yap
        while len(used_rows) < len(object_ids) and len(used_cols) < len(input_centroids):
            # En küçük mesafeyi bul
            min_val = float('inf')
            min_row = -1
            min_col = -1
            for r in range(len(object_ids)):
                if r in used_rows:
                    continue
                for c in range(len(input_centroids)):
                    if c in used_cols:
                        continue
                    if distances[r][c] < min_val:
                        min_val = distances[r][c]
                        min_row = r
                        min_col = c

            if min_row != -1 and min_val < self.max_distance:
                object_id = object_ids[min_row]
                self.objects[object_id] = input_centroids[min_col]
                self.history[object_id].append(detections[min_col])
                used_rows.add(min_row)
                used_cols.add(min_col)
            else:
                break

        # Eşleşmeyen eski nesneleri sil (sadece aktif izlemeden kaldır, geçmişi koru)
        for r in range(len(object_ids)):
            if r not in used_rows:
                self.deregister(object_ids[r])

        # Eşleşmeyen yeni tespitleri yeni nesne olarak kaydet
        for c in range(len(input_centroids)):
            if c not in used_cols:
                self.register(input_centroids[c], detections[c])
