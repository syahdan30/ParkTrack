import cv2
import numpy as np
import pickle
from src.utils import Park_classifier

def demonstration():
    """Aplikasi deteksi tempat parkir dengan perhitungan slot kosong/terisi"""
    
    # Parameter
    rect_width, rect_height = 107, 48
    carp_park_positions_path = "data/source/CarParkPos"
    video_path = "data/source/carPark.mp4"

    # Muat posisi parkir langsung dari file
    try:
        with open(carp_park_positions_path, 'rb') as f:
            parking_positions = pickle.load(f)
    except Exception as e:
        print(f"Error loading parking positions: {e}")
        return

    total_spaces = len(parking_positions)
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocessing frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 1)
        processed_frame = cv2.adaptiveThreshold(blur, 255, 
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY_INV, 25, 16)
        
        # Deteksi status parkir
        denoted_image = frame.copy()
        occupied_spaces = 0
        
        for pos in parking_positions:
            x, y = pos
            crop = processed_frame[y:y+rect_height, x:x+rect_width]
            count = cv2.countNonZero(crop)
            
            # Tentukan status dan warna
            if count > 900:  # Terisi
                occupied_spaces += 1
                color = (0, 0, 255)  # Merah
            else:  # Kosong
                color = (0, 255, 0)  # Hijau
            
            # Gambar kotak - PERBAIKAN DI SINI
            cv2.rectangle(denoted_image, 
                          (x, y),  # Titik kiri atas
                          (x+rect_width, y+rect_height),  # Titik kanan bawah
                          color, 2)
        
        # Tampilkan informasi
        available_spaces = total_spaces - occupied_spaces
        cv2.putText(denoted_image, f"Free: {available_spaces}/{total_spaces}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Parking Space Detection", denoted_image)
        
        # Exit dengan tombol 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    demonstration()