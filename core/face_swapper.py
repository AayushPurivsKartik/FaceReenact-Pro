# core/face_swapper.py
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image
import onnxruntime as ort
import os
from PIL import Image, ImageEnhance
import threading

class FaceSwapper:
    def __init__(self):
        self.model_path = "models/inswapper_128.onnx"
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0 if ort.get_device() == 'GPU' else -1, det_size=(640, 640))
        
        # Load the swap model
        self.swap_model = insightface.model_zoo.get_model(self.model_path, download=False)
        print("Face swap model loaded successfully!")

    def get_faces(self, image):
        """Detect all faces in image"""
        faces = self.app.get(image)
        return sorted(faces, key=lambda x: x.bbox[0])  # Sort left to right

    def swap_faces(self, source_img, target_img, source_face_index=0, target_face_index=0):
        """Swap specific face from source to target"""
        source_faces = self.get_faces(source_img)
        target_faces = self.get_faces(target_img)

        if source_face_index >= len(source_faces):
            raise ValueError(f"Source face index {source_face_index} not found. Only {len(source_faces)} faces detected.")
        if target_face_index >= len(target_faces):
            raise ValueError(f"Target face index {target_face_index} not found. Only {len(target_faces)} faces detected.")

        source_face = source_faces[source_face_index]
        target_face = target_faces[target_face_index]

        result = target_img.copy()
        result = self.swap_model.get(result, target_face, source_face, paste_back=True)
        
        return result

    def process_video(self, source_path, target_path, output_path, callback=None):
        """Process video frame by frame"""
        source_img = cv2.imread(source_path)
        if source_img is None:
            raise FileNotFoundError(f"Source image not found: {source_path}")
            
        cap = cv2.VideoCapture(target_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        
        source_faces = self.get_faces(source_img)
        if len(source_faces) == 0:
            raise ValueError("No face detected in source image!")
            
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                # Auto select first face or match by position
                target_faces = self.get_faces(frame)
                if target_faces:
                    # Use the face closest to the source face position
                    source_face = source_faces[0]
                    target_face = min(target_faces, key=lambda f: abs(f.bbox[0] - source_face.bbox[0]))
                    
                    frame = self.swap_model.get(frame, target_face, source_face, paste_back=True)
            except:
                pass  # Keep original frame if swap fails
                
            out.write(frame)
            
            current_frame += 1
            if callback:
                callback(current_frame, total_frames)
        
        cap.release()
        out.release()

# Global instance
swapper = None
swapper_lock = threading.Lock()

def get_swapper():
    global swapper
    with swapper_lock:
        if swapper is None:
            swapper = FaceSwapper()
        return swapper
