import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd
from ultralytics import YOLO
import time
import pickle

class FaceAttendanceSystem:
    def __init__(self):
        try:
            print("Trying to use YOLOv8 face detection model...")
            if not os.path.exists('yolov8n-face.pt'):
                print("YOLOv8 face detection model not found. Using standard YOLOv8 model.")
                self.model = YOLO('yolov8n.pt')
                self.model.classes = [0]
            else:
                self.model = YOLO('yolov8n-face.pt')
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            print("Using standard YOLOv8 model.")
            self.model = YOLO('yolov8n.pt')
            self.model.classes = [0]
        
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_encoding_to_name = {}
        
        self.last_process_time = 0
        self.process_interval = 0.1
        self.last_detection_time = 0
        self.detection_cooldown = 0.5
        
        if not os.path.exists('attendance'):
            os.makedirs('attendance')
            
        if not os.path.exists('cache'):
            os.makedirs('cache')
            
    def load_known_faces(self, faces_dir='known_faces'):
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)
            print(f"Created directory: {faces_dir}")
            print("Please add known face images to this directory")
            return
            
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_encoding_to_name = {}
        
        print(f"\nStarting to load faces from: {faces_dir}")
        
        cache_file = os.path.join('cache', 'face_encodings.pkl')
        cache_valid = False
        
        if os.path.exists(cache_file):
            try:
                cache_time = os.path.getmtime(cache_file)
                cache_valid = True
                
                for root, _, files in os.walk(faces_dir):
                    for file in files:
                        if file.endswith((".jpg", ".jpeg", ".png")):
                            file_path = os.path.join(root, file)
                            if os.path.getmtime(file_path) > cache_time:
                                cache_valid = False
                                print(f"Cache invalidated by newer file: {file_path}")
                                break
                    if not cache_valid:
                        break
                
                if cache_valid:
                    print("Loading face encodings from cache...")
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                        self.known_face_encodings = cache_data['encodings']
                        self.known_face_names = cache_data['names']
                        self.face_encoding_to_name = cache_data['encoding_to_name']
                    
                    print(f"\nCache Summary:")
                    print(f"Total face encodings loaded: {len(self.known_face_encodings)}")
                    print(f"Number of unique people: {len(set(self.known_face_names))}")
                    print(f"People loaded: {', '.join(set(self.known_face_names))}")
                    return
            except Exception as e:
                print(f"Error loading cache: {e}")
                cache_valid = False
        
        for person_dir in os.listdir(faces_dir):
            person_path = os.path.join(faces_dir, person_dir)
            if os.path.isdir(person_path):
                print(f"\nProcessing person directory: {person_dir}")
                
                person_encodings = []
                
                for filename in os.listdir(person_path):
                    if filename.endswith((".jpg", ".jpeg", ".png")):
                        path = os.path.join(person_path, filename)
                        print(f"Processing image: {filename}")
                        
                        image = face_recognition.load_image_file(path)
                        
                        face_locations = face_recognition.face_locations(image, model="hog")
                        if not face_locations:
                            try:
                                face_locations = face_recognition.face_locations(image, model="cnn")
                            except:
                                pass
                                
                        face_encodings = face_recognition.face_encodings(image, face_locations, num_jitters=10)
                        
                        if face_encodings:
                            for face_encoding in face_encodings:
                                person_encodings.append(face_encoding)
                            print(f"✓ Successfully loaded {len(face_encodings)} faces from {filename}")
                        else:
                            print(f"✗ No face found in {filename}")
                
                if person_encodings:
                    for face_encoding in person_encodings:
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_names.append(person_dir)
                        self.face_encoding_to_name[tuple(face_encoding)] = person_dir
        
        if self.known_face_encodings:
            try:
                cache_data = {
                    'encodings': self.known_face_encodings,
                    'names': self.known_face_names,
                    'encoding_to_name': self.face_encoding_to_name
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                print("Saved face encodings to cache")
            except Exception as e:
                print(f"Error saving cache: {e}")
        
        print(f"\nSummary:")
        print(f"Total face encodings loaded: {len(self.known_face_encodings)}")
        print(f"Number of unique people: {len(set(self.known_face_names))}")
        print(f"People loaded: {', '.join(set(self.known_face_names))}")
                    
    def mark_attendance(self, name):
        now = datetime.now()
        date_string = now.strftime('%Y-%m-%d')
        time_string = now.strftime('%H:%M:%S')
        
        attendance_file = f'attendance/attendance_{date_string}.csv'
        if os.path.exists(attendance_file):
            df = pd.read_csv(attendance_file)
            if name in df['Name'].values:
                return False
                
        attendance_data = {
            'Name': name,
            'Time': time_string,
            'Date': date_string
        }
        
        df = pd.DataFrame([attendance_data])
        if os.path.exists(attendance_file):
            df.to_csv(attendance_file, mode='a', header=False, index=False)
        else:
            df.to_csv(attendance_file, index=False)
        return True
            
    def run(self):
        cap = cv2.VideoCapture(0)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        process_every_n_frames = 1
        
        recognized_faces = {}
        recognition_threshold = 2
        
        identity_counter = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            if frame_count % process_every_n_frames != 0:
                cv2.imshow('Face Recognition Attendance', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            current_time = time.time()
            
            if current_time - self.last_process_time < self.process_interval:
                cv2.imshow('Face Recognition Attendance', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
                
            if current_time - self.last_detection_time < self.detection_cooldown:
                cv2.imshow('Face Recognition Attendance', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            self.last_process_time = current_time
            
            results = self.model(frame, conf=0.40)
            
            if len(results[0].boxes) == 0:
                current_faces = set()
                
                cv2.imshow('Face Recognition Attendance', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            current_faces = set()
            current_names = set()
            
            for det in results[0].boxes:
                x1, y1, x2, y2 = map(int, det.xyxy[0].cpu().numpy())
                conf = det.conf[0].cpu().numpy()
                
                if conf < 0.40:
                    continue
                
                face_id = f"{(x1+x2)//2}_{(y1+y2)//2}"
                current_faces.add(face_id)
                
                padding = 20
                y1_padded = max(0, y1 - padding)
                y2_padded = min(frame.shape[0], y2 + padding)
                x1_padded = max(0, x1 - padding)
                x2_padded = min(frame.shape[1], x2 + padding)
                
                face_img = frame[y1_padded:y2_padded, x1_padded:x2_padded]
                
                if face_img.size == 0 or face_img.shape[0] < 20 or face_img.shape[1] < 20:
                    continue
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                
                try:
                    rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    
                    face_encodings = face_recognition.face_encodings(rgb_face, num_jitters=1)
                    
                    if face_encodings:
                        face_encoding = face_encodings[0]
                        
                        if len(self.known_face_encodings) > 0:
                            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)
                            best_match_distance = face_distances[best_match_index]
                            
                            threshold = 0.60
                            
                            if best_match_distance < threshold:
                                name = self.known_face_names[best_match_index]
                                current_names.add(name)
                                
                                raw_confidence = 1 - best_match_distance
                                adjusted_confidence = 80 + (raw_confidence * 15)
                                confidence = min(98, adjusted_confidence)
                                
                                if face_id not in recognized_faces:
                                    recognized_faces[face_id] = {
                                        'name': name,
                                        'count': 1,
                                        'last_seen': current_time,
                                        'confidence': confidence
                                    }
                                    print(f"New face: {name}, count: 1")
                                else:
                                    if recognized_faces[face_id]['name'] == name:
                                        recognized_faces[face_id]['count'] += 1
                                        print(f"Same face: {name}, count increased to {recognized_faces[face_id]['count']}")
                                    else:
                                        if confidence > recognized_faces[face_id]['confidence']:
                                            old_name = recognized_faces[face_id]['name']
                                            recognized_faces[face_id]['name'] = name
                                            recognized_faces[face_id]['count'] = 1
                                            recognized_faces[face_id]['confidence'] = confidence
                                            print(f"Changed face: {old_name} -> {name}, count reset to 1")
                                    
                                    recognized_faces[face_id]['last_seen'] = current_time
                                
                                if name not in identity_counter:
                                    identity_counter[name] = {
                                        'count': 1,
                                        'last_seen': current_time
                                    }
                                else:
                                    if identity_counter[name]['last_seen'] < current_time - 0.1:
                                        identity_counter[name]['count'] += 1
                                    identity_counter[name]['last_seen'] = current_time
                                
                                face_count = recognized_faces[face_id]['count']
                                identity_count = identity_counter[name]['count']
                                current_count = max(face_count, identity_count)
                                print(f"[DEBUG] Face ID: {face_id}, Name: {name}, Count: {current_count}/{recognition_threshold}")
                                
                                if current_count >= recognition_threshold:
                                    self.mark_attendance(name)
                                    
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    label = f"{name} ({confidence:.1f}%)"
                                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                                    self.last_detection_time = current_time
                                else:
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                                    label = f"Confirming: {name} ({confidence:.1f}%)"
                                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                            else:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                                confidence = (1 - best_match_distance) * 100
                                label = f"Unknown ({confidence:.1f}%)"
                                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
                        else:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                            label = "Unknown (No known faces)"
                            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
                except Exception as e:
                    continue
            
            for name in list(identity_counter.keys()):
                if name not in current_names or current_time - identity_counter[name]['last_seen'] > 3.0:
                    identity_counter.pop(name, None)
            
            cv2.imshow('Face Recognition Attendance', frame)
                                        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    attendance_system = FaceAttendanceSystem()
    attendance_system.load_known_faces()
    attendance_system.run() 