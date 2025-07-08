import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance
from collections import defaultdict
import time

class PlayerTracker:
    def __init__(self, model_path, confidence_threshold=0.3, iou_threshold=0.3, feature_threshold=0.4):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.feature_threshold = feature_threshold
        self.tracks = defaultdict(list)
        self.next_id = 0
        self.last_positions = {}
        self.last_features = {}
        self.max_frames_missing = 45  # Increased for goal events (3 seconds at 25 FPS)
        self.frame_count = 0

    def iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xi1, yi1 = max(x1, x2), max(y1, y2)
        xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h1)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        return inter_area / (box1_area + box2_area - inter_area + 1e-6)

    def extract_features(self, frame, box):
        x, y, w, h = box
        x, y, w, h = int(x), int(y), int(w), int(h)
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return None
        roi = cv2.resize(roi, (64, 128))  # Smaller for faster processing
        roi = roi / 255.0
        return roi.flatten()

    def assign_id(self, boxes, scores, features):
        current_boxes = []
        for box, score, feat in zip(boxes, scores, features):
            if score < self.confidence_threshold or feat is None:
                continue
            current_boxes.append((box, feat))

        matched_ids = set()
        for box, feat in current_boxes:
            best_iou = 0
            best_dist = float('inf')
            best_id = None
            for track_id, track in self.last_positions.items():
                if track_id in matched_ids:
                    continue
                last_box = track['box']
                iou_score = self.iou(box, last_box)
                if iou_score > best_iou and iou_score > self.iou_threshold:
                    best_iou = iou_score
                    best_id = track_id
                elif track_id in self.last_features:
                    dist = distance.cosine(feat, self.last_features[track_id])
                    if dist < best_dist and dist < self.feature_threshold:
                        best_dist = dist
                        best_id = track_id

            if best_id is not None:
                self.tracks[best_id].append((box, self.frame_count))
                self.last_positions[best_id] = {'box': box, 'frame': self.frame_count}
                self.last_features[best_id] = feat
                matched_ids.add(best_id)
            else:
                self.tracks[self.next_id].append((box, self.frame_count))
                self.last_positions[self.next_id] = {'box': box, 'frame': self.frame_count}
                self.last_features[self.next_id] = feat
                self.next_id += 1

        for track_id in list(self.last_positions.keys()):
            if self.frame_count - self.last_positions[track_id]['frame'] > self.max_frames_missing:
                del self.last_positions[track_id]
                if track_id in self.last_features:
                    del self.last_features[track_id]

    def process_video(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        start_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            results = self.model(frame, conf=self.confidence_threshold)
            
            boxes = []
            scores = []
            features = []
            
            for result in results:
                for box in result.boxes:
                    if box.cls == 1 or box.cls == 2:  # Goalkeepers (1) and players (2)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf.cpu().numpy()
                        box = [x1, y1, x2-x1, y2-y1]
                        feat = self.extract_features(frame, box)
                        boxes.append(box)
                        scores.append(conf)
                        features.append(feat)
            
            self.assign_id(boxes, scores, features)
            
            for track_id, track in self.tracks.items():
                for box, frame_num in track:
                    if frame_num == self.frame_count - 1:
                        x, y, w, h = box
                        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
                        cv2.putText(frame, f'ID: {track_id}', (int(x), int(y)-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            out.write(frame)
        
        end_time = time.time()
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        processing_time = end_time - start_time
        fps_processed = total_frames / processing_time if processing_time > 0 else 0
        print(f"Processed {total_frames} frames in {processing_time:.2f} seconds ({fps_processed:.2f} FPS)")

def main():
    model_path = "best.pt"
    video_path = "15sec_input_720p.mp4"
    output_path = "output_tracked.mp4"
    tracker = PlayerTracker(model_path)
    tracker.process_video(video_path, output_path)

if __name__ == "__main__":
    main()