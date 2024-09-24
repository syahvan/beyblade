from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()

    
    def add_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    position = get_center_of_bbox(bbox)
                    tracks[object][frame_num][track_id]['position'] = position


    def detect_frames(self, frames):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections


    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks={
            "Beyblade":[],
            "Hand":[],
            "Launcher":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["Beyblade"].append({})
            tracks["Hand"].append({})
            tracks["Launcher"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['Beyblade']:
                    tracks["Beyblade"][frame_num][track_id] = {"bbox":bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['Hand']:
                    tracks["Hand"][frame_num][1] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['Launcher']:
                    tracks["Launcher"][frame_num][1] = {"bbox":bbox}
            
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks


    def draw_triangle(self,frame,bbox,color,track_id=None):
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        rectangle_width = 140
        x1_rect = x - rectangle_width//2
        x2_rect = x + rectangle_width//2
        y1_rect = y - 50
        y2_rect = y - 30

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect + 10
            
            cv2.putText(
                frame,
                f"Beyblade #{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                2
            )

        return frame

    def draw_annotations(self, video_frames, tracks):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            beyblade_dict = tracks["Beyblade"][frame_num]

            # Draw Beyblade
            for track_id, beyblade in beyblade_dict.items():
                color = beyblade.get("beyblade_color",(0,0,255))
                team = beyblade.get("team",1)
                bbox = beyblade['bbox']

                frame = self.draw_triangle(frame,bbox,color,team)

            output_video_frames.append(frame)

        return output_video_frames