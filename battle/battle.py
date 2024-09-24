import cv2
import numpy as np
import pandas as pd
import sys 
sys.path.append('../')
from utils import is_overlapping

class Battle:
    def __init__(self):
        self.vertices = np.array([(779,0),(175,400),(245,1080),(1750,1080),(1820,400),(1250,0)])
        self.start_battle_time = None
        self.end_battle_time = None
        self.beyblade_time = {1:0,2:0}
        self.battle_time = 0
        self.winner = None
        self.winner_bbox = None
        self.winner_frame_num = None
        self.total_collision = 0

    def add_beyblade_status(self,tracks,video_frames):
        prev_gray = None
        for frame_num, beyblade_track in enumerate(tracks['Beyblade']):
            frame_gray = cv2.cvtColor(video_frames[frame_num], cv2.COLOR_BGR2GRAY)
            for beyblade_id, track in beyblade_track.items():
                position = track['position']
                tracks['Beyblade'][frame_num][beyblade_id]['inside_polygon'] = cv2.pointPolygonTest(self.vertices, position, False) >= 0

                hand_bbox = tracks["Hand"][frame_num].get(1, {}).get('bbox', None)
                launcher_bbox = tracks["Launcher"][frame_num].get(1, {}).get('bbox', None)
                IoU_hand = is_overlapping(track['bbox'], hand_bbox)
                IoU_launcher = is_overlapping(track['bbox'], launcher_bbox)
                if IoU_hand or IoU_launcher:
                    tracks['Beyblade'][frame_num][beyblade_id]['is_taken'] = True
                else:
                    tracks['Beyblade'][frame_num][beyblade_id]['is_taken'] = False

                if prev_gray is not None:
                    # Optical Flow Farneback
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                    x_min = max(0, int(position[0]) - 5)
                    x_max = min(magnitude.shape[1], int(position[0]) + 5)
                    y_min = max(0, int(position[1]) - 5)
                    y_max = min(magnitude.shape[0], int(position[1]) + 5)
                    movement = np.mean(magnitude[y_min:y_max, x_min:x_max])
                    if movement > 1.0:
                        tracks['Beyblade'][frame_num][beyblade_id]['is_rotating'] = True
                    else:
                        tracks['Beyblade'][frame_num][beyblade_id]['is_rotating'] = False
                else:
                    tracks['Beyblade'][frame_num][beyblade_id]['is_rotating'] = False
            prev_gray = frame_gray


    def check_battle(self,frame_num,beyblade_track):
        beyblade_inside_polygon = []
        for beyblade_id, track in beyblade_track.items():
            inside_polygon = track['inside_polygon']
            taken = track['is_taken']
            rotate = track['is_rotating']
            if inside_polygon and not taken and rotate:
                team = track['team']
                beyblade_inside_polygon.append(team)
                if self.start_battle_time:
                    time_now = frame_num / 30
                    self.beyblade_time[team] = time_now - self.start_battle_time
                if self.winner == team:
                    self.winner_bbox = track['bbox']
                    self.winner_frame_num = frame_num

        if len(beyblade_inside_polygon) >= 2:
            if not self.start_battle_time:
                self.start_battle_time = frame_num / 30
                return 1, self.battle_time, self.beyblade_time
            else:
                self.end_battle_time = frame_num / 30
                self.battle_time = self.end_battle_time - self.start_battle_time
                return 1, self.battle_time, self.beyblade_time
        elif len(beyblade_inside_polygon) == 1 and self.start_battle_time:
            self.winner = beyblade_inside_polygon[0]
            return 2, self.battle_time, self.beyblade_time
        else:
            if self.start_battle_time and self.winner:
                return 2, self.battle_time, self.beyblade_time
            else:
                return 0, self.battle_time, self.beyblade_time

    def get_battle_stat(self, tracks, battle_log_path):
        battle_stat = {
            'battle_status':[],
            'battle_time':[],
            'beyblade1_time':[],
            'beyblade2_time':[],
            'collision':[],
            'total_collision':[]
        }

        battle_log = pd.DataFrame(columns=["frame_num", "battle_status", "collision"])

        for frame_num, beyblade_track in enumerate(tracks['Beyblade']):
            battle_stat["battle_status"].append({})
            battle_stat["battle_time"].append({})
            battle_stat["beyblade1_time"].append({})
            battle_stat["beyblade2_time"].append({})
            battle_stat["collision"].append({})
            battle_stat["total_collision"].append({})

            battle_status, battle_time, beyblade_time = self.check_battle(frame_num,beyblade_track)
            battle_stat['battle_status'][frame_num] = battle_status
            battle_stat['battle_time'][frame_num] = battle_time
            battle_stat['beyblade1_time'][frame_num] = beyblade_time[1]
            battle_stat['beyblade2_time'][frame_num] = beyblade_time[2]

            beyblade_bbox = []
            for beyblade_id, track in beyblade_track.items():
                beyblade_bbox.append(track['bbox'])
            if len(beyblade_bbox) == 2:
                IoU_beyblade = is_overlapping(beyblade_bbox[0], beyblade_bbox[1])
                battle_stat['collision'][frame_num] = IoU_beyblade
                if IoU_beyblade:
                    self.total_collision += 1
                battle_stat['total_collision'][frame_num] = self.total_collision
            else:
                battle_stat['total_collision'][frame_num] = self.total_collision
                battle_stat['collision'][frame_num] = False

            battle_log_data = [frame_num,battle_status,battle_stat['collision'][frame_num]]
            battle_log_series = pd.Series(battle_log_data, index=battle_log.columns)
            battle_log = pd.concat([battle_log, battle_log_series.to_frame().T], ignore_index=True)

        battle_log.to_csv(battle_log_path, index=False)

        return battle_stat

    def draw_stat(self, video_frames, battle_stat, tracks):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            cv2.polylines(frame, [self.vertices], isClosed=True, color=(255, 0, 0), thickness=2)

            battle_status = battle_stat["battle_status"][frame_num]
            battle_time = battle_stat["battle_time"][frame_num]
            beyblade1_time = battle_stat["beyblade1_time"][frame_num]
            beyblade2_time = battle_stat["beyblade2_time"][frame_num]
            total_collision = battle_stat["total_collision"][frame_num]

            if battle_status == 0:
                frame = cv2.putText(frame,f"Waiting Opponent...",(50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            else:
                frame = cv2.putText(frame,f"Battle Time: {battle_time:.2f} s",(50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
                frame = cv2.putText(frame,f"Beyblade 1 Time: {beyblade1_time:.2f} s",(50,90), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
                frame = cv2.putText(frame,f"Beyblade 2 Time: {beyblade2_time:.2f} s",(50,130), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
                frame = cv2.putText(frame,f"Total Collision: {total_collision}",(50,170), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            if frame_num == self.winner_frame_num:
                x1, y1, x2, y2 = [int(value) for value in self.winner_bbox]
                winner_img = frame[y1:y2, x1:x2]

            output_video_frames.append(frame)

        return output_video_frames, winner_img