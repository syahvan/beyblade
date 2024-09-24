from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
import pandas as pd
from assigner import Assigner
from battle import Battle
import argparse


def main(input_video, model_path):
    # Read Video
    video_frames = read_video(input_video)

    # Initialize Tracker
    tracker = Tracker(model_path)

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')

    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # Assign Beyblade Teams
    team_assigner = Assigner()
    team_assigner.assign_beyblade_color(video_frames[240], 
                                    tracks['Beyblade'][240])
    
    for frame_num, beyblade_track in enumerate(tracks['Beyblade']):
        for beyblade_id, track in beyblade_track.items():
            team = team_assigner.get_beyblade_team(video_frames[frame_num],   
                                                   track['bbox'],
                                                   beyblade_id)
            tracks['Beyblade'][frame_num][beyblade_id]['team'] = team 
            tracks['Beyblade'][frame_num][beyblade_id]['beyblade_color'] = team_assigner.beyblade_colors[team]

    battle = Battle()
    battle.add_beyblade_status(tracks,video_frames)
    battle_stat = battle.get_battle_stat(tracks,'output/battle_log.csv')

    battle_time = battle.battle_time
    winner = battle.winner
    beyblade_time = battle.beyblade_time
    beyblade1_time = beyblade_time[1]
    beyblade2_time = beyblade_time[2]
    remaining_time = abs(beyblade2_time - beyblade1_time)
    total_collision = battle.total_collision

    battle_data = {
        'battle_time': [round(battle_time,2)],
        'winner': [winner],
        'beyblade1_time': [round(beyblade1_time,2)],
        'beyblade2_time': [round(beyblade2_time,2)],
        'remaining_time': [round(remaining_time,2)],
        'total_collision': [total_collision]
    }

    # Membuat DataFrame dari data
    df = pd.DataFrame(battle_data)

    # Menyimpan DataFrame ke file CSV
    df.to_csv('output/battle_results.csv', index=False)

    print('\n')
    print(f'Winner: Beyblade {winner}')
    print(f'Battle Time: {battle_time:.2f} s')

    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    ## Draw Battle Stat
    output_video_frames, winner_img = battle.draw_stat(output_video_frames, battle_stat, tracks)

    # Save video
    save_video(output_video_frames, 'output/output_video.avi')
    cv2.imwrite('output/winner.jpg', winner_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a beyblade battle video using a trained model.")
    parser.add_argument("--input_video", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained YOLO model file.")

    args = parser.parse_args()
    
    main(args.input_video, args.model_path)