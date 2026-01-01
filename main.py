from utils.utils import read_video, save_video
from trackers.tracker import Tracker
def main():
    # Read video
    video_frames = read_video('Input Videos/08fd33_4.mp4')

    # Initialize tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,read_from_stub=True,stub_path='stubs/track_stubs.pkl')

    #Draw output
    ## Draw objecct tracks
    output_video_frames = tracker.draw_annotations(video_frames,tracks)
    
    # Save video
    save_video(output_video_frames, 'Output Videos/output2.mp4')

if __name__ == "__main__":
    main()