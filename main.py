from utils.record_capture import RecordVideo

def main():
    effects_name = "background_removal"
    record_directory_name = "recorded_videos"
    record_name = "recorded_video.mp4"
    import_pic_path = None
    import_video_path = None
    record_time = 10

    recorder = RecordVideo()
    recorder.record_video_capture()

if __name__ == '__main__':
    main()