from utils.record_capture import RecordVideo

def main():
    # change effect name to this list to use effects:
    ## "background_removal": change background to specific image from 'import_pic_path', use GPU to speed up recording
    effects_name = None
    record_directory_name = "recorded_videos"
    record_name = "recorded_video.mp4"
    # default pic path is import_pics/dh_cntt.jpg
    import_pic_path = None
    import_video_path = None
    # Currently record time is infinity, press 'q' to terminate recording
    record_time = 10

    recorder = RecordVideo(effects=effects_name)
    recorder.record_video_capture()

if __name__ == '__main__':
    main()