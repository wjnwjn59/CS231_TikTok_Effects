import argparse
from utils.record_capture import RecordVideo

def parse_arguments():
    """
        Parse the command line arguments of the program.
    """

    parser = argparse.ArgumentParser(
        description="Parameters for video recording"
    )
    parser.add_argument(
        "--output_dir", type=str, nargs="?", help="The output directory", default="recorded_videos/"
    )
    parser.add_argument(
        "--output_video_name", type=str, nargs="?", help="The name of the recorded video", default="recorded_video.mp4"
    )
    parser.add_argument(
        "-im",
        "--img_path",
        type=str,
        nargs="?",
        help="Image path to import to effect",
        default="import_pics/dh_cntt.jpg", 
    )
    parser.add_argument(
        "-vid",
        "--video_path",
        type=str,
        nargs="?",
        help="Video path to import to effect",
        default=None,
    )
    parser.add_argument(
        "-e",
        "--effect_name",
        type=str,
        nargs="?",
        help="Name of the effect to apply to recording video",
        default=None,
    )
    parser.add_argument(
        "-t",
        "--time_record",
        type=int,
        nargs="?",
        help="Record time",
        default=10,
    )

    return parser.parse_args()

def main():
    # Argument parsing
    args = parse_arguments()
    # change effect name to this list to use effects:
    ## "background_removal": change background to specific image from 'import_pic_path', use GPU to speed up recording
    ## "zoom_in"
    ## "sepia"
    effect_name = args.effect_name
    record_directory_name = args.output_dir
    record_name = args.output_video_name
    # default pic path is import_pics/dh_cntt.jpg
    import_pic_path = args.img_path
    import_video_path = args.video_path
    # Currently record time is infinity, press 'q' to terminate recording
    record_time = args.time_record

    recorder = RecordVideo(effects=effect_name)
    recorder.record_video_capture()

if __name__ == '__main__':
    main()