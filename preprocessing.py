from typing import Dict
from pathlib import Path
import os
import glob

import ffmpeg

CWD = Path(__file__).parent

DEFAULT_INPUT_DIR = 'raw-videos'
DEFAULT_INPUT_PATH = os.path.join(CWD, DEFAULT_INPUT_DIR)

DEFAULT_OUTPUT_DIR = 'output'
DEFAULT_OUTPUT_PATH = os.path.join(CWD, DEFAULT_OUTPUT_DIR)


class MoviePreprocessor(object):
    """
    General utility class for preprocessing videos for use in StyleGAN
    training pipelines
    """

    def __init__(self, input_path: str = DEFAULT_INPUT_PATH, output_path: str = DEFAULT_OUTPUT_PATH,
                 ignore_vertical: bool = True) -> None:
        """
        :param input_path: Directory which contains movie files to preprocess
        :param output_path: Output directory for storing individual frames
        :param ignore_vertical: Ignore vertical videos
        """
        self.input_path = input_path
        self.output_path = output_path
        self.ignore_vertical = ignore_vertical

        self.curr_stream = None
        self.curr_stream_meta = None

        self.input_paths = None
        self.output_paths = None

    @staticmethod
    def file_is_video(f: str) -> bool:
        """
        Determines if a file is a video based on the existence of a video stream somewhere in the container
        """
        try:
            meta = ffmpeg.probe(f)
        except ffmpeg.Error as e:
            stderr = e.stderr.decode('utf-8')
            if 'Invalid data found when processing input' in stderr:
                return False
            else:
                print(stderr)
                raise

        return any([s.get('codec_type') == 'video' for s in meta.get('streams', [])])

    @staticmethod
    def video_is_horizontal(f: str) -> bool:
        """
        Determines if a video is vertically aligned
        """
        meta = ffmpeg.probe(f)
        stream = [s for s in meta['streams'] if s['codec_type'] == 'video'][0]
        return stream['width'] > stream['height']

    def _init_input_paths(self) -> filter:
        """
        Gets paths of all movie files in the input directory
        """
        files = glob.glob(os.path.join(self.input_path, '*'))
        movie_files = filter(self.file_is_video, files)
        if self.ignore_vertical:
            movie_files = filter(self.video_is_horizontal, movie_files)

        return movie_files

    def _init_output_paths(self) -> Dict[str: str]:
        output = {}
        for path in self.input_paths:
            output_dir = path.split('/')[1]
            output_dir.strip('.')
            output[path] = (os.path.join(self.output_path, output_dir))

        return output

    def setup(self) -> None:
        self.input_paths = self._init_input_paths()
        self.output_paths = self._init_output_paths()

    def _convert_stream_to_frames(self) -> None:
        raise NotImplementedError

    def _adjust_stream_dimensions(self) -> None:
        raise NotImplementedError

    def process_videos(self) -> None:
        for path in self.input_paths:
            self.curr_stream = ffmpeg.input(path)
            self._adjust_stream_dimensions()
            self._convert_stream_to_frames()

    def start(self) -> None:
        self.setup()
        self.process_videos()


if __name__ == '__main__':
    p = MoviePreprocessor()
    p = list(p.input_paths)
    print(p)
    print(len(p))
