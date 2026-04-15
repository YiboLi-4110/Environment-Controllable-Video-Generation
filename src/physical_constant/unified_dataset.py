import torch, torchvision, imageio, os, json, pandas
import numpy as np
import imageio.v3 as iio
from PIL import Image
from tqdm import tqdm
from controlnet_aux import CannyDetector
import pickle
import cv2
import glob
import math
from torchvision import transforms
from einops import rearrange
from typing import List


def load_video_to_pil(video_path: str) -> List[Image.Image]:
    """
    Loads a video file from the given path and returns a list of its frames
    as PIL Image objects.

    Args:
        video_path (str): The file path to the video.

    Returns:
        List[Image.Image]: A list of frames as PIL Images.

    Raises:
        IOError: If the video file cannot be opened or found.
    """
    
    # 1. Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        raise IOError(f"Error: Could not open video file at {video_path}")

    pil_frames: List[Image.Image] = []
    
    try:
        while True:
            # 2. Read one frame at a time
            # ret is a boolean: True if a frame was read, False otherwise
            # frame is the NumPy array (in BGR format)
            ret, frame = cap.read()

            if not ret:
                # If ret is False, we've reached the end of the video
                break

            # 3. Convert frame from BGR (OpenCV) to RGB (PIL)
            # This is a crucial step!
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 4. Convert the NumPy array to a PIL Image
            pil_image = Image.fromarray(frame_rgb)

            # 5. Add the PIL Image to the list
            pil_frames.append(pil_image)
    
    finally:
        # 6. Release the video capture object
        # This is done in a 'finally' block to ensure it happens
        # even if an error occurs during frame reading.
        cap.release()

    return pil_frames


class DataProcessingPipeline:
    def __init__(self, operators=None):
        self.operators: list[DataProcessingOperator] = [] if operators is None else operators
        
    def __call__(self, data):
        for operator in self.operators:
            data = operator(data)
        return data
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline(self.operators + pipe.operators)



class DataProcessingOperator:
    def __call__(self, data):
        raise NotImplementedError("DataProcessingOperator cannot be called directly.")
    
    def __rshift__(self, pipe):
        if isinstance(pipe, DataProcessingOperator):
            pipe = DataProcessingPipeline([pipe])
        return DataProcessingPipeline([self]).__rshift__(pipe)



class DataProcessingOperatorRaw(DataProcessingOperator):
    def __call__(self, data):
        return data



class ToInt(DataProcessingOperator):
    def __call__(self, data):
        return int(data)



class ToFloat(DataProcessingOperator):
    def __call__(self, data):
        return float(data)



class ToStr(DataProcessingOperator):
    def __init__(self, none_value=""):
        self.none_value = none_value
    
    def __call__(self, data):
        if data is None: data = self.none_value
        return str(data)



class LoadImage(DataProcessingOperator):
    def __init__(self, convert_RGB=True):
        self.convert_RGB = convert_RGB
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        return image



class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height, width, max_pixels, height_division_factor, width_division_factor):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image
    
    def get_height_width(self, image):
        if self.height is None or self.width is None:
            width, height = image.size
            if width * height > self.max_pixels:
                scale = (width * height / self.max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image



class ToList(DataProcessingOperator):
    def __call__(self, data):
        return [data]
    


class LoadVideo(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, reader):
        num_frames = self.num_frames
        if int(reader.count_frames()) < num_frames:
            num_frames = int(reader.count_frames())
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        try:
            # This is the line that can fail
            reader = imageio.get_reader(data)
            num_frames = self.get_num_frames(reader)
            frames = []
            for frame_id in range(num_frames):
                frame = reader.get_data(frame_id)
                frame = Image.fromarray(frame)
                frame = self.frame_processor(frame)
                frames.append(frame)
            reader.close()
            return frames
        except Exception as e:
            # If any error occurs, log it and return None instead of crashing
            print(f"WARNING: Skipping corrupted or unreadable video file: {data}. Error: {e}")
            return None



class SequencialProcess(DataProcessingOperator):
    def __init__(self, operator=lambda x: x):
        self.operator = operator
        
    def __call__(self, data):
        return [self.operator(i) for i in data]



class LoadGIF(DataProcessingOperator):
    def __init__(self, num_frames=81, time_division_factor=4, time_division_remainder=1, frame_processor=lambda x: x):
        self.num_frames = num_frames
        self.time_division_factor = time_division_factor
        self.time_division_remainder = time_division_remainder
        # frame_processor is build in the video loader for high efficiency.
        self.frame_processor = frame_processor
        
    def get_num_frames(self, path):
        num_frames = self.num_frames
        images = iio.imread(path, mode="RGB")
        if len(images) < num_frames:
            num_frames = len(images)
            while num_frames > 1 and num_frames % self.time_division_factor != self.time_division_remainder:
                num_frames -= 1
        return num_frames
        
    def __call__(self, data: str):
        num_frames = self.get_num_frames(data)
        frames = []
        images = iio.imread(data, mode="RGB")
        for img in images:
            frame = Image.fromarray(img)
            frame = self.frame_processor(frame)
            frames.append(frame)
            if len(frames) >= num_frames:
                break
        return frames
    


class RouteByExtensionName(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data: str):
        file_ext_name = data.split(".")[-1].lower()
        for ext_names, operator in self.operator_map:
            if ext_names is None or file_ext_name in ext_names:
                return operator(data)
        raise ValueError(f"Unsupported file: {data}")



class RouteByType(DataProcessingOperator):
    def __init__(self, operator_map):
        self.operator_map = operator_map
        
    def __call__(self, data):
        for dtype, operator in self.operator_map:
            if dtype is None or isinstance(data, dtype):
                return operator(data)
        raise ValueError(f"Unsupported data: {data}")



class LoadTorchPickle(DataProcessingOperator):
    def __init__(self, map_location="cpu"):
        self.map_location = map_location
        
    def __call__(self, data):
        return torch.load(data, map_location=self.map_location, weights_only=False)



class ToAbsolutePath(DataProcessingOperator):
    def __init__(self, base_path=""):
        self.base_path = base_path
        
    def __call__(self, data):
        return os.path.join(self.base_path, data)



class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.data = []
        self.cached_data = []
        self.load_from_cache = metadata_path is None
        self.load_metadata(metadata_path)
    
    @staticmethod
    def default_image_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
            (list, SequencialProcess(ToAbsolutePath(base_path) >> LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor))),
        ])
    
    @staticmethod
    def default_video_operator(
        base_path="",
        max_pixels=1920*1080, height=None, width=None,
        height_division_factor=16, width_division_factor=16,
        num_frames=81, time_division_factor=4, time_division_remainder=1,
    ):
        return RouteByType(operator_map=[
            (str, ToAbsolutePath(base_path) >> RouteByExtensionName(operator_map=[
                (("jpg", "jpeg", "png", "webp"), LoadImage() >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor) >> ToList()),
                (("gif",), LoadGIF(num_frames, time_division_factor, time_division_remainder) >> ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor)),
                (("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"), LoadVideo(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
            ])),
        ])
        
    def search_for_cached_data_files(self, path):
        for file_name in os.listdir(path):
            subpath = os.path.join(path, file_name)
            if os.path.isdir(subpath):
                self.search_for_cached_data_files(subpath)
            elif subpath.endswith(".pth"):
                self.cached_data.append(subpath)
    
    def load_metadata(self, metadata_path):
        if metadata_path is None:
            print("No metadata_path. Searching for cached data files.")
            self.search_for_cached_data_files(self.base_path)
            print(f"{len(self.cached_data)} cached data files found.")
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    metadata.append(json.loads(line.strip()))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        if self.load_from_cache:
            data = self.cached_data[data_id % len(self.cached_data)]
            data = self.cached_data_operator(data)
        else:
            data = self.data[data_id % len(self.data)].copy()
            for key in self.data_file_keys:
                if key in data:
                    if key in self.special_operator_map:
                        data[key] = self.special_operator_map[key]
                    elif key in self.data_file_keys:
                        data[key] = self.main_data_operator(data[key])
        return data

    def __len__(self):
        if self.load_from_cache:
            return len(self.cached_data) * self.repeat
        else:
            return len(self.data) * self.repeat
        
    def check_data_equal(self, data1, data2):
        # Debug only
        if len(data1) != len(data2):
            return False
        for k in data1:
            if data1[k] != data2[k]:
                return False
        return True


class ControlSignalDataset_Falling(torch.utils.data.Dataset):
    """
    Dataset for falling-ball/cube videos with gravity as the controllable physical parameter.
    Generates a 3-channel control signal video:
        Ch0: normalized gravity g' = (g - 1) / (20 - 1) filled uniformly across all pixels/frames
        Ch1: vertical gravitational field, linear gradient from +1 (top) to -1 (bottom)
        Ch2: all zeros (reserved)
    """

    GRAVITY_MIN = 1.0
    GRAVITY_MAX = 20.0

    def __init__(
        self,
        base_path=None,
        metadata_path=None,
        repeat=1,
        num_frames=81,
        height=480,
        width=832,
        is_validation_dataset=False,
        control_signal_encoding="num_encode",
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_validation_dataset = is_validation_dataset
        self.control_signal_encoding = control_signal_encoding

        if self.is_validation_dataset:
            self.media_type = "image"
            self.blob_ext = "*.jpg"
        else:
            self.media_type = "video"
            self.blob_ext = "*.mp4"

        self.to_tensor_transform = transforms.ToTensor()
        self.to_pil_transform = transforms.ToPILImage()

        self._vertical_field = torch.linspace(1.0, -1.0, steps=self.height).unsqueeze(1).expand(self.height, self.width)

        self.load_metadata()

    def load_metadata(self):
        if not self.is_validation_dataset:
            file_paths = glob.glob(os.path.join(self.base_path, self.blob_ext))
            file_names = set(os.path.basename(x) for x in file_paths)
            self.df = pandas.read_csv(self.metadata_path)

            self.df['checked'] = self.df[self.media_type].map(lambda x, files=file_names: int(x in files))
            self.df = self.df[self.df['checked'] == 1]
            self.df.reset_index(drop=True, inplace=True)

            print(f"[ControlSignalDataset_Falling] Loaded {len(self.df)} training samples from {self.metadata_path}")
            print(f"  gravity range in csv: [{self.df['gravity'].min()}, {self.df['gravity'].max()}]")
        else:
            file_paths = glob.glob(os.path.join(self.base_path, "images", self.blob_ext))
            file_names = set(os.path.basename(x) for x in file_paths)
            self.df = pandas.read_csv(self.metadata_path)

            self.df['checked'] = self.df[self.media_type].map(lambda x, files=file_names: int(x in files))
            self.df = self.df[self.df['checked'] == 1]
            self.df.reset_index(drop=True, inplace=True)

            print(f"[ControlSignalDataset_Falling] Loaded {len(self.df)} validation samples from {self.metadata_path}")
            if len(self.df) > 0:
                print(f"  gravity range in csv: [{self.df['gravity'].min()}, {self.df['gravity'].max()}]")

    def _normalize_gravity(self, g):
        temp_gravity = (g - self.GRAVITY_MIN) / (self.GRAVITY_MAX - self.GRAVITY_MIN)
        norm_gravity = 2*temp_gravity - 1.  # -> [-1, 1]
        return norm_gravity

    def _generate_control_video_num_encode(self, gravity, num_frames, num_channels=3, height=480, width=832):
        controlnet_signal = torch.zeros((num_frames, num_channels, height, width))

        g_norm = self._normalize_gravity(gravity)
        controlnet_signal[:, 0, :, :] = g_norm

        controlnet_signal[:, 1, :, :] = self._vertical_field[:height, :width]

        return rearrange(controlnet_signal, 'f c h w -> f h w c').to(torch.bfloat16)

    def _generate_control_video_visual_encode(self, gravity, num_frames, num_channels=3, height=480, width=832):
        """
        Generate an arrow-based control signal video.
        The arrow is white on a black background, pointing downward.
        Vertically it spans the full height; horizontally its width is
        proportional to the gravity value (min→1 pixel, max→full width).
        The arrow consists of a shaft (1/3 of arrow_width) and a triangular
        arrowhead (bottom 1/4 of height) that widens to the full arrow_width.
        All frames are identical (static signal).
        """
        frame = torch.zeros((num_channels, height, width))

        ratio = (gravity - self.GRAVITY_MIN) / (self.GRAVITY_MAX - self.GRAVITY_MIN)
        ratio = max(0.0, min(1.0, ratio))
        arrow_width = max(1, int(ratio * width))

        center_x = width // 2
        half_arrow = arrow_width // 2
        ax_start = max(0, center_x - half_arrow)
        ax_end = min(width, ax_start + arrow_width)

        shaft_width = max(1, arrow_width // 3)
        half_shaft = shaft_width // 2
        sx_start = max(0, center_x - half_shaft)
        sx_end = min(width, sx_start + shaft_width)

        arrowhead_h = max(1, height // 4)
        shaft_end = height - arrowhead_h

        # Draw shaft
        frame[:, :shaft_end, sx_start:sx_end] = 1.0

        # Draw arrowhead: triangle that widens from shaft_width at top to arrow_width at bottom
        for dy in range(arrowhead_h):
            cur_w = shaft_width + int((arrow_width - shaft_width) * dy / max(1, arrowhead_h - 1))
            half_cw = cur_w // 2
            tx_start = max(0, center_x - half_cw)
            tx_end = min(width, tx_start + cur_w)
            frame[:, shaft_end + dy, tx_start:tx_end] = 1.0

        # Broadcast single frame to all frames (static signal)
        controlnet_signal = frame.unsqueeze(0).expand(num_frames, -1, -1, -1).contiguous()

        return rearrange(controlnet_signal, 'f c h w -> f h w c').to(torch.bfloat16)

    def _generate_control_video(self, gravity, num_frames, num_channels=3, height=480, width=832):
        if self.control_signal_encoding == "visual_encode":
            return self._generate_control_video_visual_encode(gravity, num_frames, num_channels, height, width)
        else:
            return self._generate_control_video_num_encode(gravity, num_frames, num_channels, height, width)

    def get_batch(self, idx):
        item = self.df.iloc[idx]
        caption = str(item['caption'])
        file_name = str(item[self.media_type])
        gravity = float(item['gravity'])

        if self.is_validation_dataset:
            file_path = os.path.join(self.base_path, "images", file_name)
            image = Image.open(file_path).convert("RGB")
            desired_size = (self.width, self.height)
            if image.size != desired_size:
                image = image.resize(desired_size, resample=Image.Resampling.LANCZOS)
            pixel_values = self.to_tensor_transform(image)   # (3, H, W) in [0, 1]
            pixel_values = 2 * pixel_values - 1              # -> [-1, 1]
            file_id = file_name.rsplit(".", 1)[0]
        else:
            file_path = os.path.join(self.base_path, file_name)
            pixel_values = load_video_to_pil(file_path)

            if len(pixel_values) > self.num_frames:
                indices = np.linspace(0, len(pixel_values) - 1, self.num_frames, dtype=int)
                pixel_values = [pixel_values[i] for i in indices]
            pixel_values = pixel_values[:self.num_frames]

            pixel_values = torch.stack([self.to_tensor_transform(img) for img in pixel_values])  # (F, 3, H, W) in [0, 1]
            pixel_values = 2 * pixel_values - 1  # -> [-1, 1]
            file_id = file_name.rsplit(".mp4", 1)[0]

        return pixel_values, caption, gravity, file_id

    def __getitem__(self, data_id):
        pixel_values, caption, gravity, file_id = self.get_batch(data_id % len(self.df))

        control_video = self._generate_control_video(
            gravity,
            num_frames=self.num_frames,
            num_channels=3,
            height=self.height,
            width=self.width,
        )

        pixel_values = (pixel_values + 1) / 2  # -> [0, 1]
        if not self.is_validation_dataset:
            pil_image_list = [self.to_pil_transform(tensor) for tensor in pixel_values]
        else:
            pil_image_list = [self.to_pil_transform(pixel_values)]

        return {
            "video": pil_image_list,
            "prompt": caption,
            "control_video": control_video,
            "gravity": gravity,
            "file_id": file_id,
        }

    def __len__(self):
        return len(self.df) * self.repeat


class ControlSignalDataset_Sliding(torch.utils.data.Dataset):
    """
    Dataset for sliding-ball/cube videos with gravity as the controllable physical parameter.
    Generates a 3-channel control signal video:
        Ch0: normalized gravity g' = (g - 1) / (20 - 1) filled uniformly across all pixels/frames
        Ch1: vertical gravitational field, linear gradient from +1 (top) to -1 (bottom)
        Ch2: all zeros (reserved)
    """

    GRAVITY_MIN = 1.0
    GRAVITY_MAX = 20.0

    def __init__(
        self,
        base_path=None,
        metadata_path=None,
        repeat=1,
        num_frames=81,
        height=480,
        width=832,
        is_validation_dataset=False,
        control_signal_encoding="num_encode",
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_validation_dataset = is_validation_dataset
        self.control_signal_encoding = control_signal_encoding

        if self.is_validation_dataset:
            self.media_type = "image"
            self.blob_ext = "*.jpg"
        else:
            self.media_type = "video"
            self.blob_ext = "*.mp4"

        self.to_tensor_transform = transforms.ToTensor()
        self.to_pil_transform = transforms.ToPILImage()

        self._vertical_field = torch.linspace(1.0, -1.0, steps=self.height).unsqueeze(1).expand(self.height, self.width)

        self.load_metadata()

    def load_metadata(self):
        if not self.is_validation_dataset:
            file_paths = glob.glob(os.path.join(self.base_path, self.blob_ext))
            file_names = set(os.path.basename(x) for x in file_paths)
            self.df = pandas.read_csv(self.metadata_path)

            self.df['checked'] = self.df[self.media_type].map(lambda x, files=file_names: int(x in files))
            self.df = self.df[self.df['checked'] == 1]
            self.df.reset_index(drop=True, inplace=True)

            print(f"[ControlSignalDataset_Sliding] Loaded {len(self.df)} training samples from {self.metadata_path}")
            print(f"  gravity range in csv: [{self.df['gravity'].min()}, {self.df['gravity'].max()}]")
        else:
            file_paths = glob.glob(os.path.join(self.base_path, "images", self.blob_ext))
            file_names = set(os.path.basename(x) for x in file_paths)
            self.df = pandas.read_csv(self.metadata_path)

            self.df['checked'] = self.df[self.media_type].map(lambda x, files=file_names: int(x in files))
            self.df = self.df[self.df['checked'] == 1]
            self.df.reset_index(drop=True, inplace=True)

            print(f"[ControlSignalDataset_Sliding] Loaded {len(self.df)} validation samples from {self.metadata_path}")
            if len(self.df) > 0:
                print(f"  gravity range in csv: [{self.df['gravity'].min()}, {self.df['gravity'].max()}]")

    def _normalize_gravity(self, g):
        temp_gravity = (g - self.GRAVITY_MIN) / (self.GRAVITY_MAX - self.GRAVITY_MIN)
        norm_gravity = 2*temp_gravity - 1.  # -> [-1, 1]
        return norm_gravity

    def _generate_control_video_num_encode(self, gravity, num_frames, num_channels=3, height=480, width=832):
        controlnet_signal = torch.zeros((num_frames, num_channels, height, width))

        g_norm = self._normalize_gravity(gravity)
        controlnet_signal[:, 0, :, :] = g_norm

        controlnet_signal[:, 1, :, :] = self._vertical_field[:height, :width]

        return rearrange(controlnet_signal, 'f c h w -> f h w c').to(torch.bfloat16)

    def _generate_control_video_visual_encode(self, gravity, num_frames, num_channels=3, height=480, width=832):
        """
        Generate an arrow-based control signal video.
        The arrow is white on a black background, pointing downward.
        Vertically it spans the full height; horizontally its width is
        proportional to the gravity value (min→1 pixel, max→full width).
        The arrow consists of a shaft (1/3 of arrow_width) and a triangular
        arrowhead (bottom 1/4 of height) that widens to the full arrow_width.
        All frames are identical (static signal).
        """
        frame = torch.zeros((num_channels, height, width))

        ratio = (gravity - self.GRAVITY_MIN) / (self.GRAVITY_MAX - self.GRAVITY_MIN)
        ratio = max(0.0, min(1.0, ratio))
        arrow_width = max(1, int(ratio * width))

        center_x = width // 2
        half_arrow = arrow_width // 2
        ax_start = max(0, center_x - half_arrow)
        ax_end = min(width, ax_start + arrow_width)

        shaft_width = max(1, arrow_width // 3)
        half_shaft = shaft_width // 2
        sx_start = max(0, center_x - half_shaft)
        sx_end = min(width, sx_start + shaft_width)

        arrowhead_h = max(1, height // 4)
        shaft_end = height - arrowhead_h

        # Draw shaft
        frame[:, :shaft_end, sx_start:sx_end] = 1.0

        # Draw arrowhead: triangle that widens from shaft_width at top to arrow_width at bottom
        for dy in range(arrowhead_h):
            cur_w = shaft_width + int((arrow_width - shaft_width) * dy / max(1, arrowhead_h - 1))
            half_cw = cur_w // 2
            tx_start = max(0, center_x - half_cw)
            tx_end = min(width, tx_start + cur_w)
            frame[:, shaft_end + dy, tx_start:tx_end] = 1.0

        # Broadcast single frame to all frames (static signal)
        controlnet_signal = frame.unsqueeze(0).expand(num_frames, -1, -1, -1).contiguous()

        return rearrange(controlnet_signal, 'f c h w -> f h w c').to(torch.bfloat16)

    def _generate_control_video(self, gravity, num_frames, num_channels=3, height=480, width=832):
        if self.control_signal_encoding == "visual_encode":
            return self._generate_control_video_visual_encode(gravity, num_frames, num_channels, height, width)
        else:
            return self._generate_control_video_num_encode(gravity, num_frames, num_channels, height, width)

    def get_batch(self, idx):
        item = self.df.iloc[idx]
        caption = str(item['caption'])
        file_name = str(item[self.media_type])
        gravity = float(item['gravity'])

        if self.is_validation_dataset:
            file_path = os.path.join(self.base_path, "images", file_name)
            image = Image.open(file_path).convert("RGB")
            desired_size = (self.width, self.height)
            if image.size != desired_size:
                image = image.resize(desired_size, resample=Image.Resampling.LANCZOS)
            pixel_values = self.to_tensor_transform(image)   # (3, H, W) in [0, 1]
            pixel_values = 2 * pixel_values - 1              # -> [-1, 1]
            file_id = file_name.rsplit(".", 1)[0]
        else:
            file_path = os.path.join(self.base_path, file_name)
            pixel_values = load_video_to_pil(file_path)

            if len(pixel_values) > self.num_frames:
                indices = np.linspace(0, len(pixel_values) - 1, self.num_frames, dtype=int)
                pixel_values = [pixel_values[i] for i in indices]
            pixel_values = pixel_values[:self.num_frames]

            pixel_values = torch.stack([self.to_tensor_transform(img) for img in pixel_values])  # (F, 3, H, W) in [0, 1]
            pixel_values = 2 * pixel_values - 1  # -> [-1, 1]
            file_id = file_name.rsplit(".mp4", 1)[0]

        return pixel_values, caption, gravity, file_id

    def __getitem__(self, data_id):
        pixel_values, caption, gravity, file_id = self.get_batch(data_id % len(self.df))

        control_video = self._generate_control_video(
            gravity,
            num_frames=self.num_frames,
            num_channels=3,
            height=self.height,
            width=self.width,
        )

        pixel_values = (pixel_values + 1) / 2  # -> [0, 1]
        if not self.is_validation_dataset:
            pil_image_list = [self.to_pil_transform(tensor) for tensor in pixel_values]
        else:
            pil_image_list = [self.to_pil_transform(pixel_values)]

        return {
            "video": pil_image_list,
            "prompt": caption,
            "control_video": control_video,
            "gravity": gravity,
            "file_id": file_id,
        }

    def __len__(self):
        return len(self.df) * self.repeat