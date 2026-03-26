import torch, torchvision, imageio, os
import imageio.v3 as iio
from PIL import Image


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
    def __init__(self, convert_RGB=True, convert_RGBA=False):
        self.convert_RGB = convert_RGB
        self.convert_RGBA = convert_RGBA
    
    def __call__(self, data: str):
        image = Image.open(data)
        if self.convert_RGB: image = image.convert("RGB")
        if self.convert_RGBA: image = image.convert("RGBA")
        return image


class ImageCropAndResize(DataProcessingOperator):
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1):
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
    
    def get_height_width(self, image, max_pixels=None):
        if max_pixels is None:
            max_pixels = self.max_pixels
        if self.height is None or self.width is None:
            width, height = image.size
            if max_pixels is not None and width * height > max_pixels:
                scale = (width * height / max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    def __call__(self, data: Image.Image):
        image = self.crop_and_resize(data, *self.get_height_width(data))
        return image


class ImageCropAndResizeDynamic(DataProcessingOperator):
    """
    支持动态 max_pixels 的图像处理器，用于处理输入图像（edit_image）。
    
    类似 OmniGen 的 max_input_pixels 逻辑：
    - 如果 max_pixels 是 list，根据输入图像数量选择对应索引的像素限制
    - 如果 max_pixels 是 int，所有图像使用相同的限制
    
    该处理器直接处理图像路径列表，能感知列表长度来确定动态 max_pixels。
    """
    def __init__(self, height=None, width=None, max_pixels=None, height_division_factor=1, width_division_factor=1, 
                 base_path="", convert_RGB=True, convert_RGBA=False, debug=False):
        self.height = height
        self.width = width
        self.max_pixels = max_pixels  # 可以是 int 或 list
        self.height_division_factor = height_division_factor
        self.width_division_factor = width_division_factor
        self.base_path = base_path
        self.convert_RGB = convert_RGB
        self.convert_RGBA = convert_RGBA
        self.debug = debug
        self._debug_count = 0  # 用于控制调试输出频率

    def _get_max_pixels_for_img_num(self, img_num):
        """根据图像数量获取单张图片的 max_pixels 限制"""
        if isinstance(self.max_pixels, (list, tuple)):
            idx = min(img_num - 1, len(self.max_pixels) - 1)
            idx = max(0, idx)
            return self.max_pixels[idx]
        return self.max_pixels

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
    
    def get_height_width(self, image, max_pixels):
        if self.height is None or self.width is None:
            width, height = image.size
            if max_pixels is not None and width * height > max_pixels:
                scale = (width * height / max_pixels) ** 0.5
                height, width = int(height / scale), int(width / scale)
            height = height // self.height_division_factor * self.height_division_factor
            width = width // self.width_division_factor * self.width_division_factor
        else:
            height, width = self.height, self.width
        return height, width
    
    def process_single_image(self, image_path, max_pixels, img_idx=0, total_imgs=1):
        """处理单张图像"""
        # 转换为绝对路径
        if self.base_path:
            full_path = os.path.join(self.base_path, image_path)
        else:
            full_path = image_path
        
        # 加载图像
        image = Image.open(full_path)
        original_size = image.size  # (width, height)
        if self.convert_RGB:
            image = image.convert("RGB")
        if self.convert_RGBA:
            image = image.convert("RGBA")
        
        # 裁剪和缩放
        target_height, target_width = self.get_height_width(image, max_pixels)
        image = self.crop_and_resize(image, target_height, target_width)
        
        return image
    
    def __call__(self, data):
        """
        处理图像路径或路径列表。
        
        Args:
            data: 可以是字符串（单张图像路径）或字符串列表（多张图像路径）
        
        Returns:
            处理后的图像或图像列表
        """
        if isinstance(data, str):
            # 单张图像
            max_pixels = self._get_max_pixels_for_img_num(1)
            result = self.process_single_image(data, max_pixels, img_idx=0, total_imgs=1)
            return result
        elif isinstance(data, (list, tuple)):
            # 多张图像：根据列表长度确定 max_pixels
            img_num = len(data)
            max_pixels = self._get_max_pixels_for_img_num(img_num)
            results = [self.process_single_image(img_path, max_pixels, img_idx=i, total_imgs=img_num) 
                      for i, img_path in enumerate(data)]
            return results
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")


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


class LoadAudio(DataProcessingOperator):
    def __init__(self, sr=16000):
        self.sr = sr
    def __call__(self, data: str):
        import librosa
        input_audio, sample_rate = librosa.load(data, sr=self.sr)
        return input_audio
