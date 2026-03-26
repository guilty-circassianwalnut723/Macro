from .operators import *
import torch, json, pandas
import glob as glob_module


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path=None, metadata_path=None,
        repeat=1,
        data_file_keys=tuple(),
        main_data_operator=lambda x: x,
        special_operator_map=None,
        max_data_items=None,
        max_edit_images=None,
    ):
        self.base_path = base_path
        self.metadata_path = metadata_path
        self.repeat = repeat
        self.data_file_keys = data_file_keys
        self.main_data_operator = main_data_operator
        self.cached_data_operator = LoadTorchPickle()
        self.special_operator_map = {} if special_operator_map is None else special_operator_map
        self.max_data_items = max_data_items
        self.max_edit_images = max_edit_images
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
                (("gif",), LoadGIF(
                    num_frames, time_division_factor, time_division_remainder,
                    frame_processor=ImageCropAndResize(height, width, max_pixels, height_division_factor, width_division_factor),
                )),
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
        elif os.path.isdir(metadata_path):
            # 目录模式：加载目录下所有 .jsonl 文件
            print(f"Loading metadata from directory: {metadata_path}")
            metadata = []
            jsonl_files = sorted(glob_module.glob(os.path.join(metadata_path, "*.jsonl")))
            if not jsonl_files:
                # 如果没有 .jsonl 文件，尝试加载 .json 文件
                jsonl_files = sorted(glob_module.glob(os.path.join(metadata_path, "*.json")))
            
            for jsonl_file in jsonl_files:
                print(f"  Loading: {os.path.basename(jsonl_file)}")
                if jsonl_file.endswith(".jsonl"):
                    with open(jsonl_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:  # 跳过空行
                                metadata.append(json.loads(line))
                elif jsonl_file.endswith(".json"):
                    with open(jsonl_file, "r") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            metadata.extend(data)
                        else:
                            metadata.append(data)
            
            print(f"Total {len(metadata)} items loaded from {len(jsonl_files)} files.")
            self.data = metadata
        elif metadata_path.endswith(".json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.data = metadata
        elif metadata_path.endswith(".jsonl"):
            metadata = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:  # 跳过空行
                        metadata.append(json.loads(line))
            self.data = metadata
        else:
            metadata = pandas.read_csv(metadata_path)
            self.data = [metadata.iloc[i].to_dict() for i in range(len(metadata))]

    def __getitem__(self, data_id):
        # 尝试加载数据，如果失败则跳过并尝试下一条
        max_retries = 10  # 最多尝试10条数据
        original_data_id = data_id
        
        for retry in range(max_retries):
            try:
                if self.load_from_cache:
                    data = self.cached_data[data_id % len(self.cached_data)]
                    data = self.cached_data_operator(data)
                    # 对于缓存数据，edit_image 可能已经被处理成图像对象，检查列表长度
                    if self.max_edit_images is not None and "edit_image" in data:
                        edit_image = data.get("edit_image")
                        if edit_image is not None:
                            if isinstance(edit_image, (list, tuple)):
                                edit_image_count = len(edit_image)
                            else:
                                edit_image_count = 1
                            
                            if edit_image_count > self.max_edit_images:
                                # 跳过这条数据，尝试下一条
                                print(f"[Dataset] 跳过数据 {data_id}: edit_image 数量 ({edit_image_count}) 超过限制 ({self.max_edit_images})")
                                data_id = (data_id + 1) % len(self.cached_data)
                                continue
                    
                    # 输出图像尺寸和文本长度信息（缓存数据）
                    if not hasattr(self, '_debug_output_count'):
                        self._debug_output_count = 0
                    if self._debug_output_count < 50:  # 只输出前50个样本
                        info_parts = []
                        
                        # 输出图像尺寸
                        if "image" in data and data["image"] is not None:
                            img = data["image"]
                            if isinstance(img, (list, tuple)):
                                for i, im in enumerate(img):
                                    if hasattr(im, 'size'):
                                        info_parts.append(f"  img[{i}]: {im.size[0]}x{im.size[1]} ({im.size[0]*im.size[1]} pixels)")
                            elif hasattr(img, 'size'):
                                info_parts.append(f"output_image: {img.size[0]}x{img.size[1]} ({img.size[0]*img.size[1]} pixels)")
                        
                        # 输出 edit_image 尺寸
                        if "edit_image" in data and data["edit_image"] is not None:
                            edit_img = data["edit_image"]
                            if isinstance(edit_img, (list, tuple)):
                                info_parts.append(f"edit_image: {len(edit_img)} images")
                                total_pixels = 0
                                for i, im in enumerate(edit_img):
                                    if hasattr(im, 'size'):
                                        pixels = im.size[0] * im.size[1]
                                        total_pixels += pixels
                                        info_parts.append(f"  edit_img[{i}]: {im.size[0]}x{im.size[1]} ({pixels} pixels)")
                                info_parts.append(f"  total_edit_pixels: {total_pixels}")
                            elif hasattr(edit_img, 'size'):
                                pixels = edit_img.size[0] * edit_img.size[1]
                                info_parts.append(f"edit_image: {edit_img.size[0]}x{edit_img.size[1]} ({pixels} pixels)")
                        
                        # 输出文本长度
                        if "prompt" in data and data["prompt"] is not None:
                            prompt = data["prompt"]
                            if isinstance(prompt, str):
                                prompt_len = len(prompt)
                                info_parts.append(f"prompt_length: {prompt_len} chars")
                            elif isinstance(prompt, (list, tuple)):
                                prompt_len = sum(len(p) if isinstance(p, str) else 0 for p in prompt)
                                info_parts.append(f"prompt_length: {prompt_len} chars ({len(prompt)} items)")
                        
                        if info_parts:
                            print(f"[Dataset] data_id={data_id} (cached): " + ", ".join(info_parts))
                        self._debug_output_count += 1
                else:
                    data = self.data[data_id % len(self.data)].copy()
                    
                    # 检查 edit_image 数量限制（在数据操作之前检查，避免不必要的处理）
                    # 此时 edit_image 还是原始数据（字符串或字符串列表）
                    if self.max_edit_images is not None and "edit_image" in data:
                        edit_image = data.get("edit_image")
                        if edit_image is not None:
                            # 如果是列表，检查长度；如果是字符串，算作1张
                            if isinstance(edit_image, (list, tuple)):
                                edit_image_count = len(edit_image)
                            else:
                                edit_image_count = 1
                            
                            if edit_image_count > self.max_edit_images:
                                # 跳过这条数据，尝试下一条
                                print(f"[Dataset] 跳过数据 {data_id}: edit_image 数量 ({edit_image_count}) 超过限制 ({self.max_edit_images})")
                                data_id = (data_id + 1) % len(self.data)
                                continue

                    # 处理数据操作
                    for key in self.data_file_keys:
                        if key in data:
                            # 跳过 None 值，不进行数据操作（用于 T2I 数据的 edit_image=null）
                            if data[key] is None:
                                continue
                            if key in self.special_operator_map:
                                data[key] = self.special_operator_map[key](data[key])
                            elif key in self.data_file_keys:
                                data[key] = self.main_data_operator(data[key])
                
                # 输出图像尺寸和文本长度信息（限制输出频率）
                if not hasattr(self, '_debug_output_count'):
                    self._debug_output_count = 0
                if self._debug_output_count < 50:  # 只输出前50个样本
                    info_parts = []
                    
                    # 输出图像尺寸
                    if "image" in data and data["image"] is not None:
                        img = data["image"]
                        if isinstance(img, (list, tuple)):
                            img_info = f"output_image: {len(img)} images"
                            for i, im in enumerate(img):
                                if hasattr(im, 'size'):
                                    info_parts.append(f"  img[{i}]: {im.size[0]}x{im.size[1]} ({im.size[0]*im.size[1]} pixels)")
                        elif hasattr(img, 'size'):
                            info_parts.append(f"output_image: {img.size[0]}x{img.size[1]} ({img.size[0]*img.size[1]} pixels)")
                    
                    # 输出 edit_image 尺寸
                    if "edit_image" in data and data["edit_image"] is not None:
                        edit_img = data["edit_image"]
                        if isinstance(edit_img, (list, tuple)):
                            info_parts.append(f"edit_image: {len(edit_img)} images")
                            total_pixels = 0
                            for i, im in enumerate(edit_img):
                                if hasattr(im, 'size'):
                                    pixels = im.size[0] * im.size[1]
                                    total_pixels += pixels
                                    info_parts.append(f"  edit_img[{i}]: {im.size[0]}x{im.size[1]} ({pixels} pixels)")
                            info_parts.append(f"  total_edit_pixels: {total_pixels}")
                        elif hasattr(edit_img, 'size'):
                            pixels = edit_img.size[0] * edit_img.size[1]
                            info_parts.append(f"edit_image: {edit_img.size[0]}x{edit_img.size[1]} ({pixels} pixels)")
                    
                    # 输出文本长度
                    if "prompt" in data and data["prompt"] is not None:
                        prompt = data["prompt"]
                        if isinstance(prompt, str):
                            prompt_len = len(prompt)
                            info_parts.append(f"prompt_length: {prompt_len} chars")
                        elif isinstance(prompt, (list, tuple)):
                            prompt_len = sum(len(p) if isinstance(p, str) else 0 for p in prompt)
                            info_parts.append(f"prompt_length: {prompt_len} chars ({len(prompt)} items)")
                    
                    if info_parts:
                        print(f"[Dataset] data_id={data_id}: " + ", ".join(info_parts))
                    self._debug_output_count += 1
                
                return data
            except Exception as e:
                # 记录错误并尝试下一条数据
                if retry == 0:
                    # 只在第一次打印详细错误
                    print(f"[Dataset] 加载数据 {data_id} 失败: {type(e).__name__}: {str(e)[:100]}, 跳过...")
                data_id = (data_id + 1) % len(self.data) if not self.load_from_cache else (data_id + 1) % len(self.cached_data)
        
        # 如果所有重试都失败，抛出异常
        raise RuntimeError(f"连续 {max_retries} 条数据加载失败，从 data_id={original_data_id} 开始")

    def __len__(self):
        if self.max_data_items is not None:
            return self.max_data_items
        elif self.load_from_cache:
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
