# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# import io  # 旧版本使用，新版本不再需要
# import random  # 未使用
from PIL import Image, ImageFile, PngImagePlugin

from .interleave_t2i_dataset import InterleavedBaseIterableDataset, ParquetStandardIterableDataset
from ..data_utils import pil_img2rgb


Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


# ============================================================================
# 旧版本的UnifiedMultiRefIterableDataset（已废弃，保留用于参考）
# 旧版本从parquet中读取bytes，然后使用io.BytesIO加载图像
# ============================================================================
# class UnifiedMultiRefIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):
#
#     def parse_row(self, row):
#         image_num = len(row["image_list"])
#
#         data = self._init_data()
#         token_len_1 = []
#         for idx in range(image_num - 1):
#             data = self._add_image(
#                 data, 
#                 pil_img2rgb(Image.open(io.BytesIO(row["image_list"][idx]))),
#                 need_loss=False, 
#                 need_vae=True, 
#                 need_vit=True, 
#             )
#             token_len_1.append(data['num_tokens'])
#
#         # Handle both string and PyArrow scalar/numpy array types
#         instruction = row['instruction_list'][0]
#         if isinstance(instruction, str):
#             instruction_text = instruction
#         else:
#             instruction_text = instruction.item()
#         data = self._add_text(data, instruction_text, need_loss=False)
#         token_len_2 = data['num_tokens']
#         
#         data = self._add_image(
#             data, 
#             pil_img2rgb(Image.open(io.BytesIO(row["image_list"][-1]))),
#             need_loss=True, 
#             need_vae=False, 
#             need_vit=False,
#         )
#         token_len_3 = data['num_tokens']
#
#         # print(f"token_len_1: {token_len_1}, token_len_2: {token_len_2}, token_len_3: {token_len_3}")
#         
#         return data
# ============================================================================


# ============================================================================
# 新版本的UnifiedMultiRefIterableDataset
# 新版本从parquet中读取路径字符串，然后根据路径读取图像文件
# ============================================================================
class UnifiedMultiRefIterableDataset(InterleavedBaseIterableDataset, ParquetStandardIterableDataset):

    def parse_row(self, row):
        image_num = len(row["image_list"])
        input_image_num = image_num - 1  # 输入图像数量（不包括输出图像）

        data = self._init_data()
        token_len_1 = []
        
        # 处理前面的输入图像（需要VAE和ViT，不需要loss）
        for idx in range(input_image_num):
            # row["image_list"][idx] 现在是一个路径字符串
            image_path = row["image_list"][idx]
            if isinstance(image_path, str):
                image_path_str = image_path
            else:
                # 如果是PyArrow标量，转换为字符串
                image_path_str = str(image_path)
            
            # 根据路径读取图像
            try:
                image = pil_img2rgb(Image.open(image_path_str))
            except Exception as e:
                print(f"Error loading image from path {image_path_str}: {e}")
                # 如果图像加载失败，返回空数据
                return {}
            
            # 传入 img_num 参数，用于动态 max_pixels 逻辑
            # img_num 是 1-indexed，表示当前是第几张输入图像
            data = self._add_image(
                data, 
                image,
                need_loss=False, 
                need_vae=True, 
                need_vit=True, 
                img_num=input_image_num,  # 使用输入图像总数来决定 max_pixels
            )
            token_len_1.append(data['num_tokens'])

        # Handle both string and PyArrow scalar/numpy array types
        instruction = row['instruction_list'][0]
        if isinstance(instruction, str):
            instruction_text = instruction
        else:
            instruction_text = instruction.item()
        data = self._add_text(data, instruction_text, need_loss=False)
        token_len_2 = data['num_tokens']
        
        # 处理最后的输出图像（需要loss，不需要VAE和ViT）
        output_image_path = row["image_list"][-1]
        if isinstance(output_image_path, str):
            output_image_path_str = output_image_path
        else:
            # 如果是PyArrow标量，转换为字符串
            output_image_path_str = str(output_image_path)
        
        # 根据路径读取图像
        try:
            output_image = pil_img2rgb(Image.open(output_image_path_str))
        except Exception as e:
            print(f"Error loading output image from path {output_image_path_str}: {e}")
            # 如果图像加载失败，返回空数据
            return {}
        
        data = self._add_image(
            data, 
            output_image,
            need_loss=True, 
            need_vae=False, 
            need_vit=False,
        )
        token_len_3 = data['num_tokens']

        # print(f"token_len_1: {token_len_1}, token_len_2: {token_len_2}, token_len_3: {token_len_3}")
        
        return data
