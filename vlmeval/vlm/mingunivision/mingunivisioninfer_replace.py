import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import AutoProcessor

from PIL import Image
import numpy as np
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr # Import PSNR function

from ..base import BaseModel
from modeling_bailingmm import MingUniVisionForConditionalGeneration
import torchvision.transforms as T
import warnings
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    QuantoConfig,
    BitsAndBytesConfig,
)

warnings.filterwarnings("ignore")


def tensor_to_pil(image_tensor):
    mean = torch.Tensor([0.5,0.5,0.5]).view(1,-1,1,1).cuda()
    std = torch.Tensor([0.5,0.5,0.5]).view(1,-1,1,1).cuda()
    image_tensor = (image_tensor*std + mean)[0]
    image_tensor = T.ToPILImage()(image_tensor)
    return image_tensor


class MingUniVisionInfer_REPLACE(BaseModel):
    def __init__(
        self,
        model_name_or_path,
        dtype="bf16",
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.dtype = dtype
        self.model, self.tokenizer, self.processor = self.load_model_processor()
        self.model.tokenizer = self.tokenizer
        self.model.model.tokenizer = self.tokenizer
        

    def load_model_processor(self):
        tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/home/tongyujun/VLMEvalKit/vlmeval/vlm/mingunivision", trust_remote_code=True)
        processor = AutoProcessor.from_pretrained("/root/autodl-tmp/home/tongyujun/VLMEvalKit/vlmeval/vlm/mingunivision", trust_remote_code=True)
        
        if self.dtype == "int4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["BailingAudioModel"]   
            )
            model = MingUniVisionForConditionalGeneration.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                quantization_config=quantization_config,
                trust_remote_code=True,
                device_map="cuda"
            )
        elif self.dtype == "int8":
            quantization_config = QuantoConfig(weights="int8", modules_to_not_convert=["BailingAudioModel"])
            model = MingUniVisionForConditionalGeneration.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                quantization_config=quantization_config,
                trust_remote_code=True,
                device_map="cuda"
            )
        else:
            model = MingUniVisionForConditionalGeneration.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
                device_map="cuda"
            )

        return model, tokenizer, processor

    def generate_inner_inner(self, messages, max_new_tokens=512, output_image_prefix="output", for_edit=False):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, use_system=True
        )

        image_inputs, _, _ = self.processor.process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            return_tensors="pt",
            image_patch_size=self.model.vision.patch_size,
            for_edit=for_edit,
        ).to(self.model.device)

        for k in inputs.keys():
            if k == "pixel_values":
                inputs[k] = inputs[k].to(dtype=torch.bfloat16)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                output_image_prefix=output_image_prefix,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text
    
    def reset_inner_state(self):
        self.model.reset_inner_state()
    

    def modify_path(self, original_path, dataset_name):
        # 先找到 'LMUData/images/' 后面的部分
        prefix = "/root/autodl-tmp/home/tongyujun/LMUData/images/"
        
        # 检查路径是否以这个前缀开始
        if original_path.startswith(prefix):
            # 取得 'LMUData/images/' 后的路径
            suffix_path = original_path[len(prefix):]
            
            # 将原来的 dataset 名字替换成新的
            new_suffix_path = suffix_path.replace(suffix_path.split('/')[0], f"{dataset_name}_Ming_Refined", 1)
            
            # 拼接新的路径
            new_path = os.path.join(prefix, new_suffix_path)
            return new_path
        return original_path

    def pad_to_square(self, image):
        width, height = image.size
        max_dim = max(width, height)
        
        # Create a new square image with the larger dimension as both width and height
        new_image = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))  # Black padding
        new_image.paste(image, ((max_dim - width) // 2, (max_dim - height) // 2))
        
        return new_image

    def calculate_mse(self, image_path1, image_path2):
        # Open the images
        image1 = Image.open(image_path1).convert('RGB')
        image2 = Image.open(image_path2).convert('RGB')

        # Pad image1 to make it square
        if image1.size[0] != image1.size[1]:
            image1 = self.pad_to_square(image1)

        # Ensure both images are the same size (image2 is already square)
        if image1.size != image2.size:
            image2 = image2.resize(image1.size)

        # Convert images to numpy arrays
        img1_array = np.array(image1)
        img2_array = np.array(image2)

        # Calculate the MSE
        return mse(img1_array, img2_array)


    def calculate_psnr(self, image_path1, image_path2):
        """Calculates the PSNR between two images."""
        # Open the images
        image1 = Image.open(image_path1).convert('RGB')
        image2 = Image.open(image_path2).convert('RGB')

        # Pad image1 to make it square
        if image1.size[0] != image1.size[1]:
            image1 = self.pad_to_square(image1)

        # Ensure both images are the same size (image2 is already square)
        if image1.size != image2.size:
            image2 = image2.resize(image1.size)

        # Convert images to numpy arrays
        img1_array = np.array(image1)
        img2_array = np.array(image2)

        # Calculate the PSNR. data_range is 255 for 8-bit RGB images.
        return psnr(img1_array, img2_array, data_range=255)

        
    def generate_inner(self, message, dataset=None):
        self.reset_inner_state()

       
        # print(message)
        converted_data = []

        for item in message:
            if item['type'] == 'image':
                enhanced_image_path = self.modify_path(item['value'], dataset)
                if not os.path.exists(enhanced_image_path):
                    enhanced_messages = [{
                        "role": "HUMAN",
                        "content": [
                            {"type": "image", "image": item['value']},
                            {"type": "text", "text": "Enhance the picture."},
                        ],
                    }]
                    output_text = self.generate_inner_inner(enhanced_messages, max_new_tokens=512, for_edit=True, output_image_prefix=enhanced_image_path)
                    self.reset_inner_state()
                
                original_image_path = item['value']  # Assuming 'value' is the path to the original image
                #  # Calculate MSE between the original image and enhanced image
                # mse_value = self.calculate_mse(original_image_path, enhanced_image_path)

                # # If the enhanced image has lower MSE (better quality), replace it
                # print(mse_value)

                # Calculate PSNR between the original image and the enhanced image
                psnr_value = self.calculate_psnr(original_image_path, enhanced_image_path)

                # If PSNR is high (better quality), use the enhanced image.
                # A common threshold for good quality is around 30-35 dB.
                print(f"PSNR: {psnr_value:.2f} dB")
                if psnr_value > 25:
                # if mse_value <500:
                    converted_data.append({"type": "image", "image": enhanced_image_path})
                else:
                    converted_data.append({"type": "image", "image": original_image_path})
            elif item['type'] == 'text':
                converted_data.append({'type': 'text', 'text': item['value']})
        messages = [{
            "role": "HUMAN",
            "content": converted_data,
            }]
        print(messages)
        output_text = self.generate_inner_inner(messages, max_new_tokens=512, for_edit=False)
        return output_text


