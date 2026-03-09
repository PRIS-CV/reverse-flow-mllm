import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import AutoProcessor

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


class MingUniVisionInfer(BaseModel):
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
        texts = []
        for message in messages:
            text = self.processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True, use_system=True
            )
            texts.append(text)
        

        image_inputs, _, _ = self.processor.process_vision_info(messages)

        inputs = self.processor(
            text=texts,
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

    def generate_inner(self, message, dataset=None):
        self.reset_inner_state()
        # if dataset is not None:
        #     kwargs = self.adjust_kwargs(dataset)
        # else:
        #     kwargs = self.kwargs
        # prompt = ''
        # for s in message:
        #     if s['type'] == 'image':
        #         prompt += f'<img>{s["value"]}</img>'
        #     elif s['type'] == 'text':
        #         prompt += s['value']
        # if dataset is not None and DATASET_TYPE(dataset) == 'VQA':
        #     prompt += ' Answer:'
        # encoded = self.tokenizer([prompt], return_tensors='pt', padding='longest')
        # input_ids = encoded.input_ids.to('cuda')
        # attention_mask = encoded.attention_mask.to('cuda')

        # pred = self.model.generate(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     **kwargs)
        # answer = self.tokenizer.decode(pred[0][input_ids.size(1):].cpu(), skip_special_tokens=True).strip()
        # return answer
        print(message)
        converted_data = []

        for item in message:
            if item['type'] == 'image':
                converted_data.append({'type': 'image', 'image': item['value']})
            elif item['type'] == 'text':
                converted_data.append({'type': 'text', 'text': item['value']})
        messages = [[{
            "role": "HUMAN",
            "content": converted_data,
            }]]
        output_text = self.generate_inner_inner(messages, max_new_tokens=512, for_edit=False)
        print(output_text)
        return output_text
