import argparse
import torch
import json
import os
import math
from tqdm import tqdm
import numpy as np
from transformers import AutoConfig
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--video_path", default="", help="Path to the video files.")
    parser.add_argument("--model_path", type=str, default="lmms-lab/llava-onevision-qwen2-7b-ov")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--load_8bit", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--mm_newline_position", type=str, default="no_token")
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser.parse_args()


def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = np.array(load_image(image_file))
        out.append(image)
    return np.stack(out, axis=0)


def load_model(args):
    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    if args.overwrite == True:
        overwrite_config = {}
        overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
        overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
        overwrite_config["mm_newline_position"] = args.mm_newline_position
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, load_8bit=args.load_8bit, overwrite_config=overwrite_config)
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    
    if getattr(model.config, "force_sample", None) is not None:
        args.force_sample = model.config.force_sample
    else:
        args.force_sample = False
        
    if getattr(model.config, "add_time_instruction", None) is not None:
        args.add_time_instruction = model.config.add_time_instruction
    else:
        args.add_time_instruction = False
    return tokenizer, model, image_processor


def run_one_inference(prompt, image_files, tokenizer, model, image_processor, verbose=False):
    video = load_images(image_files) 
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
    video = [video] # [(num_images, 3, 384, 384)]
    
    question = prompt
    qs = question
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = (DEFAULT_IMAGE_TOKEN + "\n") + qs
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    if tokenizer.pad_token_id is None:
        if "qwen" in tokenizer.name_or_path.lower():
            print("Setting pad token to bos token for qwen model.")
            tokenizer.pad_token_id = 151643
                    
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    with torch.inference_mode():
        # deterministic sampling
        output_ids = model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=False, temperature=0.0, max_new_tokens=1024, top_p=0.1, num_beams=1, use_cache=True) 
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if verbose:
        print(outputs)
    return outputs


def main():
    with open(args.data_file, 'r') as f:
        data_list = json.load(f)
    output_file = args.data_file.replace('input', 'output')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f'Reading {len(data_list)} queries from {args.data_file}, saving output to {output_file}')
    
    tokenizer, model, image_processor = load_model(args)
    
    new_data_list = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            new_data_list = json.load(f)
    existing_idx = [d['idx'] for d in new_data_list]
    
    for i, data_dict in enumerate(tqdm(data_list)):
        # if data_dict['idx'] in existing_idx:
            # continue
        image_files = [os.path.join(args.video_path, f) for f in data_dict['image_files']]
        if not all([os.path.exists(f) for f in image_files]):
            print(f"Missing image file {image_files[0]} for query {i}, skipping")
            continue
        print('-' * 20, i, '-' * 20)
        output = run_one_inference(data_dict['query0'], image_files, tokenizer, model, image_processor, verbose=True)
        data_dict['response0'] = output
        new_data_list.append(data_dict)
    with open(output_file, 'w') as f:
        json.dump(new_data_list, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    main()
    