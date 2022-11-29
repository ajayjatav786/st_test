import inspect
import warnings
from typing import List, Optional, Union
import torch
import io
from torch import autocast
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import torch
import uuid
import boto3
import os

from diffusers import StableDiffusionImg2ImgPipeline, DiffusionPipeline, DPMSolverMultistepScheduler
from common.enhancement.codeformer import CodeFormerEnhancer
# torch.cuda.empty_cache()
negative_prompt = "smile, mask, painting, cartoon, green, watermark, statue, cartoon"
preamble = "Symmetry!! A high detailed headshot portrait photo of "
postamble = " looking at the camera, Piercing gaze, eos-1d, f/1.4, iso 200, 1/160s, 8k, raw, symmetrical balance, hasselblad camera, 5 0 mm, sharp focus, by brandon stanton"  # , award winning photo"
face = Image.open('visual-bias/face-avg-0.png')
torso = Image.open('visual-bias/new_torso.png')
torso_dilated = Image.open('visual-bias/torso_dilated.png')
torso.paste(face, (0, 0), face)

device = "cuda"
CODEFORMER_MODELS = 'CodeFormer'
bucket_name = 'character_image_output'

HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
aws_secret = os.getenv("aws_id")
aws_id = os.getenv("aws_secret")

client = boto3.client('s3', region_name="us-east-2", aws_access_key_id=aws_id, aws_secret_access_key=aws_secret,
                      config=boto3.session.Config(signature_version='s3v4', ))


def init_models():
    global pipe, inpaint_pipe, generator, codeformer

    scheduler = DPMSolverMultistepScheduler.from_pretrained("AletheaAI/epoch000007v3", subfolder="scheduler",
                                                            use_auth_token=HF_AUTH_TOKEN)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained("AletheaAI/epoch000007v3", use_auth_token=HF_AUTH_TOKEN,
                                                          scheduler=scheduler).to(device)
    pipe.safety_checker = lambda images, clip_input: (images, False)

    inpaint_pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", dtype=torch.float16,
                                                     revision="fp16", use_auth_token=HF_AUTH_TOKEN,
                                                     scheduler=scheduler).to(device)
    inpaint_pipe.safety_checker = lambda images, clip_input: (images, False)
    generator = torch.Generator(device=device)

    codeformer = CodeFormerEnhancer(base_model_path=CODEFORMER_MODELS)

# torch.cuda.empty_cache()
def inference(name, attire):
    request_id = uuid.uuid4().hex[:7]
    prompt = preamble + name + postamble
    prompt = prompt.replace('green', 'red')
    with autocast("cuda"):
        image = pipe_new(prompt=prompt, negative_prompt=negative_prompt, init_image=torso.convert('RGB'), strength=0.9,
                         guidance_scale=8, num_inference_steps=(15 if attire else 65), generator=generator).images[0]

        if attire:
            attire = attire.replace('green', 'red')
            bodywear = f'Symmetry!! Front view of standard {attire} uniform, trending on artstation, eos-1d, f/1.4, iso 200, 1/160s, symmetrical balance, in-frame, epic composition, hasselblad camera, 50 mm'
            if any(word in attire for word in ['nude', 'naked']):
                bodywear = f'Front view of beautiful {attire}'
            image = inpaint_pipe(prompt=bodywear, negative_prompt='green, watermark',
                                 image=image, mask_image=torso_dilated, guidance_scale=9,
                                 strength=0, num_inference_steps=10).images[0]

    object_key = f'{request_id}.jpg'  #file_name_in_s3
    in_mem_file = io.BytesIO()
    pil_image.save(in_mem_file, format=image.format)
    in_mem_file.seek(0)
    client.upload_fileobj(in_mem_file, Bucket=bucket_name, Key=object_key)
    print("File uploaded successfully")

    response = client.generate_presigned_url('get_object',
                                             Params={'Bucket': bucket_name,
                                                     'Key': object_key},
                                             ExpiresIn=3600
                                             )
    return {"url": response, "result_id": request_id}
