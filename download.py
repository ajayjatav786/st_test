# In this file, we define download_model
# It runs during container build time to get model weights built into the container

from diffusers import StableDiffusionImg2ImgPipeline, DiffusionPipeline, DPMSolverMultistepScheduler
from common.enhancement.codeformer import CodeFormerEnhancer
import os

device = 'cuda'
HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
print(HF_AUTH_TOKEN)

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    scheduler = DPMSolverMultistepScheduler.from_pretrained("AletheaAI/epoch000007v3", subfolder="scheduler", use_auth_token=HF_AUTH_TOKEN)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained("AletheaAI/epoch000007v3",
                                                          use_auth_token=HF_AUTH_TOKEN, scheduler=scheduler).to(device)
    inpaint_pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", dtype=torch.float16,
                                                     revision="fp16", use_auth_token=HF_AUTH_TOKEN, scheduler=scheduler).to(device)



if __name__ == "__main__":
    download_model()
