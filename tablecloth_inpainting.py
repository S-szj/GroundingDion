import torch
from PIL import Image
from segment_anything import build_sam, SamPredictor 
from GroundingDINO.groundingdino.util import box_ops
import numpy as np
from diffusers import StableDiffusionPipeline

torch.cuda.set_device(1)
device = torch.device("cuda")

image_src = './test_img/input_img/table_cloth/Leather material, dark blue, desktop.jpg'

image_original = Image.open(image_src)
pipe = StableDiffusionPipeline.from_pretrained(
    "./stable-diffusion-v1-5",
    torch_dtype=torch.float32,
).to("cuda")

image_add = pipe(prompt="Reinforced metal material", 
                 negative_prompt="",
                 image=image_original, 
                 generator = torch.Generator("cuda").manual_seed(1),
                 strength=0.75,
                 guidance_scale=7.5,
                 num_inference_steps=100,
                ).images[0].resize(image_original.size)
image_add.save("./test_img/output_img/add_image.jpg")
