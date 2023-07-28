from diffusers import StableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from diffusers import AutoencoderKL
from PIL import Image
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

vae = AutoencoderKL.from_pretrained(
    "stabilityai/sd-vae-ft-mse",
    torch_dtype=torch.float16,
).to("cuda")
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
).to("cuda")
pipe.enable_attention_slicing(1)
pipe.enable_xformers_memory_efficient_attention()
pipe.vae = vae
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

generator = torch.Generator("cuda").manual_seed(0)

prompt = "An image of a squirrel in Picasso style"


def get_inputs(batch_size=1):
    generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
    prompts = batch_size * [prompt]
    num_inference_steps = 50
    width = 1024
    height = 1024

    return {
        "prompt": prompts,
        "generator": generator,
        "num_inference_steps": num_inference_steps,
        "width": width,
        "height": height,
    }


def image_grid(imgs, rows=2, cols=2):
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    grid.save("test.png")


images = pipe(**get_inputs(batch_size=4)).images
image_grid(images)
