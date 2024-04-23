import base64
import io
import os
import uuid
from io import BytesIO

from PIL import Image
from fastapi import FastAPI, Body
import json

from huggingface_hub import hf_hub_download, HfFileSystem, ModelCard, snapshot_download
from safetensors.torch import load_file
from pipeline_stable_diffusion_xl_instantid_img2img import StableDiffusionXLInstantIDImg2ImgPipeline, draw_kps
from controlnet_aux import ZoeDetector

import torch

torch.jit.script = lambda f: f
import timm
import time
from safetensors.torch import load_file
from cog_sdxl_dataset_and_utils import TokenEmbeddingsHandler

import lora
import copy
import json
import gc
import random
from urllib.parse import quote
import gdown
import os
import re
import requests

import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
import cv2
import torch
import numpy as np
from PIL import Image

from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid_img2img import StableDiffusionXLInstantIDImg2ImgPipeline, draw_kps
from controlnet_aux import ZoeDetector

from compel import Compel, ReturnedEmbeddingsType




import torch

torch.jit.script = lambda f: f
import timm
import time
from huggingface_hub import hf_hub_download, HfFileSystem, ModelCard, snapshot_download
from safetensors.torch import load_file
from share_btn import community_icon_html, loading_icon_html, share_js
from cog_sdxl_dataset_and_utils import TokenEmbeddingsHandler

import lora
import copy
import json
import gc
import random
from urllib.parse import quote
import gdown
import os
import re
import requests

import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
import cv2
import torch
import numpy as np
from PIL import Image

from pipeline_stable_diffusion_xl_instantid_img2img import StableDiffusionXLInstantIDImg2ImgPipeline, draw_kps
from controlnet_aux import ZoeDetector

from compel import Compel, ReturnedEmbeddingsType
import spaces

app = FastAPI()

with open("sdxl_loras.json", "r") as file:
    data = json.load(file)
    sdxl_loras_raw = [
        {
            "image": item["image"],
            "title": item["title"],
            "repo": item["repo"],
            "trigger_word": item["trigger_word"],
            "weights": item["weights"],
            "is_compatible": item["is_compatible"],
            "is_pivotal": item.get("is_pivotal", False),
            "text_embedding_weights": item.get("text_embedding_weights", None),
            "likes": item.get("likes", 0),
            "downloads": item.get("downloads", 0),
            "is_nc": item.get("is_nc", False),
            "new": item.get("new", False),
        }
        for item in data
    ]

sdxl_loras = sdxl_loras_raw
with open("defaults_data.json", "r") as file:
    lora_defaults = json.load(file)

device = "cuda"

state_dicts = {}

for item in sdxl_loras_raw:
    saved_name = hf_hub_download(item["repo"], item["weights"])

    if not saved_name.endswith('.safetensors'):
        state_dict = torch.load(saved_name)
    else:
        state_dict = load_file(saved_name)

    state_dicts[item["repo"]] = {
        "saved_name": saved_name,
        "state_dict": state_dict
    }

sdxl_loras_raw = [item for item in sdxl_loras_raw if item.get("new") != True]

# download models
hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ControlNetModel/config.json",
    local_dir="/data/checkpoints",
)
hf_hub_download(
    repo_id="InstantX/InstantID",
    filename="ControlNetModel/diffusion_pytorch_model.safetensors",
    local_dir="/data/checkpoints",
)
hf_hub_download(
    repo_id="InstantX/InstantID", filename="ip-adapter.bin", local_dir="/data/checkpoints"
)
hf_hub_download(
    repo_id="latent-consistency/lcm-lora-sdxl",
    filename="pytorch_lora_weights.safetensors",
    local_dir="/data/checkpoints",
)
# download antelopev2
# if not os.path.exists("/data/antelopev2.zip"):
#    gdown.download(url="https://drive.google.com/file/d/18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8/view?usp=sharing", output="/data/", quiet=False, fuzzy=True)
#    os.system("unzip /data/antelopev2.zip -d /data/models/")

antelope_download = snapshot_download(repo_id="DIAMONIK7777/antelopev2", local_dir="/data/models/antelopev2")
print(antelope_download)
face_analysis_app = FaceAnalysis(name='antelopev2', root='/data', providers=['CPUExecutionProvider'])
face_analysis_app.prepare(ctx_id=0, det_size=(640, 640))

# prepare models under ./checkpoints
face_adapter = f'/data/checkpoints/ip-adapter.bin'
controlnet_path = f'/data/checkpoints/ControlNetModel'

# load IdentityNet
st = time.time()
identitynet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
zoedepthnet = ControlNetModel.from_pretrained("diffusers/controlnet-zoe-depth-sdxl-1.0", torch_dtype=torch.float16)
et = time.time()
elapsed_time = et - st
print('Loading ControlNet took: ', elapsed_time, 'seconds')
st = time.time()
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
et = time.time()
elapsed_time = et - st
print('Loading VAE took: ', elapsed_time, 'seconds')
st = time.time()
pipe = StableDiffusionXLInstantIDImg2ImgPipeline.from_pretrained("rubbrband/albedobaseXL_v21",
                                                                 vae=vae,
                                                                 controlnet=[identitynet, zoedepthnet],
                                                                 torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
pipe.load_ip_adapter_instantid(face_adapter)
pipe.set_ip_adapter_scale(0.8)
et = time.time()
elapsed_time = et - st
print('Loading pipeline took: ', elapsed_time, 'seconds')
st = time.time()
compel = Compel(tokenizer=[pipe.tokenizer, pipe.tokenizer_2], text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True])
et = time.time()
elapsed_time = et - st
print('Loading Compel took: ', elapsed_time, 'seconds')

st = time.time()
zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")
et = time.time()
elapsed_time = et - st
print('Loading Zoe took: ', elapsed_time, 'seconds')
zoe.to(device)
pipe.to(device)

last_lora = ""
last_fused = False
js = '''
var button = document.getElementById('button');
// Add a click event listener to the button
button.addEventListener('click', function() {
    element.classList.add('selected');
});
'''
lora_archive = "/data"



def center_crop_image_as_square(img):
    square_size = min(img.size)

    left = (img.width - square_size) / 2
    top = (img.height - square_size) / 2
    right = (img.width + square_size) / 2
    bottom = (img.height + square_size) / 2

    img_cropped = img.crop((left, top, right, bottom))
    return img_cropped



def merge_incompatible_lora(full_path_lora, lora_scale):
    for weights_file in [full_path_lora]:
        if ";" in weights_file:
            weights_file, multiplier = weights_file.split(";")
            multiplier = float(multiplier)
        else:
            multiplier = lora_scale

        lora_model, weights_sd = lora.create_network_from_weights(
            multiplier,
            full_path_lora,
            pipe.vae,
            pipe.text_encoder,
            pipe.unet,
            for_inference=True,
        )
        lora_model.merge_to(
            pipe.text_encoder, pipe.unet, weights_sd, torch.float16, "cuda"
        )
        del weights_sd
        del lora_model


@spaces.GPU
def generate_image(prompt, negative, face_emb, face_image, face_kps, image_strength, guidance_scale, face_strength,
                   depth_control_scale, repo_name, loaded_state_dict, lora_scale, sdxl_loras, selected_state_index, st):
    print(loaded_state_dict)
    et = time.time()
    elapsed_time = et - st
    print('Getting into the decorated function took: ', elapsed_time, 'seconds')
    global last_fused, last_lora
    print("Last LoRA: ", last_lora)
    print("Current LoRA: ", repo_name)
    print("Last fused: ", last_fused)
    # prepare face zoe
    st = time.time()
    with torch.no_grad():
        image_zoe = zoe(face_image)
    width, height = face_kps.size
    images = [face_kps, image_zoe.resize((height, width))]
    et = time.time()
    elapsed_time = et - st
    print('Zoe Depth calculations took: ', elapsed_time, 'seconds')
    if last_lora != repo_name:
        if (last_fused):
            st = time.time()
            pipe.unfuse_lora()
            pipe.unload_lora_weights()
            pipe.unload_textual_inversion()
            et = time.time()
            elapsed_time = et - st
            print('Unfuse and unload LoRA took: ', elapsed_time, 'seconds')
        st = time.time()
        pipe.load_lora_weights(loaded_state_dict)
        pipe.fuse_lora(lora_scale)
        et = time.time()
        elapsed_time = et - st
        print('Fuse and load LoRA took: ', elapsed_time, 'seconds')
        last_fused = True
        is_pivotal = sdxl_loras[selected_state_index]["is_pivotal"]
        if (is_pivotal):
            # Add the textual inversion embeddings from pivotal tuning models
            text_embedding_name = sdxl_loras[selected_state_index]["text_embedding_weights"]
            embedding_path = hf_hub_download(repo_id=repo_name, filename=text_embedding_name, repo_type="model")
            state_dict_embedding = load_file(embedding_path)
            pipe.load_textual_inversion(
                state_dict_embedding["clip_l" if "clip_l" in state_dict_embedding else "text_encoders_0"],
                token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
            pipe.load_textual_inversion(
                state_dict_embedding["clip_g" if "clip_g" in state_dict_embedding else "text_encoders_1"],
                token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)

    print("Processing prompt...")
    st = time.time()
    conditioning, pooled = compel(prompt)
    if (negative):
        negative_conditioning, negative_pooled = compel(negative)
    else:
        negative_conditioning, negative_pooled = None, None
    et = time.time()
    elapsed_time = et - st
    print('Prompt processing took: ', elapsed_time, 'seconds')
    print("Processing image...")
    st = time.time()
    image = pipe(
        prompt_embeds=conditioning,
        pooled_prompt_embeds=pooled,
        negative_prompt_embeds=negative_conditioning,
        negative_pooled_prompt_embeds=negative_pooled,
        width=width,
        height=height,
        image_embeds=face_emb,
        image=face_image,
        strength=1 - image_strength,
        control_image=images,
        num_inference_steps=20,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=[face_strength, depth_control_scale],
    ).images[0]
    et = time.time()
    elapsed_time = et - st
    print('Image processing took: ', elapsed_time, 'seconds')
    last_lora = repo_name
    return image


def run_lora(face_image,  selected_state_index):
    prompt = ""
    negative = ""
    lora_scale = 0.9
    face_strength = 0.85
    image_strength = 0.5
    guidance_scale = 7
    depth_control_scale = 0.8
    #custom_lora_path = custom_lora[0] if custom_lora else None
    #selected_state_index = selected_state.index if selected_state else -1
    st = time.time()
    face_image = center_crop_image_as_square(face_image)
    try:
        face_info = face_analysis_app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * x['bbox'][3] - x['bbox'][1])[
            -1]  # only use the maximum face
        face_emb = face_info['embedding']
        face_kps = draw_kps(face_image, face_info['kps'])
    except:
        raise HTTPException(status_code=500, detail="No face found in your image. Only face images work here. Try again")
        #raise ValueError("No face found in your image. Only face images work here. Try again")
    et = time.time()
    elapsed_time = et - st
    print('Cropping and calculating face embeds took: ', elapsed_time, 'seconds')

    st = time.time()


    for lora_list in lora_defaults:
        if lora_list["model"] == sdxl_loras[selected_state_index]["repo"]:
            prompt_full = lora_list.get("prompt", None)
            lora_scale = lora_list["weight"]
            face_strength = lora_list["face_strength"]
            image_strength = lora_list["image_strength"]
            guidance_scale = lora_list["guidance_scale"]
            depth_control_scale = lora_list["depth_control_scale"]
            if (prompt_full):
                prompt = prompt_full.replace("<subject>", prompt)

    print("Prompt:", prompt)
    if (prompt == ""):
        prompt = "a person"

    # print("Selected State: ", selected_state_index)
    # print(sdxl_loras[selected_state_index]["repo"])
    if negative == "":
        negative = None

    repo_name = sdxl_loras[selected_state_index]["repo"]
    weight_name = sdxl_loras[selected_state_index]["weights"]
    full_path_lora = state_dicts[repo_name]["saved_name"]
    print("Full path LoRA ", full_path_lora)
    # loaded_state_dict = copy.deepcopy(state_dicts[repo_name]["state_dict"])
    cross_attention_kwargs = None
    et = time.time()
    elapsed_time = et - st
    print('Small content processing took: ', elapsed_time, 'seconds')

    st = time.time()
    image = generate_image(prompt, negative, face_emb, face_image, face_kps, image_strength, guidance_scale,
                           face_strength, depth_control_scale, repo_name, full_path_lora, lora_scale, sdxl_loras,
                           selected_state_index, st)
    return image


def shuffle_gallery(sdxl_loras):
    random.shuffle(sdxl_loras)
    return [(item["image"], item["title"]) for item in sdxl_loras], sdxl_loras


def classify_gallery(sdxl_loras):
    sorted_gallery = sorted(sdxl_loras, key=lambda x: x.get("likes", 0), reverse=True)
    return [(item["image"], item["title"]) for item in sorted_gallery], sorted_gallery


def swap_gallery(order, sdxl_loras):
    if (order == "random"):
        return shuffle_gallery(sdxl_loras)
    else:
        return classify_gallery(sdxl_loras)









@app.post("/filters")
async def filter_photo(
        input_image: str = Body("", title=' input image'),
        selected_state_index: int = Body(0, title='LoRA Model Index')
):
    # print(input_image)
    im = Image.open(BytesIO(base64.b64decode(input_image)))
    output_img_file = run_lora(im, selected_state_index)
    # print(output_img_file)
    # image = Image.open(output_img_file)
    # image = image.resize(im.size)
    with io.BytesIO() as output_bytes:
        output_img_file.save(output_bytes, format="JPEG")
        img_str = base64.b64encode(output_bytes.getvalue()).decode("utf-8")
    return {"image": img_str}
