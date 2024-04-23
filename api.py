import base64
import io
import json
import requests
from PIL import Image
from fastapi import FastAPI, Body
from modules.api import api
import gradio as gr

import rembg


# models = [
#     "None",
#     "u2net",
#     "u2netp",
#     "u2net_human_seg",
#     "u2net_cloth_seg",
#     "silueta",
# ]


def rembg_api(_: gr.Blocks, app: FastAPI):
    @app.post("/rembg")
    async def rembg_remove(
            input_image: str = Body("", title='rembg input image'),
            model: str = Body("u2net", title='rembg model'),
            return_mask: bool = Body(False, title='return mask'),
            alpha_matting: bool = Body(False, title='alpha matting'),
            alpha_matting_foreground_threshold: int = Body(240, title='alpha matting foreground threshold'),
            alpha_matting_background_threshold: int = Body(10, title='alpha matting background threshold'),
            alpha_matting_erode_size: int = Body(10, title='alpha matting erode size')
    ):
        if not model or model == "None":
            return

        input_image = api.decode_base64_to_image(input_image)

        image = rembg.remove(
            input_image,
            session=rembg.new_session(model),
            only_mask=return_mask,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            alpha_matting_erode_size=alpha_matting_erode_size,
        )

        return {"image": api.encode_pil_to_base64(image).decode("utf-8")}

    @app.post("/inpaint")
    async def lama_cleaner(
            input_image: str = Body("", title=' input image'),
            mask_image: str = Body("", title='mask image'),
    ):
        url = 'http://127.0.0.1:8080/inpaint'

        form_data = dict([
            ('ldmSteps', '25'), ('ldmSampler', 'plms'), ('zitsWireframe', 'true'),
            ('hdStrategy', 'Crop'), ('hdStrategyCropMargin', '196'), ('hdStrategyCropTrigerSize', '800'),
            ('hdStrategyResizeLimit', '2048'), ('prompt', ''), ('negativePrompt', ''),
            ('croperX', '256'), ('croperY', '256'), ('croperHeight', '512'), ('croperWidth', '512'),
            ('useCroper', 'false'), ('sdMaskBlur', '5'), ('sdStrength', '0.75'), ('sdSteps', '50'),
            ('sdGuidanceScale', '7.5'), ('sdSampler', 'uni_pc'), ('sdSeed', '-1'), ('sdMatchHistograms', 'false'),
            ('sdScale', '1'), ('cv2Radius', '5'), ('cv2Flag', 'INPAINT_NS'), ('paintByExampleSteps', '50'),
            ('paintByExampleGuidanceScale', '7.5'), ('paintByExampleSeed', '-1'), ('paintByExampleMaskBlur', '5'),
            ('paintByExampleMatchHistograms', 'false'), ('p2pSteps', '50'), ('p2pImageGuidanceScale', '1.5'),
            ('p2pGuidanceScale', '7.5'), ('controlnet_conditioning_scale', '0.4'),
            ('controlnet_method', 'control_v11p_sd15_canny')
        ])
        inputImage = io.BytesIO(base64.b64decode(input_image))
        outputImage = io.BytesIO(base64.b64decode(mask_image))

        files = {
            'image': ('image.png', inputImage.read(), 'image/png'),
            'mask': ('blob', outputImage.read(), 'image/png')
        }

        response = requests.post(url, data=form_data, files=files)
        image = Image.open(io.BytesIO(response.content))
        return {"image": api.encode_pil_to_base64(image).decode("utf-8")}

    @app.post("/restore")
    async def restore_old_photo(
            input_image: str = Body("", title=' input image'),
            scratch: bool = Body(False, title='wit Scratch'),
            colorize: bool = Body(False, title='colorize')
    ):
        form_data = {
            "input_image": input_image,
            "with_scratch": scratch,
            "colorize": colorize
        }
        headers = {'Content-type': 'application/json', 'Accept': 'application/json'}
        url = 'http://127.0.0.1:8081/restore'
        response = requests.post(url, data=json.dumps(form_data), headers=headers)
        return  response.json()

    @app.post("/toonify")
    async def toonify_photo(
            input_image: str = Body("", title=' input image'),
            style_index: int = Body(1, title='style_index'),
            style_type: str = Body("cartoon", title='style_type'),
    ):
        form_data = {
            "input_image": input_image,
            "style_index": style_index,
            "style_type": style_type
        }
        headers = {'Content-type': 'application/json', 'Accept': 'application/json'}
        url = 'http://127.0.0.1:8089/toonify'
        response = requests.post(url, data=json.dumps(form_data), headers=headers)
        return response.json()

    @app.post("/filters")
    async def filter_photo(
            input_image: str = Body("", title=' input image'),
            style_index: int = Body(1, title='style_index'),
    ):
        form_data = {
            "input_image": input_image,
            "selected_state_index": style_index,
        }
        headers = {'Content-type': 'application/json', 'Accept': 'application/json'}
        url = 'http://127.0.0.1:8089/filters'
        response = requests.post(url, data=json.dumps(form_data), headers=headers)
        return response.json()


try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(rembg_api)
except:
    pass
