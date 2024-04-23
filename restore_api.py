import base64
import io
import os
import uuid
from io import BytesIO

from PIL import Image
from fastapi import FastAPI, Body
import main


app = FastAPI()

def modify(input_image, colorize=False, scratches=False):
    print(f'handling input: {input_image}')
    folderName = str(uuid.uuid4())
    inputDirectory = os.path.join(os.path.dirname(__file__), "uploads")
    dynamicDirectory = os.path.join(inputDirectory, folderName)
    dynamicInputDirectory = os.path.join(dynamicDirectory, "input")
    dynamicOutputDirectory = os.path.join(dynamicDirectory, "output")
    if not os.path.exists(dynamicInputDirectory):
        os.makedirs(dynamicInputDirectory)
    fileName = "a.png"
    image_filename = os.path.join(dynamicInputDirectory, fileName)
    image_file = input_image.save(image_filename)
    image_name = os.path.basename(image_filename)
    output_dir = main.run(input_dir=dynamicInputDirectory, output_dir=dynamicOutputDirectory, sr_scale=2,
                          run_mode=main.RunMode.ENHANCE_RESTORE,
                          hr_quality=True, hr_restore=False, use_gpu=True, colorize=colorize,
                          inpaint_scratches=scratches)

    img_name, _ = os.path.splitext(image_name)
    for output_file in os.listdir(output_dir):
        output_name, _ = os.path.splitext(output_file)
        if img_name == output_name:
            output_img = os.path.join(output_dir, output_file)
            print(f'Finished, output: {output_img}')
            return output_img

    raise ValueError("couldn't find output image")


def image_to_data(image):
    with BytesIO() as output:
        image.save(output, format="PNG")
        data = output.getvalue()
    return data

@app.post("/restore")
async def restore_photos(
        input_image: str = Body("", title=' input image'),
        with_scratch: bool = Body(False, title='Scratches'),
        colorize: bool = Body(False, title='colorize'),
):
    #print(input_image)
    im = Image.open(BytesIO(base64.b64decode(input_image)))
    output_img_file = modify(im, colorize=colorize, scratches=with_scratch)

    image = Image.open(output_img_file)
    image = image.resize(im.size)
    with io.BytesIO() as output_bytes:
        image.save(output_bytes, format="PNG")
        img_str = base64.b64encode(output_bytes.getvalue()).decode("utf-8")
    return {"image": img_str }
