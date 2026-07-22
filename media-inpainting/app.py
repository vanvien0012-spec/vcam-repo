import os
import io
import cv2
import base64
import gradio as gr
import numpy as np
from PIL import Image
import requests

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def ensure_binary_mask(mask_pil, target_size=None):
    mask_np = np.array(mask_pil.convert("L"))
    if target_size is not None and (mask_np.shape[1], mask_np.shape[0]) != target_size:
        mask_np = cv2.resize(mask_np, target_size)
    _, binary_mask = cv2.threshold(mask_np, 1, 255, cv2.THRESH_BINARY)
    return binary_mask


def save_result(image, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    image.save(path)
    return path


# ---------------------------------------------------------------------------
# Image Inpainting Backends
# ---------------------------------------------------------------------------
def inpaint_opencv(image_pil, mask_pil, algorithm="Telea", radius=3):
    image_np = pil_to_cv2(image_pil)
    binary_mask = ensure_binary_mask(mask_pil, (image_np.shape[1], image_np.shape[0]))

    if cv2.countNonZero(binary_mask) == 0:
        return cv2_to_pil(image_np), "No mask drawn."

    flag = cv2.INPAINT_TELEA if algorithm == "Telea" else cv2.INPAINT_NS
    result = cv2.inpaint(image_np, binary_mask, radius, flag)
    return cv2_to_pil(result), "OpenCV inpainting complete."


def inpaint_lama(image_pil, mask_pil):
    try:
        from simple_lama_inpainting import SimpleLama
    except ImportError:
        return (
            None,
            "LaMa not installed. Run: pip install simple-lama-inpainting",
        )

    mask_pil = mask_pil.convert("L")
    lama = SimpleLama()
    result = lama(image_pil, mask_pil)
    return result, "LaMa inpainting complete."


def inpaint_stability(image_pil, mask_pil, api_key, prompt=""):
    if not api_key:
        return None, "Stability AI API key required."

    url = "https://api.stability.ai/v2beta/stable-image/inpaint"

    image_bytes = io.BytesIO()
    image_pil.convert("RGB").save(image_bytes, format="PNG")
    image_bytes.seek(0)

    mask_bytes = io.BytesIO()
    mask_pil.convert("L").save(mask_bytes, format="PNG")
    mask_bytes.seek(0)

    files = {
        "image": ("image.png", image_bytes.getvalue(), "image/png"),
        "mask": ("mask.png", mask_bytes.getvalue(), "image/png"),
    }
    data = {
        "prompt": prompt or "clean background, realistic",
        "output_format": "png",
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.post(url, headers=headers, files=files, data=data)
    if response.status_code != 200:
        return None, f"Stability AI error: {response.status_code} - {response.text}"

    result_image = Image.open(io.BytesIO(response.content))
    return result_image, "Stability AI inpainting complete."


def inpaint_openai(image_pil, mask_pil, api_key, prompt=""):
    if not api_key:
        return None, "OpenAI API key required."

    try:
        import openai
    except ImportError:
        return None, "OpenAI package not installed. Run: pip install openai"

    client = openai.OpenAI(api_key=api_key)

    image_resized = image_pil.convert("RGB").resize((1024, 1024))
    mask_resized = mask_pil.convert("L").resize((1024, 1024))

    image_bytes = io.BytesIO()
    image_resized.save(image_bytes, format="PNG")
    image_b64 = base64.b64encode(image_bytes.getvalue()).decode()

    mask_bytes = io.BytesIO()
    mask_resized.save(mask_bytes, format="PNG")
    mask_b64 = base64.b64encode(mask_bytes.getvalue()).decode()

    response = client.images.edit(
        image=io.BytesIO(base64.b64decode(image_b64)),
        mask=io.BytesIO(base64.b64decode(mask_b64)),
        prompt=prompt or "realistic scene, remove masked area naturally",
        n=1,
        size="1024x1024",
    )

    result_url = response.data[0].url
    result_image = Image.open(requests.get(result_url, stream=True).raw)
    return result_image, "OpenAI inpainting complete."


def inpaint_image(image_and_mask, backend, algorithm, radius, api_key, prompt):
    if image_and_mask is None or "image" not in image_and_mask:
        return None, "Please upload an image and draw a mask."

    image = image_and_mask["image"]
    mask = image_and_mask["mask"]

    if image is None:
        return None, "Please upload an image."
    if mask is None:
        return image, "Please draw a mask on the image."

    try:
        if backend == "OpenCV":
            result, status = inpaint_opencv(image, mask, algorithm, radius)
        elif backend == "LaMa":
            result, status = inpaint_lama(image, mask)
        elif backend == "Stability AI":
            result, status = inpaint_stability(image, mask, api_key, prompt)
        elif backend == "OpenAI":
            result, status = inpaint_openai(image, mask, api_key, prompt)
        else:
            return None, f"Unknown backend: {backend}"

        if result is None:
            return None, status

        path = save_result(result, f"inpaint_{backend.lower().replace(' ', '_')}.png")
        return result, f"{status} Saved to {path}"
    except Exception as e:
        return None, f"Error: {str(e)}"


# ---------------------------------------------------------------------------
# Video Inpainting Backends
# ---------------------------------------------------------------------------
def inpaint_video_opencv(video_path, mask_pil, algorithm="Telea", radius=3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Cannot open video file."

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(OUTPUT_DIR, "inpainted_video_opencv.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    binary_mask = ensure_binary_mask(mask_pil, (width, height))
    if cv2.countNonZero(binary_mask) == 0:
        return None, "No mask drawn."

    flag = cv2.INPAINT_TELEA if algorithm == "Telea" else cv2.INPAINT_NS
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        inpainted = cv2.inpaint(frame, binary_mask, radius, flag)
        out.write(inpainted)
        frame_count += 1

    cap.release()
    out.release()

    return output_path, f"OpenCV video inpainting: {frame_count} frames processed."


def inpaint_video_e2fgvi(video_path, mask_pil):
    try:
        import e2fgvi  # noqa: F401
    except ImportError:
        return (
            None,
            "E2FGVI not installed. This backend requires manual setup. "
            "See: https://github.com/MCG-NJU/E2FGVI",
        )
    return None, "E2FGVI backend placeholder - not yet implemented."


def inpaint_video(video_path, image_and_mask, backend, algorithm, radius):
    if video_path is None:
        return None, "Please upload a video."

    if image_and_mask is None or "image" not in image_and_mask:
        return None, "Please upload a reference frame and draw a mask."

    image = image_and_mask["image"]
    mask = image_and_mask["mask"]

    if image is None:
        return None, "Please upload a reference frame."
    if mask is None:
        return image, "Please draw a mask on the reference frame."

    try:
        if backend == "OpenCV":
            result, status = inpaint_video_opencv(video_path, mask, algorithm, radius)
        elif backend == "E2FGVI":
            result, status = inpaint_video_e2fgvi(video_path, mask)
        else:
            return None, f"Unknown backend: {backend}"

        return result, status
    except Exception as e:
        return None, f"Error: {str(e)}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="AI Media Inpainting") as demo:
    gr.Markdown("# AI Media Inpainting")
    gr.Markdown(
        "Upload an image or video, draw a mask, and choose a backend to remove "
        "unwanted objects, watermarks, or AI-generated artifacts."
    )

    with gr.Tab("Image Inpainting"):
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(
                    label="Upload image & draw mask",
                    type="pil",
                    tool="sketch",
                    brush_color="#FF0000",
                )
                backend_img = gr.Radio(
                    choices=["OpenCV", "LaMa", "Stability AI", "OpenAI"],
                    value="OpenCV",
                    label="Backend",
                )
                with gr.Row() as opencv_row_img:
                    algo_img = gr.Radio(
                        choices=["Telea", "Navier-Stokes"],
                        value="Telea",
                        label="OpenCV Algorithm",
                    )
                    radius_img = gr.Slider(
                        minimum=1, maximum=20, value=3, step=1, label="Radius"
                    )
                api_key_img = gr.Textbox(
                    label="API Key (Stability AI / OpenAI)",
                    type="password",
                    visible=False,
                )
                prompt_img = gr.Textbox(
                    label="Prompt (for AI backends)",
                    placeholder="e.g., clean background, realistic skin",
                    visible=False,
                )
                btn_img = gr.Button("Inpaint Image", variant="primary")
            with gr.Column():
                img_output = gr.Image(label="Result")
                status_img = gr.Textbox(label="Status")

        def update_img_controls(backend):
            is_opencv = backend == "OpenCV"
            is_api = backend in ["Stability AI", "OpenAI"]
            return (
                gr.update(visible=is_opencv),
                gr.update(visible=is_api),
                gr.update(visible=is_api),
            )

        backend_img.change(
            fn=update_img_controls,
            inputs=backend_img,
            outputs=[opencv_row_img, api_key_img, prompt_img],
        )

        btn_img.click(
            fn=inpaint_image,
            inputs=[
                img_input,
                backend_img,
                algo_img,
                radius_img,
                api_key_img,
                prompt_img,
            ],
            outputs=[img_output, status_img],
        )

    with gr.Tab("Video Inpainting"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload video")
                mask_ref = gr.Image(
                    label="Upload reference frame & draw mask",
                    type="pil",
                    tool="sketch",
                    brush_color="#FF0000",
                )
                backend_vid = gr.Radio(
                    choices=["OpenCV", "E2FGVI"],
                    value="OpenCV",
                    label="Backend",
                )
                with gr.Row() as opencv_row_vid:
                    algo_vid = gr.Radio(
                        choices=["Telea", "Navier-Stokes"],
                        value="Telea",
                        label="OpenCV Algorithm",
                    )
                    radius_vid = gr.Slider(
                        minimum=1, maximum=20, value=3, step=1, label="Radius"
                    )
                btn_vid = gr.Button("Inpaint Video", variant="primary")
            with gr.Column():
                video_output = gr.Video(label="Result")
                status_vid = gr.Textbox(label="Status")

        def update_vid_controls(backend):
            return gr.update(visible=(backend == "OpenCV"))

        backend_vid.change(
            fn=update_vid_controls,
            inputs=backend_vid,
            outputs=opencv_row_vid,
        )

        btn_vid.click(
            fn=inpaint_video,
            inputs=[
                video_input,
                mask_ref,
                backend_vid,
                algo_vid,
                radius_vid,
            ],
            outputs=[video_output, status_vid],
        )

    with gr.Tab("Settings"):
        gr.Markdown(
            "### API Keys\n"
            "Enter API keys above when using Stability AI or OpenAI backends.\n"
            "- Stability AI: https://platform.stability.ai/\n"
            "- OpenAI: https://platform.openai.com/"
        )
        gr.Markdown(
            "### Local Models\n"
            "- OpenCV: no setup required\n"
            "- LaMa: `pip install simple-lama-inpainting` (downloads model on first run)\n"
            "- E2FGVI: manual setup required, see https://github.com/MCG-NJU/E2FGVI"
        )

    gr.Markdown(
        "**Note:** Results are saved to the `outputs/` folder. "
        "OpenCV works offline; LaMa downloads a model on first use; "
        "Stability AI and OpenAI require API keys and internet."
    )

if __name__ == "__main__":
    demo.launch()
