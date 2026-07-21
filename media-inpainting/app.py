import os
import cv2
import gradio as gr
import numpy as np
from PIL import Image

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def inpaint_image(image_and_mask, algorithm="Telea", radius=3):
    """
    image_and_mask: dict from Gradio Image(sketch) with keys 'image' and 'mask'
    algorithm: 'Telea' or 'Navier-Stokes'
    radius: inpainting radius
    """
    if image_and_mask is None or "image" not in image_and_mask:
        return None, "Please upload an image and draw a mask."

    image = image_and_mask["image"]
    mask = image_and_mask["mask"]

    # Convert PIL images to numpy arrays
    image_np = np.array(image.convert("RGB"))
    mask_np = np.array(mask.convert("L"))  # Grayscale mask

    # Create binary mask: any non-zero pixel is part of the mask
    _, binary_mask = cv2.threshold(mask_np, 1, 255, cv2.THRESH_BINARY)

    if cv2.countNonZero(binary_mask) == 0:
        return image, "No mask drawn. Please paint over the area to remove."

    # Choose inpainting algorithm
    flag = cv2.INPAINT_TELEA if algorithm == "Telea" else cv2.INPAINT_NS

    result = cv2.inpaint(image_np, binary_mask, radius, flag)

    # Save output
    output_path = os.path.join(OUTPUT_DIR, "inpainted_result.png")
    Image.fromarray(result).save(output_path)

    return Image.fromarray(result), f"Saved to {output_path}"


def inpaint_video(video_path, mask_image, algorithm="Telea", radius=3):
    """
    Simple video inpainting by extracting frames, applying mask to each frame,
    and re-encoding the video.
    """
    if video_path is None:
        return None, "Please upload a video."
    if mask_image is None or "mask" not in mask_image:
        return None, "Please draw a mask on the reference image."

    mask = mask_image["mask"]
    mask_np = np.array(mask.convert("L"))
    _, binary_mask = cv2.threshold(mask_np, 1, 255, cv2.THRESH_BINARY)

    if cv2.countNonZero(binary_mask) == 0:
        return None, "No mask drawn."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Cannot open video file."

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(OUTPUT_DIR, "inpainted_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    flag = cv2.INPAINT_TELEA if algorithm == "Telea" else cv2.INPAINT_NS
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize mask if frame size differs
        if frame.shape[1] != binary_mask.shape[1] or frame.shape[0] != binary_mask.shape[0]:
            resized_mask = cv2.resize(binary_mask, (frame.shape[1], frame.shape[0]))
        else:
            resized_mask = binary_mask

        inpainted = cv2.inpaint(frame, resized_mask, radius, flag)
        out.write(inpainted)
        frame_count += 1

    cap.release()
    out.release()

    return output_path, f"Processed {frame_count} frames. Saved to {output_path}"


with gr.Blocks(title="AI Media Inpainting") as demo:
    gr.Markdown("# AI Media Inpainting")
    gr.Markdown(
        "Upload an image or video, draw a mask over the area you want to remove, "
        "and the app will fill it with realistic content."
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
                algo_img = gr.Radio(
                    choices=["Telea", "Navier-Stokes"],
                    value="Telea",
                    label="Algorithm",
                )
                radius_img = gr.Slider(
                    minimum=1, maximum=20, value=3, step=1, label="Inpaint radius"
                )
                btn_img = gr.Button("Inpaint Image", variant="primary")
            with gr.Column():
                img_output = gr.Image(label="Result")
                status_img = gr.Textbox(label="Status")

        btn_img.click(
            fn=inpaint_image,
            inputs=[img_input, algo_img, radius_img],
            outputs=[img_output, status_img],
        )

    with gr.Tab("Video Inpainting"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload video")
                mask_ref = gr.Image(
                    label="Upload first frame reference & draw mask",
                    type="pil",
                    tool="sketch",
                    brush_color="#FF0000",
                )
                algo_vid = gr.Radio(
                    choices=["Telea", "Navier-Stokes"],
                    value="Telea",
                    label="Algorithm",
                )
                radius_vid = gr.Slider(
                    minimum=1, maximum=20, value=3, step=1, label="Inpaint radius"
                )
                btn_vid = gr.Button("Inpaint Video", variant="primary")
            with gr.Column():
                video_output = gr.Video(label="Result")
                status_vid = gr.Textbox(label="Status")

        btn_vid.click(
            fn=inpaint_video,
            inputs=[video_input, mask_ref, algo_vid, radius_vid],
            outputs=[video_output, status_vid],
        )

    gr.Markdown(
        "**Note:** OpenCV inpainting works well for small, simple removals. "
        "For larger or more complex areas, we'll integrate LaMa or Stable Diffusion in the next phase."
    )

if __name__ == "__main__":
    demo.launch(share=False)
