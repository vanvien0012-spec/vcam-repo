"""
Test script for Stability AI / OpenAI inpainting APIs.

Usage:
    python test_api.py --backend stability --image photo.jpg --mask mask.png \
        --api-key YOUR_KEY --prompt "clean background, realistic" --output result.png

    python test_api.py --backend openai --image photo.jpg --mask mask.png \
        --api-key YOUR_KEY --prompt "remove watermark, realistic" --output result.png
"""

import argparse
import base64
import io
import os

import requests
from PIL import Image


def inpaint_stability(image_path, mask_path, api_key, prompt, output_path):
    url = "https://api.stability.ai/v2beta/stable-image/inpaint"

    with open(image_path, "rb") as f:
        image_bytes = f.read()
    with open(mask_path, "rb") as f:
        mask_bytes = f.read()

    files = {
        "image": ("image.png", image_bytes, "image/png"),
        "mask": ("mask.png", mask_bytes, "image/png"),
    }
    data = {
        "prompt": prompt or "clean background, realistic",
        "output_format": "png",
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    print("Sending request to Stability AI...")
    response = requests.post(url, headers=headers, files=files, data=data)

    if response.status_code != 200:
        raise RuntimeError(f"Stability AI error: {response.status_code} - {response.text}")

    result_image = Image.open(io.BytesIO(response.content))
    result_image.save(output_path)
    print(f"Saved result to {output_path}")
    return result_image


def inpaint_openai(image_path, mask_path, api_key, prompt, output_path):
    try:
        import openai
    except ImportError:
        raise ImportError("OpenAI package not installed. Run: pip install openai")

    client = openai.OpenAI(api_key=api_key)

    image = Image.open(image_path).convert("RGB").resize((1024, 1024))
    mask = Image.open(mask_path).convert("L").resize((1024, 1024))

    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_b64 = base64.b64encode(image_bytes.getvalue()).decode()

    mask_bytes = io.BytesIO()
    mask.save(mask_bytes, format="PNG")
    mask_b64 = base64.b64encode(mask_bytes.getvalue()).decode()

    print("Sending request to OpenAI...")
    response = client.images.edit(
        image=io.BytesIO(base64.b64decode(image_b64)),
        mask=io.BytesIO(base64.b64decode(mask_b64)),
        prompt=prompt or "realistic scene, remove masked area naturally",
        n=1,
        size="1024x1024",
    )

    result_url = response.data[0].url
    result_image = Image.open(requests.get(result_url, stream=True).raw)
    result_image.save(output_path)
    print(f"Saved result to {output_path}")
    return result_image


def main():
    parser = argparse.ArgumentParser(description="Test AI inpainting APIs")
    parser.add_argument("--backend", choices=["stability", "openai"], required=True)
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--mask", required=True, help="Path to mask image (white = remove)")
    parser.add_argument("--api-key", required=True, help="API key")
    parser.add_argument("--prompt", default="", help="Prompt describing the desired result")
    parser.add_argument("--output", default="api_result.png", help="Output path")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not os.path.exists(args.mask):
        raise FileNotFoundError(f"Mask not found: {args.mask}")

    if args.backend == "stability":
        inpaint_stability(args.image, args.mask, args.api_key, args.prompt, args.output)
    elif args.backend == "openai":
        inpaint_openai(args.image, args.mask, args.api_key, args.prompt, args.output)


if __name__ == "__main__":
    main()
