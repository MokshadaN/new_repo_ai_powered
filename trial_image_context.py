# trial_image_context.py
import base64
from ollama import Client

OLLAMA_HOST = "http://127.0.0.1:11434"
MODEL_NAME = "gemma3:latest"

def encode_image_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def generate_image_context(image_path: str) -> str:
    client = Client(host=OLLAMA_HOST)
    prompt = (
        "Describe this image in a concise, search-friendly way: main objects, colors, materials, "
        "any readable text, scene/setting, actions, viewpoint, lighting, and object counts. "
        "Example style: 'A bright red rubber ball on green grass, single ball, daylight, close-up shot.'"
    )
    b64_img = encode_image_b64(image_path)
    res = client.generate(
        model=MODEL_NAME,
        prompt=prompt,
        images=[b64_img],  # list of base64-encoded images
        stream=False,
    )
    print(res)
    return res.get("response", "").strip()

if __name__ == "__main__":
    image_path = "F:/test_folder/test_folder/IMG-20240811-WA0030.jpg" # update this
    context = generate_image_context(image_path)
    print("Generated context:\n", context)
