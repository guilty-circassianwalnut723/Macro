import sys
import json
import requests
import base64
import os
import time
import io
from PIL import Image

from api_generator.utils import img2base64, base642img
from api_generator.generator import APIGenerator


class GPTImageAPIGenerator(APIGenerator):
    def __init__(self, api_key: str = "", model_name: str = "gpt-image-1",
                 timeout: int = 300, max_try: int = sys.maxsize - 1):
        """
        GPT Image API Generator.
        Supports both image generation and image editing.
        
        Args:
            api_key: Your OpenAI API key (set via environment variable OPENAI_API_KEY or pass directly).
            model_name: Image generation model name (e.g., "gpt-image-1", "dall-e-3").
            timeout: Request timeout in seconds.
            max_try: Maximum number of retries.
        
        Note:
            Set self.base_url to your actual API endpoint before use.
        """
        super().__init__(api_key, "", 0, model_name, timeout, max_try)
        self.api_key = api_key
        self.model_name = model_name
        # Set via environment variable OPENAI_API_BASE or pass base_url directly
        self.base_url = os.environ.get("OPENAI_API_BASE", "")  # e.g., "https://api.openai.com/v1" or your proxy
        # Will be set based on whether images are provided
        self.usage_app_url = None

    def gen_payload(self, prompt: str, response_format: str, think=False, images=None):
        # Determine which endpoint to use based on whether images are provided
        if images and len(images) > 0:
            # Use edit endpoint when images are provided
            self.usage_app_url = f"{self.base_url}/images/edits" if self.base_url else ""
            return self._gen_edit_payload(prompt, images)
        else:
            # Use generation endpoint when no images are provided
            self.usage_app_url = f"{self.base_url}/images/generations" if self.base_url else ""
            return self._gen_generation_payload(prompt)

    def _gen_generation_payload(self, prompt: str):
        """Generate payload for image generation endpoint"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024"
        }

        # Add model-specific parameters
        if self.model_name == "dall-e-3":
            payload["quality"] = "standard"
            payload["style"] = "vivid"
            payload['size'] = "1024x1024"
        elif self.model_name == "gpt-image-1":
            payload["quality"] = "auto"
            payload["background"] = "auto"
            payload["moderation"] = "low"
            payload["output_compression"] = 100
            payload['size'] = "auto"

        return json.dumps(payload)

    def _gen_edit_payload(self, prompt: str, images):
        """Generate payload for image edit endpoint"""
        # Convert images to base64
        images_base64 = [img2base64(img) for img in images] if images else []

        payload = {
            "model": "gpt-image-1",  # Edit endpoint only supports gpt-image-1
            "prompt": prompt,
            "image": images_base64,
            "n": 1,
            "size": "auto",
            "quality": "auto"
        }

        return json.dumps(payload)

    def deal_response(self, response, response_format):
        response_data = response.json()
        if response.status_code == 200 and "data" in response_data:
            # Get the image data from response
            image_data = response_data["data"][0]

            # Check if we have b64_json (gpt-image-1) or url (dall-e-3)
            if "b64_json" in image_data:
                # gpt-image-1 returns base64 encoded image
                img = base642img(image_data["b64_json"])
                return img
            elif "url" in image_data:
                # dall-e-3 returns image URL
                image_url = image_data["url"]

                # Download the image from URL
                img_response = requests.get(image_url, timeout=self.timeout)
                if img_response.status_code == 200:
                    # Convert to PIL Image
                    img = Image.open(io.BytesIO(img_response.content))
                    return img
                else:
                    raise Exception(f"Failed to download image from URL: {img_response.status_code}")
            else:
                raise Exception("No image data found in response")
        else:
            raise Exception(f"API Error: {response_data}")


# Usage example
if __name__ == "__main__":
    import os
    # Set your API key via environment variable
    api_key = os.getenv("OPENAI_API_KEY", "")
    api_base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    
    generator = GPTImageAPIGenerator(api_key=api_key)
    generator.base_url = api_base_url

    prompt = "A beautiful sunset over the ocean with vibrant orange and pink colors."
    response = generator.gen_response(prompt)
    if response:
        response.save("output.png")
