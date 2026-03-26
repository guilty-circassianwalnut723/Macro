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


class NanoAPIGenerator(APIGenerator):
    def __init__(self, api_key: str = "", model_name: str = "gemini-image-preview",
                 timeout: int = 60, max_try: int = sys.maxsize - 1, print_log: bool = False):
        """
        Nano API Generator for image generation.
        
        Args:
            api_key: Your API key (set via environment variable NANO_API_KEY or pass directly).
            model_name: Image generation model name.
            timeout: Request timeout in seconds.
            max_try: Maximum number of retries.
            print_log: Whether to print request logs.
        
        Note:
            Set self.base_url to your actual API endpoint before use.
            This API uses a two-step process: submit job -> poll for result.
        """
        super().__init__(api_key, "", 0, model_name, timeout, max_try, print_log)
        self.api_key = api_key
        self.model_name = model_name
        # Set via environment variable NANO_API_BASE_URL or pass base_url directly
        self.base_url = os.environ.get("NANO_API_BASE_URL", "")  # e.g., "https://your-api-endpoint/v1/models"
        self.usage_app_url = f"{self.base_url}/{self.model_name}:imageGenerate" if self.base_url else ""
        self.print_log = print_log

    def gen_payload(self, prompt: str, response_format: str, think=False, images=None):
        """Generate payload for image generation API"""
        contents = {
            "parts": [
                {"text": prompt}
            ]
        }

        # Add images if provided
        if images:
            images_base64 = [img2base64(img) for img in images] if images else []
            for img_base64 in images_base64:
                contents["parts"].append({
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": img_base64
                    }
                })

        payload = {
            "contents": [contents],
            "generationConfig": {
                "imageConfig": {
                    "imageSize": "2K"
                }
            }
        }

        return json.dumps(payload, ensure_ascii=False).encode('utf-8')

    def deal_response(self, response, response_format):
        """Handle the two-step response process for Nano API"""
        if response.status_code == 200:
            # First response contains a task ID
            task_id = response.text.strip()
            if self.print_log:
                print(f"Task ID: {task_id}", flush=True)

            # Poll for completion
            # Set your query URL template here
            query_url = f"{self.base_url.replace('https://', 'http://')}/{task_id}:imageGenerateQuery" if self.base_url else ""
            headers = {
                'Authorization': f'Bearer {self.app_id}',
                'Content-Type': 'application/json',
            }

            # Poll until completion
            while True:
                time.sleep(1)  # Wait 1 second between polls
                try:
                    query_response = requests.post(query_url, headers=headers, timeout=self.timeout)
                    if query_response.status_code == 200:
                        query_data = query_response.json()
                        status = query_data.get("status", 0)
                        if self.print_log:
                            print(f"Status: {status}", flush=True)

                        if status == 1:  # Completed
                            # Download the generated image
                            # Set your image download URL template here
                            img_url = query_data.get("image_url", "")
                            img_response = requests.get(img_url, timeout=self.timeout)

                            if img_response.status_code == 200:
                                # Convert to PIL Image
                                img = Image.open(io.BytesIO(img_response.content))
                                return img
                            else:
                                raise Exception(f"Failed to download image from URL: {img_url}")
                        elif status == -1:  # Failed
                            raise Exception("Image generation failed")
                        # Continue polling for status 0 (processing)
                    else:
                        raise Exception(f"Query request failed: {query_response.status_code}")
                except Exception as e:
                    if self.print_log:
                        print(f"Polling error: {e}", flush=True)
                    raise e
        else:
            raise Exception(f"Initial request failed: {response.status_code} - {response.text}")


# Usage example
if __name__ == "__main__":
    import os
    # Set your API key via environment variable
    api_key = os.getenv("NANO_API_KEY", "")
    
    generator = NanoAPIGenerator(api_key=api_key, print_log=True)
    # Set base_url before use
    # generator.base_url = "https://your-api-endpoint/v1/models"
    # generator.usage_app_url = f"{generator.base_url}/{generator.model_name}:imageGenerate"
    
    prompt = "Generate an image of a sunset over the ocean."
    response = generator.gen_response(prompt)
    if response:
        response.save("output.png")
