import os
import sys
import time
import json
import requests
from PIL import Image
import re

from api_generator.utils import img2base64
from api_generator.generator import APIGenerator


class GeminiAPIGenerator(APIGenerator):
    def __init__(self, app_key: str = "",
                 max_tokens: int = 65535, model_name: str = "gemini-3.0-flash-preview",
                 timeout: int = 300, max_try: int = sys.maxsize - 1, print_log: bool = False):
        """
        Gemini API Generator.
        
        Args:
            app_key: Your Gemini API key (set via environment variable GEMINI_API_KEY or pass directly).
            max_tokens: Maximum number of tokens to generate.
            model_name: Gemini model name (e.g., "gemini-2.5-flash", "gemini-2.5-pro").
            timeout: Request timeout in seconds.
            max_try: Maximum number of retries.
            print_log: Whether to print request logs.
        """
        super().__init__(app_key, "", max_tokens, model_name, timeout, max_try)
        # Set via environment variable GEMINI_API_URL or assign directly after construction
        self.usage_app_url = os.environ.get("GEMINI_API_URL", "")  # e.g., "https://your-api-endpoint/v1/chat/completions"
        self.print_log = print_log

    def gen_payload(self, prompt: str, response_format: str, think=False, images=None, temperature=0.8, top_p=0.8):
        messages = [{
            'role': 'user',
            'content': [
            ]
        }]

        if images:
            if '<IMAGE_TOKEN>' in prompt:
                part_prompts = prompt.split('<IMAGE_TOKEN>')
                assert len(part_prompts) == len(images) + 1, f'prompt: {prompt} and images: {images} have different length'
                for i in range(len(images)):
                    messages[0]['content'].append({
                        'type': 'text',
                        'text': part_prompts[i]
                    })
                    messages[0]['content'].append({
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/jpeg;base64,{img2base64(images[i])}'
                        }
                    })
                messages[0]['content'].append({
                    'type': 'text',
                    'text': part_prompts[-1]
                })
            else:
                messages[0]['content'].append({
                    'type': 'text',
                    'text': prompt
                })
                for i, image in enumerate(images):
                    messages[0]['content'].append({
                        'type': 'text',
                        'text': f'image {i + 1}: '
                    })
                    messages[0]['content'].append({
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/jpeg;base64,{img2base64(image)}'
                        }
                    })
        else:
            messages[0]['content'].append({
                'type': 'text',
                'text': prompt
            })

        if response_format:
            messages[0]['content'].append({
                'type': 'text',
                'text': '\n\nDirectly answer with format below, without markdown format\n' \
                    + (('{' + '\n'.join([f'\"{item}\": [{item.lower()}]' for item in response_format]) + '}') if isinstance(response_format, dict) \
                        else '\n'.join(item for item in response_format))
            })

        # Prepare data based on model type
        if self.model_name in ["gemini-3-pro-preview",
                               "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash",
                               "gemini-2.5-pro-us-west4", "gemini-2.5-pro-europe-west4"]:
            if think:
                data = {
                    "model": self.model_name,
                    "extra_body": {
                        "google": {
                            "thinking_config": {
                                "include_thoughts": True,
                                "thinking_budget": -1
                            },
                            "thought_tag_marker": "think"
                        }
                    },
                    "messages": messages,
                    "stream": False,
                    "max_tokens": self.max_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                }
            else:
                data = {
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "max_tokens": self.max_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                }
        else:
            data = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "max_tokens": self.max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }

        return json.dumps(data)

    def deal_response(self, response, response_format):
        content = json.loads(response.content)
        content = content['choices'][0]['message']['content']

        if isinstance(response_format, list):
            response_dict = {}
            for item in response_format:
                response_dict[item] = re.search(f'{item}: (.*)', content).group(1)
        elif isinstance(response_format, dict):
            if self.print_log:
                print(content, flush=True)
            if ('```json' in content):
                content = content.split('```json')[1].split('```')[0]
            try:
                response_dict = json.loads(content)
            except Exception as e:
                try:
                    response_dict = json.loads(content + '}')
                except Exception as e:
                    raise ValueError(f"Error parsing JSON: {e}")

        return response_dict

    def gen_response(
        self,
        prompt,
        response_format=None,
        think=False,
        images=None,
        temperature=0.8,
        top_p=0.8,
    ):
        headers = {
            'Authorization': f'Bearer {self.app_id}',
            'Content-Type': 'application/json',
        } if self.app_id else None

        payload = self.gen_payload(prompt, response_format, think, images, temperature, top_p)

        for i in range(self.max_try):
            try:
                if self.print_log:
                    print(f"开始第{i+1}次请求", flush=True)
                ret = requests.post(self.usage_app_url, data=payload, headers=headers, timeout=self.timeout)
                if ret.status_code == 200:
                    output = self.deal_response(ret, response_format)
                    if self.print_log:
                        print(f"第{i+1}次请求成功", flush=True)
                    return output
                else:
                    raise Exception(f"状态码：{ret.status_code}, {self.trans_state(ret.status_code)}")
            except Exception as e:
                if self.print_log:
                    print(f"请求异常：{e}", flush=True)
                time.sleep(10)
                continue

        return None


if __name__ == "__main__":
    # Usage example
    # Set your API key and URL before running
    import os
    api_key = os.getenv("GEMINI_API_KEY", "")
    api_url = os.getenv("GEMINI_API_URL", "")
    
    generator = GeminiAPIGenerator(
        app_key=api_key,
        max_tokens=65535,
        model_name='gemini-2.5-flash'
    )
    generator.usage_app_url = api_url

    response = generator.gen_response(
        prompt="Describe this image briefly.",
        response_format=None,
    )

    print(response)
