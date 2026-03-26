import sys
import json
import time
import requests
from pathlib import Path


class APIGenerator:
    def __init__(self, app_key: str, app_url: str,
                 max_tokens: int, model_name: str, timeout: int = 120,
                 max_try: int = sys.maxsize - 1, print_log: bool = True):
        self.app_id = app_key
        self.app_url = app_url
        self.usage_app_url = app_url if app_url else None
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_try = max_try
        self.print_log = print_log
        print(f'model_name:{self.model_name}, max_tokens:{self.max_tokens}, timeout:{self.timeout}, max_try:{self.max_try}')

    def update_app_url(self):
        self.usage_app_url = self.app_url

    def gen_payload(self, prompt: str, response_format: str, think=False, images=None):
        pass

    def trans_state(self, state: int):
        status_dict = {200: 'success', 400: 'Bad Request', 408: 'Request Timeout', 429: 'Too Many Requests',
                       500: 'Internal Server Error', 504: 'Timeout'}
        return status_dict.get(state, 'Unknown Status')

    def deal_response(self, response, response_format):
        pass

    def gen_response(
        self,
        prompt,
        response_format=None,
        think=False,
        images=None,
    ):
        headers = {
            'Authorization': f'Bearer {self.app_id}',
            'Content-Type': 'application/json',
        } if self.app_id else None

        payload = self.gen_payload(prompt, response_format, think, images)

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
                time.sleep(5 + 3 * (2 ** i))
                continue

        return None
