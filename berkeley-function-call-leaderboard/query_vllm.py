import httpx
import json
from openai import OpenAI


model_name = "qwen25"
# model_name = "llama31"
model_name = "FuncGen-1204"

def stream_generate_vllm(messages, url, seed=42, temp=0.7, top_p=1.0):
    client = OpenAI(
        base_url=url,
        api_key="token-abc123",
        timeout=httpx.Timeout(30.0, read=90.0)
    )
    extra_body = {}
    if model_name == 'llama3':
        extra_body = {
            "stop_token_ids": [128009]
        }
    # elif "qwen" in model_name:

    
    completion = client.chat.completions.create(
        model = model_name,
        messages = messages,
        seed = seed,
        temperature = temp,
        top_p = top_p,
        extra_body = extra_body
    )
    print(completion)
    response_str = completion.choices[0].message.content
    return response_str

url_list = [
    '146.56.238.200:30061', '146.56.238.200:30062',
    '119.45.31.97:30061', '119.45.31.97:30062',
    '175.27.234.177:30061', '175.27.234.177:30062',
    '118.195.151.245:30061', '118.195.151.245:30062',
    '43.137.6.251:30061', '43.137.6.251:30062',
    '175.27.239.129:30061',  '175.27.239.129:30062',
    '119.45.0.25:30061', '119.45.0.25:30062',
    '119.45.145.27:30061', '119.45.145.27:30062'
]
url_list = [
    '146.56.238.200:30091', '146.56.238.200:30092',
    '119.45.31.97:30091', '119.45.31.97:30092',
    '175.27.234.177:30091', '175.27.234.177:30092',
    '118.195.151.245:30091', '118.195.151.245:30092',
    '43.137.6.251:30064', '43.137.6.251:30092',
    '175.27.239.129:30091',  '175.27.239.129:30092',
    '119.45.0.25:30091', '119.45.0.25:30092',
    '119.45.145.27:30091', '119.45.145.27:30092'
]

url_list = ["127.0.0.1:11053"]

SYSTEM_PROMPT = """You are tasked with assisting in making tool calls. To fulfill the user's request, you must select one or more tools from the available list and correctly populate the tool parameters. Your specific tasks are as follows:

1. **Make one or more tool/function calls**: Based on the user's input, determine the appropriate tool(s) to use.
2. **Handle unresolvable requests**: If none of the available tools can be used to fulfill the request, make it clear and refuse to provide an answer.
3. **Handle missing parameters**: If the user's input lacks necessary parameters for any tool, point out the missing information.

Your output must be in **JSON format**, specifying a list of generated tool calls. Follow this example format, ensuring all parameter types are correct:

```json
{{
    "codeplan": "Pseudo-code plan describing how to address the request.",
    "tool_calls": [
        {{
            "name": "api_name1",
            "arguments": {{
                "argument1": "value1",
                "argument2": "value2"
            }}
        }}
    ]
}}
```

The codeplan should be a high-level pseudo-code description of the logic used to address the user's request, written in a Python-like syntax. It should outline the steps or flow of the solution.

If no tool call is required, return an empty `"tool_calls": []`.

**Note:** You must not include any additional text in your response.

Below is a list of available tools in JSON format that you can invoke:

{tools} 
"""

# SYSTEM_PROMPT = """Use the following tools to address the user's query. 
# {tools}

# Your output must be in **JSON format**, specifying a list of generated tool calls. Follow this example format, ensuring all parameter types are correct:

# ```json
# {{
#     "tool_calls": [
#         {{
#             "name": "api_name1",
#             "arguments": {{
#                 "argument1": "value1",
#                 "argument2": "value2"
#             }}
#         }}
#     ]
# }}
# ```

# """

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

message = [
    {'role': 'system', 'content': SYSTEM_PROMPT.format(tools=json.dumps(tools))},
    {"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"}
]

action = {"codeplan": "\n            # User asks about the weather in San Francisco, Tokyo, and Paris\n            current_weather_sf = get_current_weather(\"San Francisco, CA\", unit=\"fahrenheit\")\n            current_weather_tky = get_current_weather(\"Tokyo, Japan\", unit=\"celsius\")\n            current_weather_prs = get_current_weather(\"Paris, France\", unit=\"celsius\")\n\n            # Construct the response with the current weather information\n            response = f\"Here's the current weather in those cities:\\n\\n**San Francisco, California, USA**\\n{current_weather_sf}\\n\\n**Tokyo, Japan**\\n{current_weather_tky}\\n\\n**Paris, France**\\n{current_weather_prs}\\n\\nPlease note that these are the current conditions and may change. For more detailed and up-to-date information, I recommend checking a reputable weather website or app, such as the National Weather Service, Weather.com, or a local news source.\"\n            return response\n        ", "tool_calls": [{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}, {"name": "get_current_weather", "arguments": {"location": "Tokyo, Japan", "unit": "celsius"}}, {"name": "get_current_weather", "arguments": {"location": "Paris, France", "unit": "celsius"}}]}

message = [
    {'role': 'system', 'content': SYSTEM_PROMPT.format(tools=json.dumps(tools))},
    {"role": "user", "content": "What's the weather like in San Francisco, Tokyo, and Paris?"},
    # {"role": "assistant_action", "content": json.dumps(action)}
]

for i, url in enumerate(url_list):
    url_temp = f'http://{url}/v1'
    try:
        res = stream_generate_vllm(messages=message, url=url_temp)
        print(res)
        print(f'{i+1}  OK')
    except: 
        print(f'{i+1}  FAIL')