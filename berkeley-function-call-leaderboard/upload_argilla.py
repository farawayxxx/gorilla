import re
import json
import random
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import argilla as rg
from argilla.client.feedback.utils import assign_records
from argilla import Workspace

def load_jsonl(filename):
    """Load a JSONL file and return a list of dictionaries."""
    data = []
    with open(filename, 'r') as file:
        for line in file:
            try:
                data.append(json.loads(line.strip()))
            except:
                continue
                # print(line)
    return data

rg.init(
    api_url="http://lb-8p3b92y6-bp65jl3u1qckyv0t.clb.ap-nanjing.tencentclb.com:6900",
    api_key="UesXP65QWFtkHhBCJvvdGIXRER2c6iw8aQMUgwsmtfXS2LU4J-zVlUITsSGSVbJpC5cZVM-3bjVxKKhgxSgO0GI8MDYb09qAHNusxywvWVc"
    #api_key="SZKLQJY8OJVehT27M7OH53W3W6oLUAKABoV9GFXxscqqYDGUSwQltvt2VVhyrw82RCbjije-bAtc9mFMOmM3WxLyZZvDD_NTaCI3XZsMXEY",
)
# Workspace.create('route')

dataset = rg.FeedbackDataset(
    fields = [
        rg.TextField(name="id", title="ID"),
        rg.TextField(name="split", title="Data split"),
        rg.TextField(name="functions", title="Functions"),
        rg.TextField(name="query", title="Query"),
        rg.TextField(name="function_calls", title="Function Calls"),
    ],
    questions = [
        rg.LabelQuestion(name="label", title="是否正确", labels=["Yes", "No"]),
        rg.TextQuestion(name="rewrite_title", title="标题改写", required=False),
        rg.TextQuestion(name="rewrite_content", title="正文改写", required=False),
        rg.TextQuestion(name="comment", title="备注", required=False)
    ],
    guidelines = "根据标注文档，检查是否有badcase。",
    vectors_settings=None
)

input_file = "./bfcl.jsonl"
dataset_name = "bfcl"
records = []

all_data = load_jsonl(input_file)
print(f"Total lines in file: {len(all_data)}")

sample_size = min(500, len(all_data))
sampled_data = random.sample(all_data, sample_size)
print(f"Sampled data size: {sample_size}")

records = []
for data in tqdm(sampled_data, desc="Processing records"):
    record = rg.FeedbackRecord(
        fields = {
            "id": data["id"],
            "split": data["split"],
            "query": json.dumps(data["query"], indent=2, ensure_ascii=False),
            "functions": json.dumps(data["function"], indent=2, ensure_ascii=False),
            "function_calls": json.dumps(data["ground_truth"], ensure_ascii=False)
        }
    )
    records.append(record)

print(f"Processed records: {len(records)}")
dataset.add_records(records)
remote_dataset = dataset.push_to_argilla(name=dataset_name, workspace="sft") 

# python to_argilla.py ./xhs/xhs_for_argilla_1107.jsonl xhs_1107_2