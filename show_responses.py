"""Print a few generated responses for eyeballing. Usage:
    python show_responses.py [jsonl_path] [data_type] [n]
"""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "data/responses_train_pilot.jsonl"
dt = sys.argv[2] if len(sys.argv) > 2 else "vanilla_harmful"
n = int(sys.argv[3]) if len(sys.argv) > 3 else 4

shown = 0
for line in open(path):
    r = json.loads(line)
    if r["data_type"] != dt:
        continue
    print("=" * 70)
    print("PROMPT:", r["prompt"])
    print("-- RESPONSE --")
    print(r["response"][:1500])
    print()
    shown += 1
    if shown >= n:
        break
