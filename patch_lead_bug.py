"""Patch script: fix lead_captured $setOnInsert conflict in chat_router.py"""
import re

path = "routers/chat_router.py"
with open(path, "rb") as f:
    raw = f.read()

content = raw.decode("utf-8")

# Find and report the current state
idx = content.find("lead_captured")
if idx == -1:
    print("ERROR: 'lead_captured' not found at all — may already be patched or wrong file")
else:
    print(f"Found 'lead_captured' at index {idx}")
    print("Context:", repr(content[max(0,idx-120):idx+80]))

# The $setOnInsert block to remove lead_captured from
# We'll use regex to find and replace the block
pattern = re.compile(
    r'(\"\$setOnInsert\"\s*:\s*\{[^}]*?\"lead_captured\"[^}]*?\},)',
    re.DOTALL
)
match = pattern.search(content)
if match:
    print("\nFound $setOnInsert block with lead_captured:")
    print(repr(match.group(0)))
else:
    print("\nPattern not found with regex, using line-based approach")
    lines = content.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if "lead_captured" in line:
            print(f"Line {i+1}: {repr(line)}")
