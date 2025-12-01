import re

# Read original file
with open(r'd:\BabyLM\MicroVLM-V\DIAGRAM.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove all mermaid blocks and replace with simple text note
content = re.sub(
    r'```mermaid.*?```',
    '**[Visual diagram - see architecture.md for detailed component specifications]**',
    content,
    flags=re.DOTALL
)

# Write back
with open(r'd:\BabyLM\MicroVLM-V\DIAGRAM.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("Converted all Mermaid diagrams to simple references")
