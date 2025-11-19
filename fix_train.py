#!/usr/bin/env python3
"""Fix train.py - add model card generator and update checkpoint saving"""

with open('scripts/train.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line with save_epoch_checkpoint function
save_fn_idx = None
for i, line in enumerate(lines):
    if line.strip().startswith('def save_epoch_checkpoint'):
        save_fn_idx = i
        break

if save_fn_idx is None:
    print("ERROR: Could not find save_epoch_checkpoint")
    exit(1)

# Model card generator to insert before save_epoch_checkpoint
model_card_gen = '''def generate_model_card(stats, epoch, global_step, config, stage_name):
    """Generate model card with training statistics"""
    from datetime import datetime
    
    # Calculate images seen
    batch_size = getattr(config, 'batch_size', 32)
    max_samples = getattr(config, 'max_samples', None)
    total_images = (epoch + 1) * max_samples if max_samples else f"Epoch {epoch + 1} completed"
    
    # Quantization status
    quant_status = []
    if stats.get('quantization', {}).get('vision_quantized'):
        quant_status.append("Vision: 4-bit")
    if stats.get('quantization', {}).get('language_quantized'):
        quant_status.append("Language: 4-bit")
    if stats.get('quantization', {}).get('memory_quantized'):
        quant_status.append("Memory: 1.58-bit")
    quant_text = ", ".join(quant_status) if quant_status else "None (full precision)"
    
    model_card = f"""---
license: apache-2.0
tags:
- vision-language
- multimodal
- vlm
- microvlm
library_name: transformers
---

# MicroVLM-V

**Stage:** {stage_name}  
**Current Epoch:** {epoch}  
**Training Step:** {global_step}  
**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

## Model Statistics

### Overall Parameters
- **Total Parameters:** {stats['overall']['total_params']:,}
- **Trainable Parameters:** {stats['overall']['trainable_params']:,} ({stats['overall']['trainable_percent']:.2f}%)
- **Frozen Parameters:** {stats['overall']['frozen_params']:,}

### Component Breakdown

#### Vision Encoder
- **Total:** {stats['vision']['total']:,} parameters
- **Trainable:** {stats['vision']['trainable']:,}
- **Frozen:** {stats['vision']['frozen']:,}

#### Language Model
- **Total:** {stats['language']['total']:,} parameters
- **Trainable:** {stats['language']['trainable']:,}
- **Frozen:** {stats['language']['frozen']:,}

#### Adapter Module
- **Total:** {stats['adapter']['total']:,} parameters
- **Trainable:** {stats['adapter']['trainable']:,}

#### Memory Module
- **Total:** {stats['memory']['total']:,} parameters
- **Trainable:** {stats['memory']['trainable']:,}

### Quantization
{quant_text}

### Training Progress
- **Images Processed:** {total_images}
- **Estimated Model Size:** ~{stats['size']['estimated_mb']:.2f} MB

## Training Configuration
- **Stage:** {stage_name}
- **Optimizer:** {getattr(config, 'optimizer', 'adamw')}
- **Learning Rate:** {getattr(config, 'learning_rate', 'N/A')}
- **Batch Size:** {batch_size}

## Usage

```python
from src.models.microvlm_v import MicroVLMV
from transformers import AutoTokenizer

model = MicroVLMV.from_pretrained("euhidaman/MicroVLM-V")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-0.5B-Instruct")
```

## Citation

```bibtex
@misc{{microvlm-v-2025,
  title={{MicroVLM-V: Efficient Vision-Language Model}},
  author={{Your Name}},
  year={{2025}},
  url={{https://huggingface.co/euhidaman/MicroVLM-V}}
}}
```
"""
    return model_card

'''

# Insert model card generator
lines.insert(save_fn_idx, model_card_gen)

# Now update save_epoch_checkpoint (find it again after insertion)
for i, line in enumerate(lines):
    if 'checkpoint_path = checkpoint_dir / f"epoch_{epoch}_checkpoint.pt"' in line:
        lines[i] = '    checkpoint_path = checkpoint_dir / "model_checkpoint.pt"\n'
    elif 'stats_path = checkpoint_dir / f"epoch_{epoch}_statistics.json"' in line:
        lines[i] = '    stats_path = checkpoint_dir / "statistics.json"\n'
        # Add metadata wrapper
        next_line_idx = i + 1
        if 'with open(stats_path' in lines[next_line_idx]:
            # Insert metadata wrapper before writing
            lines.insert(next_line_idx, '    stats_with_meta = {\\n')
            lines.insert(next_line_idx + 1, '        \\'epoch\\': epoch,\\n')
            lines.insert(next_line_idx + 2, '        \\'global_step\\': global_step,\\n')
            lines.insert(next_line_idx + 3, '        \\'stage\\': stage_name,\\n')
            lines.insert(next_line_idx + 4, '        \\'statistics\\': stats\\n')
            lines.insert(next_line_idx + 5, '    }\\n')
            # Update the json.dump line
            for j in range(next_line_idx + 6, next_line_idx + 10):
                if 'json.dump(stats' in lines[j]:
                    lines[j] = lines[j].replace('json.dump(stats,', 'json.dump(stats_with_meta,')
                    break

# Add model card generation before return statement in save_epoch_checkpoint
for i in range(len(lines) - 1, 0, -1):
    if lines[i].strip() == 'return checkpoint_path, stats' and i > save_fn_idx:
        # Insert model card generation before return
        indent = '    '
        lines.insert(i, f'{indent}# Generate model card\\n')
        lines.insert(i + 1, f'{indent}model_card_path = checkpoint_dir / "README.md"\\n')
        lines.insert(i + 2, f'{indent}model_card = generate_model_card(stats, epoch, global_step, config, stage_name)\\n')
        lines.insert(i + 3, f'{indent}with open(model_card_path, \\'w\\', encoding=\\'utf-8\\') as f:\\n')
        lines.insert(i + 4, f'{indent}    f.write(model_card)\\n')
        lines.insert(i + 5, f'{indent}print(f"📄 Model card updated: {{model_card_path}}")\\n')
        lines.insert(i + 6, f'\\n')
        break

# Update push_to_huggingface to use upload_file instead of upload_folder
for i, line in enumerate(lines):
    if 'from huggingface_hub import HfApi, create_repo, upload_folder' in line:
        lines[i] = 'from huggingface_hub import HfApi, create_repo, upload_file\\n'
        break

# Find and update push_to_huggingface commit message
for i, line in enumerate(lines):
    if 'commit_message = f"{stage_name}: epoch {epoch} checkpoint"' in line:
        lines[i] = '        commit_message = f"stage{stage_name}: epoch {epoch} (latest checkpoint)"\\n'
        break

# Replace upload_folder with individual file uploads
for i, line in enumerate(lines):
    if 'api.upload_folder(' in line:
        # Replace upload_folder block with upload_file calls
        # Find the end of this block (closing parenthesis)
        block_start = i
        block_end = i
        for j in range(i, min(i + 10, len(lines))):
            if ')' in lines[j] and 'token=hf_token' in lines[j]:
                block_end = j
                break
        
        # Create replacement lines
        replacement = '''        # Upload files individually to ensure overwrite
        print(f"   ⏳ Uploading checkpoint files...")
        
        checkpoint_path = Path(checkpoint_dir) / "model_checkpoint.pt"
        stats_path = Path(checkpoint_dir) / "statistics.json"
        readme_path = Path(checkpoint_dir) / "README.md"
        
        # Upload checkpoint file
        if checkpoint_path.exists():
            api.upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo="model_checkpoint.pt",
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
                token=hf_token
            )
            print(f"   ✓ Uploaded model_checkpoint.pt ({checkpoint_path.stat().st_size / 1e9:.2f} GB)")
        
        # Upload statistics file
        if stats_path.exists():
            api.upload_file(
                path_or_fileobj=str(stats_path),
                path_in_repo="statistics.json",
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
                token=hf_token
            )
            print(f"   ✓ Uploaded statistics.json")
        
        # Upload README (model card)
        if readme_path.exists():
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
                token=hf_token
            )
            print(f"   ✓ Uploaded README.md (model card updated)")
'''
        
        # Remove old block
        del lines[block_start-1:block_end+1]
        
        # Insert new code
        for new_line in reversed(replacement.split('\\n')):
            lines.insert(block_start-1, new_line + '\\n')
        
        break

# Update get_model_statistics to return nested dict structure
for i, line in enumerate(lines):
    if 'stats[\'total_parameters\'] = overall[\'total\']' in line:
        # Replace flat structure with nested
        # Find the section and replace it
        start = i
        end = i
        for j in range(i, min(i + 20, len(lines))):
            if 'stats[\'trainable_percentage\']' in lines[j]:
                end = j
                break
        
        replacement = '''    # Overall parameters
    overall = count_parameters(model)
    stats['overall'] = {
        'total_params': overall['total'],
        'trainable_params': overall['trainable'],
        'frozen_params': overall['frozen'],
        'trainable_percent': 100.0 * overall['trainable'] / overall['total'] if overall['total'] > 0 else 0.0
    }
'''
        del lines[start-2:end+1]
        for new_line in reversed(replacement.split('\\n')):
            lines.insert(start-2, new_line + '\\n')
        break

with open('scripts/train.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("✅ Fixed train.py successfully!")
print("  - Added generate_model_card() function")
print("  - Updated save_epoch_checkpoint() to use single files")
print("  - Updated push_to_huggingface() to upload individual files")
print("  - Model card will be generated and uploaded on each epoch")
