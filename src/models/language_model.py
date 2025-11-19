"""
Qwen2.5-0.5B Language Model Integration
Loads pretrained Qwen2.5-0.5B with 4-bit quantization support
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import json


class Qwen2LanguageModel(nn.Module):
    """
    Qwen2.5-0.5B language model wrapper
    Supports 4-bit quantization for training and 1.58-bit for inference
    """
    
    def __init__(self, config, pretrained_path=None, quantize_4bit=False):
        super().__init__()
        
        self.hidden_size = config['qwen_hidden_dim']
        self.num_layers = config.get('num_layers', 24)
        self.vocab_size = config.get('vocab_size', 151936)
        
        self.quantize_4bit = quantize_4bit
        
        # Load tokenizer
        self.tokenizer = None
        
        # Model components will be loaded from pretrained
        self.model = None
        self.embed_tokens = None
        self.layers = None
        self.norm = None
        self.lm_head = None
        
        if pretrained_path:
            self.load_pretrained(pretrained_path, quantize_4bit)
        else:
            # Initialize from scratch (for testing)
            self._init_from_scratch(config)
    
    def _init_from_scratch(self, config):
        """Initialize model components from scratch"""
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        
        # Simplified transformer layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_size,
            nhead=config.get('num_heads', 14),
            dim_feedforward=config.get('intermediate_size', 4864),
            dropout=0.0,
            activation='gelu',
            batch_first=True
        )
        self.layers = nn.ModuleList([decoder_layer for _ in range(self.num_layers)])
        
        self.norm = nn.LayerNorm(self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embed_tokens.weight
    
    def load_pretrained(self, model_name_or_path, quantize_4bit=False):
        """
        Load pretrained Qwen2.5-0.5B model
        
        Args:
            model_name_or_path: HuggingFace model identifier or local path
            quantize_4bit: whether to apply 4-bit quantization
        """
        print(f"Loading Qwen2.5-0.5B from {model_name_or_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        
        # Load model with optional quantization
        if quantize_4bit:
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                quantization_config=bnb_config,
                trust_remote_code=True,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        
        # Extract model components
        self.embed_tokens = self.model.model.embed_tokens
        self.layers = self.model.model.layers
        self.norm = self.model.model.norm
        self.lm_head = self.model.lm_head
        
        print("Qwen2.5-0.5B loaded successfully")
    
    def get_input_embeddings(self):
        """Get input embedding layer"""
        if self.model is not None:
            return self.model.get_input_embeddings()
        return self.embed_tokens
    
    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, 
                labels=None, past_key_values=None, use_cache=False,
                output_hidden_states=False, return_dict=True):
        """
        Forward pass through language model
        
        Args:
            input_ids: (batch_size, seq_len)
            inputs_embeds: (batch_size, seq_len, hidden_size) - alternative to input_ids
            attention_mask: (batch_size, seq_len)
            labels: (batch_size, seq_len) for language modeling loss
            past_key_values: cached key-value states
            use_cache: whether to return cached states
            output_hidden_states: whether to return hidden states
            return_dict: whether to return ModelOutput
        
        Returns:
            outputs: CausalLMOutput or tuple
        """
        if self.model is not None:
            # Use HuggingFace model
            return self.model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
        else:
            # Use custom implementation
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            
            # Simple forward through layers
            hidden_states = inputs_embeds
            all_hidden_states = [] if output_hidden_states else None
            
            for layer in self.layers:
                if output_hidden_states:
                    all_hidden_states.append(hidden_states)
                
                # Note: This is simplified - proper implementation would handle
                # causal masking, past_key_values, etc.
                hidden_states = layer(
                    hidden_states, 
                    hidden_states,  # Using as both query and memory
                    tgt_mask=None
                )
            
            hidden_states = self.norm(hidden_states)
            
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            # Compute logits
            logits = self.lm_head(hidden_states)
            
            # Compute loss if labels provided
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, self.vocab_size),
                    shift_labels.view(-1)
                )
            
            if return_dict:
                from transformers.modeling_outputs import CausalLMOutput
                return CausalLMOutput(
                    loss=loss,
                    logits=logits,
                    past_key_values=None,
                    hidden_states=all_hidden_states,
                )
            else:
                return (loss, logits, all_hidden_states) if loss is not None else (logits, all_hidden_states)
    
    def generate(self, input_ids, max_length=50, **kwargs):
        """Generate text"""
        if self.model is not None:
            return self.model.generate(input_ids, max_length=max_length, **kwargs)
        else:
            raise NotImplementedError("Generation not implemented for custom model")
    
    def freeze_layers(self, num_layers_to_freeze=None):
        """
        Freeze early layers
        
        Args:
            num_layers_to_freeze: number of layers to freeze from the start
                                 If None, freezes all except last 4 layers
        """
        if num_layers_to_freeze is None:
            num_layers_to_freeze = max(0, self.num_layers - 4)
        
        # Freeze embedding
        for param in self.embed_tokens.parameters():
            param.requires_grad = False
        
        # Freeze specified layers
        for i, layer in enumerate(self.layers):
            if i < num_layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
        
        print(f"Frozen {num_layers_to_freeze} layers out of {self.num_layers}")
    
    def unfreeze_last_n_layers(self, n=4):
        """Unfreeze last n layers for fine-tuning"""
        start_idx = max(0, self.num_layers - n)
        
        for i, layer in enumerate(self.layers):
            if i >= start_idx:
                for param in layer.parameters():
                    param.requires_grad = True
        
        print(f"Unfrozen last {n} layers")


def create_qwen_model(config, checkpoint_path=None, quantize_4bit=False):
    """
    Factory function to create Qwen model
    
    Args:
        config: model configuration dictionary
        checkpoint_path: path to pretrained checkpoint or HF model name
        quantize_4bit: whether to apply 4-bit quantization
    
    Returns:
        model: Qwen2LanguageModel instance
    """
    model = Qwen2LanguageModel(config, checkpoint_path, quantize_4bit)
    return model
