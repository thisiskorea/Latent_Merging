"""
Core latent merging utilities for LERP, SLERP, RegMean, and Task Vector (delta steering).
This module is distilled from the project notebooks for reuse in scripts.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

# Defaults from the notebook experiments
DEFAULT_BASE = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_FT = "open-thoughts/OpenThinker3-7B"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Simple caches to avoid reloading
_MODEL_CACHE: Dict[Tuple[str, bool, bool], AutoModelForCausalLM] = {}
_TOKENIZER_CACHE: Dict[str, AutoTokenizer] = {}


def get_tokenizer(name: str) -> AutoTokenizer:
    if name not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[name] = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    return _TOKENIZER_CACHE[name]


def get_model(name: str, load_in_8bit: bool = False, load_in_4bit: bool = False) -> AutoModelForCausalLM:
    key = (name, bool(load_in_8bit), bool(load_in_4bit))
    if key not in _MODEL_CACHE:
        kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": DTYPE,
            "device_map": "auto",
        }
        if load_in_8bit:
            kwargs["load_in_8bit"] = True
        if load_in_4bit:
            kwargs["load_in_4bit"] = True
        _MODEL_CACHE[key] = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    return _MODEL_CACHE[key]


def apply_chat_template(
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
    return_tensors: str = "pt",
    device: Optional[torch.device] = None,
):
    prompt_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
    inputs = tokenizer(prompt_text, return_tensors=return_tensors)
    if device is not None:
        inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def get_decoder_layers(model: AutoModelForCausalLM) -> List[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise RuntimeError("Decoder layers not found (model structure changed?)")


def sanitize_layers(model: AutoModelForCausalLM, requested: List[int]) -> Tuple[List[int], int]:
    n = len(get_decoder_layers(model))
    valid = sorted([i for i in set(requested) if 0 <= i < n])
    return valid, n


def to_device_dtype(x: torch.Tensor, like: torch.nn.Module) -> torch.Tensor:
    param = next(like.parameters())
    return x.to(device=param.device, dtype=param.dtype)


def top_p_sample(logits: Tensor, top_p: float = 0.9, temperature: float = 1.0) -> int:
    if temperature and temperature != 1.0:
        logits = logits / float(temperature)
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumulative > top_p
    cutoff[..., 0] = False
    sorted_probs[cutoff] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    idx = torch.multinomial(sorted_probs, num_samples=1)
    next_token = sorted_indices.gather(-1, idx).item()
    return int(next_token)


def _normalize(x: Tensor, eps: float = 1e-9) -> Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)


@torch.no_grad()
def _slerp(u: Tensor, v: Tensor, beta: Tensor, eps: float = 1e-6) -> Tensor:
    dot = (u * v).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    small = theta < 1e-3
    sin_theta = torch.sin(theta).clamp_min(eps)
    w1 = torch.sin((1.0 - beta) * theta) / sin_theta
    w2 = torch.sin(beta * theta) / sin_theta
    out = w1 * u + w2 * v
    if small.any():
        out_lin = (1.0 - beta) * u + beta * v
        out = torch.where(small, out_lin, out)
    return out


class ActivationSteerer:
    """
    Task Vector / delta steering: hidden <- hidden + alpha * delta(layer).
    """

    def __init__(self, model: AutoModelForCausalLM, deltas: Dict[int, Tensor], alpha: float = 0.10,
                 apply_to_all_tokens: bool = True):
        self.model = model
        self.alpha = alpha
        self.apply_to_all_tokens = apply_to_all_tokens
        valid_layers, _ = sanitize_layers(model, list(deltas.keys()))
        self.deltas = {int(k): to_device_dtype(deltas[k], model) for k in valid_layers}
        self.hooks: List[Any] = []
        self._register()

    def _pre_hook(self, layer_idx: int):
        delta_vec = self.deltas[layer_idx]

        def hook(module, inputs):
            if not inputs:
                return
            hidden_states = inputs[0]
            if hidden_states is None:
                return
            delta = self.alpha * delta_vec.view(1, 1, -1).to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            if self.apply_to_all_tokens:
                new_hs = hidden_states + delta
            else:
                new_hs = hidden_states.clone()
                new_hs[:, -1:, :] = new_hs[:, -1:, :] + delta
            return (new_hs,) + tuple(inputs[1:])

        return hook

    def _register(self):
        for idx, layer in enumerate(get_decoder_layers(self.model)):
            if idx in self.deltas:
                h = layer.register_forward_pre_hook(self._pre_hook(idx), with_kwargs=False)
                self.hooks.append(h)

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


class LayerLatentMixer:
    """
    Base/FT hidden mixing at a specific layer.
    mix_mode: "lerp" or "slerp".
    """

    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        ft_model: AutoModelForCausalLM,
        layer_idx: int,
        beta: float = 0.5,
        last_token_only: bool = True,
        mix_mode: str = "lerp",
    ):
        self.base_model = base_model
        self.ft_model = ft_model
        self.beta = float(beta)
        self.last_token_only = last_token_only
        self.mix_mode = mix_mode
        nb = len(get_decoder_layers(base_model))
        nf = len(get_decoder_layers(ft_model))
        if layer_idx < 0 or layer_idx >= nb or layer_idx >= nf:
            layer_idx = min(nb, nf) - 1
        self.layer_idx = int(layer_idx)
        self.ft_hidden: Optional[Tensor] = None
        self._h1 = None
        self._h2 = None
        self._register()

    def _register(self):
        base_layer = get_decoder_layers(self.base_model)[self.layer_idx]
        ft_layer = get_decoder_layers(self.ft_model)[self.layer_idx]

        def ft_capture(module, inputs):
            if not inputs:
                return
            h = inputs[0]
            if h is None:
                return
            self.ft_hidden = (h[:, -1:, :].detach() if self.last_token_only else h.detach())
            return None

        def base_mix(module, inputs):
            if not inputs:
                return
            h = inputs[0]
            if h is None or self.ft_hidden is None:
                return

            h_ft = self.ft_hidden.to(h.dtype).to(h.device)
            h_b = h if not self.last_token_only else h[:, -1:, :]
            beta = torch.tensor(self.beta, device=h.device, dtype=h.dtype).view(1, 1, 1)

            if self.mix_mode == "slerp":
                u = _normalize(h_b)
                v = _normalize(h_ft)
                dir_slerp = _slerp(u, v, beta)
                dir_slerp = _normalize(dir_slerp)
                nb = h_b.norm(p=2, dim=-1, keepdim=True)
                nf = h_ft.norm(p=2, dim=-1, keepdim=True)
                norm_interp = (1.0 - beta) * nb + beta * nf
                mixed_token = norm_interp * dir_slerp
            else:
                mixed_token = (1.0 - beta) * h_b + beta * h_ft

            if self.last_token_only:
                mixed = h.clone()
                mixed[:, -1:, :] = mixed_token
            else:
                mixed = mixed_token
            return (mixed,) + tuple(inputs[1:])

        self._h1 = ft_layer.register_forward_pre_hook(ft_capture, with_kwargs=False)
        self._h2 = base_layer.register_forward_pre_hook(base_mix, with_kwargs=False)

    def remove(self):
        if self._h1:
            self._h1.remove()
        if self._h2:
            self._h2.remove()
        self._h1 = None
        self._h2 = None
        self.ft_hidden = None


class LayerLatentRegMeanMixer:
    """
    RegMean mixing: combine multiple FT hidden states with weights and a base pull (lambda_reg).
    """

    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        ft_models: Sequence[AutoModelForCausalLM],
        layer_idx: int,
        lambda_reg: float = 1.0,
        ft_weights: Optional[List[float]] = None,
        last_token_only: bool = True,
    ):
        self.base_model = base_model
        self.ft_models = list(ft_models)
        self.lambda_reg = float(lambda_reg)
        self.ft_weights = ft_weights
        self.last_token_only = last_token_only
        self.layer_idx = int(layer_idx)
        self.ft_hidden: List[Tensor] = []
        self._h_ft: List[Any] = []
        self._h_base = None
        self._register()

    def _register(self):
        base_layer = get_decoder_layers(self.base_model)[self.layer_idx]
        ft_layers = [get_decoder_layers(m)[self.layer_idx] for m in self.ft_models]

        def make_ft_capture(idx_ft: int):
            def ft_capture(module, inputs):
                if not inputs:
                    return
                h = inputs[0]
                if h is None:
                    return
                val = h[:, -1:, :].detach() if self.last_token_only else h.detach()
                if len(self.ft_hidden) <= idx_ft:
                    self.ft_hidden.append(val)
                else:
                    self.ft_hidden[idx_ft] = val
                return None

            return ft_capture

        for i, layer in enumerate(ft_layers):
            h = layer.register_forward_pre_hook(make_ft_capture(i), with_kwargs=False)
            self._h_ft.append(h)

        def base_mix(module, inputs):
            if not inputs:
                return
            h = inputs[0]
            if h is None or not self.ft_hidden:
                return
            h_b = h if not self.last_token_only else h[:, -1:, :]
            weights = None
            if self.ft_weights is not None:
                weights = torch.tensor(self.ft_weights, device=h.device, dtype=h.dtype)
                weights = weights / (weights.sum() + 1e-12)
            ft_stack = torch.stack([t.to(h.device).to(h.dtype) for t in self.ft_hidden], dim=0)  # (M,B,T,H)
            if weights is None:
                ft_mean = ft_stack.mean(dim=0)
            else:
                ft_mean = (weights.view(-1, 1, 1, 1) * ft_stack).sum(dim=0)
            beta = torch.tensor(1.0 / (1.0 + self.lambda_reg), device=h.device, dtype=h.dtype).view(1, 1, 1)
            mixed_token = (1.0 - beta) * h_b + beta * ft_mean
            if self.last_token_only:
                mixed = h.clone()
                mixed[:, -1:, :] = mixed_token
            else:
                mixed = mixed_token
            return (mixed,) + tuple(inputs[1:])

        self._h_base = base_layer.register_forward_pre_hook(base_mix, with_kwargs=False)

    def remove(self):
        if self._h_base:
            self._h_base.remove()
        for h in self._h_ft:
            h.remove()
        self._h_base = None
        self._h_ft = []
        self.ft_hidden = []


@torch.no_grad()
def chat_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    device = next(model.parameters()).device
    inputs = apply_chat_template(tokenizer, messages, add_generation_prompt=True, return_tensors="pt", device=device)
    out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=temperature, top_p=top_p)
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


@torch.no_grad()
def ensemble_generate(
    base_model: AutoModelForCausalLM,
    ft_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    beta: float = 0.5,
    beta_schedule: Optional[Callable[[int, int], float]] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    eos_token_id: Optional[int] = None,
) -> str:
    device = next(base_model.parameters()).device
    inputs = apply_chat_template(tokenizer, messages, add_generation_prompt=True, return_tensors="pt", device=device)
    input_ids = inputs["input_ids"]
    attn = inputs.get("attention_mask")
    base_out = base_model(input_ids=input_ids, attention_mask=attn, use_cache=True, return_dict=True)
    ft_out = ft_model(input_ids=input_ids, attention_mask=attn, use_cache=True, return_dict=True)

    generated: List[int] = []
    if eos_token_id is None and hasattr(tokenizer, "eos_token_id"):
        eos_token_id = tokenizer.eos_token_id

    for step in range(max_new_tokens):
        logits_base = base_out.logits[:, -1, :]
        logits_ft = ft_out.logits[:, -1, :]
        b = beta_schedule(step, max_new_tokens) if beta_schedule else beta
        logits_mix = (1.0 - b) * logits_base + b * logits_ft
        next_token_id = top_p_sample(logits_mix.squeeze(0), top_p=top_p, temperature=temperature)
        generated.append(next_token_id)
        if eos_token_id is not None and next_token_id == eos_token_id:
            break
        next_ids = torch.tensor([[next_token_id]], device=device)
        base_out = base_model(input_ids=next_ids, use_cache=True,
                              past_key_values=base_out.past_key_values, return_dict=True)
        ft_out = ft_model(input_ids=next_ids, use_cache=True,
                          past_key_values=ft_out.past_key_values, return_dict=True)

    full = torch.cat([input_ids, torch.tensor([generated], device=device)], dim=1)
    return tokenizer.decode(full[0], skip_special_tokens=True)


@torch.no_grad()
def latent_mix_generate(
    base_model: AutoModelForCausalLM,
    ft_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    mix_layer: int = 20,
    beta: float = 0.5,
    mix_mode: str = "lerp",
    last_token_only: bool = True,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    eos_token_id: Optional[int] = None,
) -> str:
    device = next(base_model.parameters()).device
    mixer = LayerLatentMixer(base_model, ft_model, mix_layer, beta=beta,
                             last_token_only=last_token_only, mix_mode=mix_mode)
    inputs = apply_chat_template(tokenizer, messages, add_generation_prompt=True, return_tensors="pt", device=device)
    input_ids = inputs["input_ids"]
    attn = inputs.get("attention_mask")
    ft_out = ft_model(input_ids=input_ids, attention_mask=attn, use_cache=True, return_dict=True)
    base_out = base_model(input_ids=input_ids, attention_mask=attn, use_cache=True, return_dict=True)

    generated: List[int] = []
    if eos_token_id is None and hasattr(tokenizer, "eos_token_id"):
        eos_token_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        logits = base_out.logits[:, -1, :]
        next_token_id = top_p_sample(logits.squeeze(0), top_p=top_p, temperature=temperature)
        generated.append(next_token_id)
        if eos_token_id is not None and next_token_id == eos_token_id:
            break
        next_ids = torch.tensor([[next_token_id]], device=device)
        ft_out = ft_model(input_ids=next_ids, use_cache=True,
                          past_key_values=ft_out.past_key_values, return_dict=True)
        base_out = base_model(input_ids=next_ids, use_cache=True,
                              past_key_values=base_out.past_key_values, return_dict=True)

    mixer.remove()
    full = torch.cat([input_ids, torch.tensor([generated], device=device)], dim=1)
    return tokenizer.decode(full[0], skip_special_tokens=True)


@torch.no_grad()
def latent_regmean_generate(
    base_model: AutoModelForCausalLM,
    ft_models: Sequence[AutoModelForCausalLM],
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    mix_layer: int = 20,
    lambda_reg: float = 1.0,
    ft_weights: Optional[List[float]] = None,
    last_token_only: bool = True,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    eos_token_id: Optional[int] = None,
) -> str:
    device = next(base_model.parameters()).device
    mixer = LayerLatentRegMeanMixer(
        base_model, ft_models, layer_idx=mix_layer,
        lambda_reg=lambda_reg, ft_weights=ft_weights,
        last_token_only=last_token_only,
    )
    inputs = apply_chat_template(tokenizer, messages, add_generation_prompt=True, return_tensors="pt", device=device)
    input_ids = inputs["input_ids"]
    attn = inputs.get("attention_mask")
    ft_outs = [m(input_ids=input_ids, attention_mask=attn, use_cache=True, return_dict=True) for m in ft_models]
    base_out = base_model(input_ids=input_ids, attention_mask=attn, use_cache=True, return_dict=True)

    generated: List[int] = []
    if eos_token_id is None and hasattr(tokenizer, "eos_token_id"):
        eos_token_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        logits = base_out.logits[:, -1, :]
        next_token_id = top_p_sample(logits.squeeze(0), top_p=top_p, temperature=temperature)
        generated.append(next_token_id)
        if eos_token_id is not None and next_token_id == eos_token_id:
            break
        next_ids = torch.tensor([[next_token_id]], device=device)
        ft_outs = [
            m(input_ids=next_ids, use_cache=True, past_key_values=o.past_key_values, return_dict=True)
            for m, o in zip(ft_models, ft_outs)
        ]
        base_out = base_model(input_ids=next_ids, use_cache=True,
                              past_key_values=base_out.past_key_values, return_dict=True)

    mixer.remove()
    full = torch.cat([input_ids, torch.tensor([generated], device=device)], dim=1)
    return tokenizer.decode(full[0], skip_special_tokens=True)


@torch.no_grad()
def delta_generate(
    base_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    deltas: Dict[int, Tensor],
    messages: List[Dict[str, str]],
    alpha: float = 0.10,
    apply_to_all_tokens: bool = True,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    steerer = ActivationSteerer(base_model, deltas, alpha=alpha, apply_to_all_tokens=apply_to_all_tokens)
    try:
        return chat_generate(base_model, tokenizer, messages,
                             max_new_tokens=max_new_tokens,
                             temperature=temperature, top_p=top_p)
    finally:
        steerer.remove()
