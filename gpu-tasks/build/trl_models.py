"""Parameter counts and checkpoint sizes for the TRL records, from each run's Hugging Face config. Stdlib only.

Each count is the standard transformer parameter formula for the architecture, computed from the config
fields W&B logged. Where the Hub publishes a safetensors parameter count the formula was checked against it
on 2026-10-19 (HUB below); the Hub figures include a few buffers a Trainer checkpoint never saves, and the
formula matches them to the buffer.
"""

HUB = {
    "gpt2": (137022720, "F32, which also counts 12 x 1024^2 attention-mask buffers; the formula gives 124,439,808 parameters"),
    "facebook/opt-350m": (None, "no safetensors count on the Hub; the formula gives 331,196,416, the 331M the model is named for"),
    "EleutherAI/pythia-1b-deduped": (1011782144, "F16, which also counts 16 x 32 rotary inv_freq buffers; the formula gives 1,011,781,632"),
    "trl-lib/pythia-1b-deduped-tldr-sft": (1011781632, "BF16, parameters only; equal to the formula"),
    "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr": (None, "no safetensors count on the Hub; same architecture as pythia-1b-deduped"),
    "trl-lib/pythia-2.8b-deduped-tldr-sft": (None, "no safetensors count on the Hub; EleutherAI/pythia-2.8b-deduped lists 2,775,209,280 F16 "
                                                   "including 32 x 10 rotary buffers; the formula gives 2,775,208,960"),
    "cleanrl/EleutherAI_pythia-6.9b-deduped__sft__tldr": (None, "no safetensors count on the Hub for it or for EleutherAI/pythia-6.9b-deduped; the formula gives 6,857,302,016"),
    "Qwen/Qwen2-0.5B": (494032768, "BF16, parameters only; equal to the formula"),
    "Qwen/Qwen2-0.5B-Instruct": (494032768, "BF16, parameters only; equal to the formula"),
    "google/gemma-2-27b": (27227128320, "F32, parameters only; equal to the formula"),
    "llava-hf/llava-1.5-7b-hf": (7063427072, "F16, parameters only; equal to the formula"),
}


def _g(c, *keys, default=None):
    for k in keys:
        v = c.get(k)
        if isinstance(v, dict) and "value" in v:
            v = v["value"]
        if v is not None:
            return v
    return default


def causal_lm_params(c):
    """(parameter count, formula) for the causal-LM form of the config's architecture, or (None, why)."""
    mt = _g(c, "model_type")
    V = _g(c, "vocab_size")
    if mt == "gpt2":
        d, L, P = _g(c, "n_embd", "hidden_size"), _g(c, "n_layer", "num_hidden_layers"), _g(c, "n_positions", "max_position_embeddings")
        n = V * d + P * d + L * (3 * d * d + 3 * d + d * d + d + 4 * d * d + 4 * d + 4 * d * d + d + 4 * d) + 2 * d
        return n, f"GPT-2: V*d + P*d + L*(12 d^2 + 13 d) + 2d with V={V}, d={d}, P={P}, L={L}; lm_head tied"
    if mt == "opt":
        d, L, F = _g(c, "hidden_size"), _g(c, "num_hidden_layers"), _g(c, "ffn_dim")
        W, P = _g(c, "word_embed_proj_dim") or d, _g(c, "max_position_embeddings")
        n = V * W + (P + 2) * d + (2 * W * d if W != d else 0) + L * (4 * (d * d + d) + d * F + F + F * d + d + 4 * d)
        n += 2 * d if _g(c, "do_layer_norm_before") else 0
        return n, (f"OPT: V*W + (P+2)*d + project_in/out + L*(4(d^2+d) + 2dF + F + d + 4d) with V={V}, W={W}, d={d}, P={P}, L={L}, F={F}; "
                   "lm_head tied; no decoder-level final norm when do_layer_norm_before is false")
    if mt == "gpt_neox":
        d, L, F = _g(c, "hidden_size"), _g(c, "num_hidden_layers"), _g(c, "intermediate_size")
        n = 2 * V * d + L * (4 * d + 3 * d * d + 3 * d + d * d + d + d * F + F + F * d + d) + 2 * d
        return n, f"GPT-NeoX: 2*V*d (untied embed_in and embed_out) + L*(4d^2 + 2dF + 9d + F) + 2d with V={V}, d={d}, L={L}, F={F}"
    if mt == "qwen2":
        d, L, F, h, kv = (_g(c, "hidden_size"), _g(c, "num_hidden_layers"), _g(c, "intermediate_size"),
                          _g(c, "num_attention_heads"), _g(c, "num_key_value_heads"))
        hd = _g(c, "head_dim") or d // h
        attn = d * h * hd + h * hd + 2 * (d * kv * hd + kv * hd) + h * hd * d
        n = V * d + L * (attn + 3 * d * F + 2 * d) + d + (0 if _g(c, "tie_word_embeddings") else V * d)
        return n, (f"Qwen2: V*d + L*(q,k,v with bias, o, gate/up/down, 2 norms) + norm (+ lm_head if untied) with V={V}, d={d}, L={L}, F={F}, "
                   f"heads={h}, kv_heads={kv}, head_dim={hd}, tied={bool(_g(c, 'tie_word_embeddings'))}")
    if mt == "gemma2":
        d, L, F, h, kv, hd = (_g(c, "hidden_size"), _g(c, "num_hidden_layers"), _g(c, "intermediate_size"),
                              _g(c, "num_attention_heads"), _g(c, "num_key_value_heads"), _g(c, "head_dim"))
        attn = d * h * hd + 2 * d * kv * hd + h * hd * d
        n = V * d + L * (attn + 3 * d * F + 4 * d) + d + (0 if _g(c, "tie_word_embeddings") else V * d)
        return n, (f"Gemma 2: V*d + L*(q,k,v,o without bias, gate/up/down, 4 norms) + norm with V={V}, d={d}, L={L}, F={F}, heads={h}, "
                   f"kv_heads={kv}, head_dim={hd}, tied={bool(_g(c, 'tie_word_embeddings'))}")
    if mt == "llava":
        t, v = _g(c, "text_config") or {}, _g(c, "vision_config") or {}
        d, L, F, h, kv = t.get("hidden_size"), t.get("num_hidden_layers"), t.get("intermediate_size"), t.get("num_attention_heads"), t.get("num_key_value_heads")
        lm = V * d + L * (4 * d * d + 3 * d * F + 2 * d) + d + (0 if _g(c, "tie_word_embeddings") else V * d)
        vd, vL, vF, ps, im = v.get("hidden_size"), v.get("num_hidden_layers"), v.get("intermediate_size"), v.get("patch_size"), v.get("image_size")
        npos = (im // ps) ** 2 + 1
        vis = vd + 3 * ps * ps * vd + npos * vd + 2 * vd + vL * (4 * (vd * vd + vd) + vd * vF + vF + vF * vd + vd + 4 * vd) + 2 * vd
        proj = vd * d + d + d * d + d
        return lm + vis + proj, (f"LLaVA: Llama text model ({lm:,}; V={V}, d={d}, L={L}, F={F}, untied head) + CLIP vision tower ({vis:,}; "
                                 f"d={vd}, L={vL}, F={vF}, {npos} positions) + 2-layer projector ({proj:,})")
    return None, f"no parameter formula for model_type {mt!r}"


def head_params(stem, c, base):
    """Parameter count of the model the script actually trains, from the causal-LM count."""
    mt = _g(c, "model_type")
    d = _g(c, "hidden_size", "n_embd")
    if stem in ("reward_modeling", "rm"):
        # AutoModelForSequenceClassification, num_labels=1: the base model without its LM head, plus a score layer
        if mt == "gpt_neox":
            return base - _g(c, "vocab_size") * d + d, "sequence-classification head: the untied embed_out is dropped and a score layer d x 1 added"
        if mt == "opt":
            W = _g(c, "word_embed_proj_dim") or d
            return base + W, f"sequence-classification head: score layer {W} x 1 without bias; the tied lm_head costs nothing to drop"
        return base + d, "sequence-classification head: score layer d x 1"
    if stem == "stepwise_reward_modeling":
        return base + 2 * d + 2, "token-classification head: 2 labels, d x 2 with bias; the tied lm_head costs nothing to drop"
    return base, "causal LM: the script trains the model as loaded"


def lora_params(c, r, targets):
    """Trainable LoRA parameters: r x (in + out) per targeted Linear, over every layer that holds it."""
    mt = _g(c, "model_type")
    mods = {}
    if mt == "gemma2":
        d, L, F, h, kv, hd = (_g(c, "hidden_size"), _g(c, "num_hidden_layers"), _g(c, "intermediate_size"),
                              _g(c, "num_attention_heads"), _g(c, "num_key_value_heads"), _g(c, "head_dim"))
        per = {"q_proj": (d, h * hd), "k_proj": (d, kv * hd), "v_proj": (d, kv * hd), "o_proj": (h * hd, d),
               "gate_proj": (d, F), "up_proj": (d, F), "down_proj": (F, d)}
        mods = {k: (v, L) for k, v in per.items()}
        mods["lm_head"] = ((d, _g(c, "vocab_size")), 1)
    elif mt == "llava":
        t, v = _g(c, "text_config") or {}, _g(c, "vision_config") or {}
        d, L, F = t["hidden_size"], t["num_hidden_layers"], t["intermediate_size"]
        vd, vL, vF = v["hidden_size"], v["num_hidden_layers"], v["intermediate_size"]
        mods = {"q_proj": ((d, d), L), "k_proj": ((d, d), L), "v_proj": ((d, d), L), "o_proj": ((d, d), L),
                "gate_proj": ((d, F), L), "up_proj": ((d, F), L), "down_proj": ((F, d), L),
                "lm_head": ((d, _g(c, "vocab_size")), 1),
                "fc1": ((vd, vF), vL), "fc2": ((vF, vd), vL), "out_proj": ((vd, vd), vL),
                "linear_1": ((vd, d), 1), "linear_2": ((d, d), 1)}
        # CLIP's attention projections share the names q_proj/k_proj/v_proj with the text model
        for k in ("q_proj", "k_proj", "v_proj"):
            mods[k + "@vision"] = ((vd, vd), vL)
    else:
        return None, f"no module table for model_type {mt!r}"
    n, used = 0, []
    for name, ((i, o), count) in mods.items():
        base = name.split("@")[0]
        if base in targets:
            n += r * (i + o) * count
            used.append(f"{name}:{count}x{r}x({i}+{o})")
    missing = [t for t in targets if t not in {k.split("@")[0] for k in mods}]
    return n, "sum of r(in+out) over " + ", ".join(used) + (f"; targets without a module table: {missing}" if missing else "")
