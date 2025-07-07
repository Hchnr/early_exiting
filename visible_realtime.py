import time
from functools import wraps

import torch

from ERNIE45_03B_Base.modeling_ernie4_5 import Ernie4_5_ForCausalLM
from ERNIE45_03B_Base.tokenization_ernie4_5 import Ernie4_5_Tokenizer


MODEL_PATH = "/share/project/hcr/early_exiting/ERNIE45_03B_Base"
PRINT_TIME = False
TEST_NUM = 1
PRINT_LOGITS = False
PRINT_OUTPUT_TOKEN_IDS = False
NEW_MAX_TOKENS = 32
USE_CACHE = True
DO_ORIGIN_INFER = True
DO_SWAP_INFER = False
TOPK = 2

def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not PRINT_TIME:
            return func(*args, **kwargs)
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"=> {func.__name__:16} cost: {execution_time:.3f} s")
        return result

    return wrapper


@timer_decorator
def load_model():
    model = Ernie4_5_ForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = model.cuda()
    model.eval()
    return model


@timer_decorator
def load_tokenizer():
    tokenizer = Ernie4_5_Tokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return tokenizer


@timer_decorator
def generate(model, tokenizer):
    prompt = ["Large language model is"]
    model_inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    out = model.generate(
        model_inputs.input_ids,
        max_new_tokens=NEW_MAX_TOKENS,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
        output_hidden_states=True,
        use_cache=USE_CACHE,
    )
    generated_ids = out["sequences"]
    if PRINT_LOGITS:
        logits = out["scores"]
        for i, logit in enumerate(logits):
            print(f"logits[{i}].shape: {logit.shape}")
            print(f"logits.topk(8): {logit.topk(8, dim=-1)[0].squeeze().tolist()}")
    hidden_states = out["hidden_states"]
    topk_value, topk_pos = [], []
    for i_tokens, token_hidden_state in enumerate(hidden_states):
        print("-" * 80)
        print(f"token-{i_tokens}")
        out = {}
        for i_layer, layer_hidden_state in enumerate(token_hidden_state):
            # layer_hidden_state: [b, num_token, h] -> [1, 4, 1024]
            last_token_hidden_state = layer_hidden_state[:,-1,:]
            # out_logits = model.lm_head(last_token_hidden_state)
            hs_topk = last_token_hidden_state.topk(TOPK, dim=-1, sorted=True)
            hs_topk_v = hs_topk.values.tolist()
            hs_topk_vs = [float(f"{v:.2f}") for v in hs_topk_v[0]]
            print(f"layer-{i_layer:2}: {hs_topk.indices.tolist()}\n    {hs_topk_vs}")
            # import pdb; pdb.set_trace()

    return generated_ids


@timer_decorator
def decode(generated_ids, tokenizer):
    output_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    return output_text


@timer_decorator
def channel_swap_after_attn(model):
    tmp = model.model.layers[1].mlp.up_proj.weight
    tmp[:, 0], tmp[:, 1] = tmp[:, 1].clone(), tmp[:, 0].clone()
    
    tmp = model.model.layers[1].mlp.gate_proj.weight
    tmp[:, 0], tmp[:, 1] = tmp[:, 1].clone(), tmp[:, 0].clone()
    tmp = model.model.layers[1].self_attn.o_proj.weight
    tmp[0, :], tmp[1, :] = tmp[1, :].clone(), tmp[0, :].clone()


@timer_decorator
def channel_swap_between_layers(model):
    h = 1024
    tmp = model.model.layers[1].input_layernorm.weight
    assert tmp.shape[0] == h
    tmp[0], tmp[1] = tmp[1].clone(), tmp[0].clone()

    tmp = model.model.layers[1].self_attn.q_proj.weight
    assert tmp.shape[1] == h
    tmp[:, 0], tmp[:, 1] = tmp[:, 1].clone(), tmp[:, 0].clone()
    tmp = model.model.layers[1].self_attn.k_proj.weight
    assert tmp.shape[1] == h
    tmp[:, 0], tmp[:, 1] = tmp[:, 1].clone(), tmp[:, 0].clone()
    tmp = model.model.layers[1].self_attn.v_proj.weight
    assert tmp.shape[1] == h
    tmp[:, 0], tmp[:, 1] = tmp[:, 1].clone(), tmp[:, 0].clone()
    tmp = model.model.layers[1].self_attn.o_proj.weight
    assert tmp.shape[0] == h
    tmp[0, :], tmp[1, :] = tmp[1, :].clone(), tmp[0, :].clone()

    tmp = model.model.layers[1].post_attention_layernorm.weight
    assert tmp.shape[0] == h
    tmp[0], tmp[1] = tmp[1].clone(), tmp[0].clone()
    tmp = model.model.layers[1].mlp.gate_proj.weight
    assert tmp.shape[1] == h
    tmp[:, 0], tmp[:, 1] = tmp[:, 1].clone(), tmp[:, 0].clone()
    tmp = model.model.layers[1].mlp.up_proj.weight
    assert tmp.shape[1] == h
    tmp[:, 0], tmp[:, 1] = tmp[:, 1].clone(), tmp[:, 0].clone()
    tmp = model.model.layers[1].mlp.down_proj.weight
    assert tmp.shape[0] == h
    tmp[0, :], tmp[1, :] = tmp[1, :].clone(), tmp[0, :].clone()


@timer_decorator
def infer():
    model = load_model()
    tokenizer = load_tokenizer()
    if DO_ORIGIN_INFER:
        for i in range(TEST_NUM):
            print("-" * 80)
            print(f"Origin Test {i + 1}/{TEST_NUM}")

            generated_ids = generate(model, tokenizer)
            output_text = decode(generated_ids, tokenizer)
            if PRINT_OUTPUT_TOKEN_IDS:
                print(f"generated_ids: {generated_ids.shape} {generated_ids}")
            print(f"output_text  : {output_text}")

    channel_swap_between_layers(model)
    # channel_swap_after_attn(model)

    if DO_SWAP_INFER:
        for i in range(TEST_NUM):
            print("-" * 80)
            print(f"Swap Test {i + 1}/{TEST_NUM}")
            generated_ids = generate(model, tokenizer)
            output_text = decode(generated_ids, tokenizer)
            if PRINT_OUTPUT_TOKEN_IDS:
                print(f"generated_ids: {generated_ids.shape} {generated_ids}")
            print(f"output_text  : {output_text}")


if __name__ == "__main__":
    with torch.no_grad():
        infer()
