import time
from functools import wraps

import torch

from ERNIE45_03B_Base.modeling_ernie4_5 import Ernie4_5_ForCausalLM
from ERNIE45_03B_Base.tokenization_ernie4_5 import Ernie4_5_Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "/share/project/hcr/models/Qwen/Qwen3-0.6B"
PRINT_TIME = True
TEST_NUM = 1
PRINT_LOGITS = False
PRINT_TOKENS_PER_LAYER = False
PRINT_OUTPUT_TOKEN_IDS = True
NEW_MAX_TOKENS = 32
USE_CACHE = True
TOPK = 32
PAD_LEN = 4

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
    # model = Ernie4_5_ForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = model.cuda()
    model.eval()
    return model


@timer_decorator
def load_tokenizer():
    # tokenizer = Ernie4_5_Tokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
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

    out_first_token_last_layer = out["hidden_states"][0][-1]
    logits = model.lm_head(out_first_token_last_layer)
    indices = logits.topk(1).indices
    bs_indices = indices.reshape(indices.shape[:2])
    print(f"bs_indices: {bs_indices}")

    topk_value, topk_pos = [], []
    for i_tokens, token_hidden_state in enumerate(hidden_states):
        if PRINT_TOKENS_PER_LAYER:
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
            if PRINT_TOKENS_PER_LAYER:
                print(f"layer-{i_layer:2}: {hs_topk.indices.tolist()}\n    {hs_topk_vs}")
    return generated_ids


@timer_decorator
def generate_chunked(model, tokenizer):
    prompt = ["Large language model is"]
    prompt[0] += "<|fim_pad|>" * PAD_LEN
    model_inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt", padding=True, truncation=True).to(model.device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    print(f"input_ids: {input_ids}")
    print(f"attention_mask: {attention_mask}")

    out = model.generate(model_inputs.input_ids, max_new_tokens=8, do_sample=False, output_scores=True,
        return_dict_in_generate=True, output_hidden_states=True, use_cache=USE_CACHE,)
    
    # out: TUPLE(out_s, n_layer+1) [b, s, h]
    out_first_token_last_layer = out.hidden_states[0][-1]
    logits = model.lm_head(out_first_token_last_layer)
    indices = logits.topk(4).indices
    # bs_indices = indices.reshape(indices.shape[:2])
    bs_indices = indices
    print(f"bs_indices: {bs_indices}")


    slen = len(input_ids[0])
    attention_mask = torch.zeros(slen, slen)
    for i in range(slen):
        for j in range(slen):
            if i >= j or j < slen - PAD_LEN:
                attention_mask[i, j] = 1
    model_inputs["attention_mask"] = attention_mask
    # print(f"attention_mask: {attention_mask}")
    out = model.generate(model_inputs.input_ids, max_new_tokens=8, do_sample=False, output_scores=True,
        return_dict_in_generate=True, output_hidden_states=True, use_cache=USE_CACHE,)
    out_first_token_last_layer = out.hidden_states[0][-1]
    logits = model.lm_head(out_first_token_last_layer)
    indices = logits.topk(4).indices
    bs_indices = indices
    # bs_indices = indices.reshape(indices.shape[:2])
    print(f"bs_indices: {bs_indices}")


    slen = len(input_ids[0])
    attention_mask = torch.zeros(slen, slen)
    for i in range(slen):
        for j in range(slen):
            if j < slen - PAD_LEN:
                attention_mask[i, j] = 1
    model_inputs["attention_mask"] = attention_mask
    # print(f"attention_mask: {attention_mask}")
    out = model.generate(model_inputs.input_ids, max_new_tokens=8, do_sample=False, output_scores=True,
        return_dict_in_generate=True, output_hidden_states=True, use_cache=USE_CACHE,)
    out_first_token_last_layer = out.hidden_states[0][-1]
    logits = model.lm_head(out_first_token_last_layer)
    indices = logits.topk(4).indices
    bs_indices = indices
    # bs_indices = indices.reshape(indices.shape[:2])
    print(f"bs_indices: {bs_indices}")

    generated_ids = out["sequences"]
    import pdb; pdb.set_trace()
    
    return generated_ids
    

@timer_decorator
def decode(generated_ids, tokenizer):
    output_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    return output_text


@timer_decorator
def infer():
    model = load_model()
    tokenizer = load_tokenizer()
    for i in range(TEST_NUM):
        print("-" * 80)
        print(f"Origin Test {i + 1}/{TEST_NUM}")

        generated_ids = generate(model, tokenizer)
        output_text = decode(generated_ids, tokenizer)
        if PRINT_OUTPUT_TOKEN_IDS:
            print(f"generated_ids: {generated_ids.shape} {generated_ids}")
        print(f"output_text  : {output_text}")

    for i in range(TEST_NUM):
        print("-" * 80)
        print(f"Chunked Test {i + 1}/{TEST_NUM}")

        generated_ids = generate_chunked(model, tokenizer)
        output_text = decode(generated_ids, tokenizer)
        if PRINT_OUTPUT_TOKEN_IDS:
            print(f"generated_ids: {generated_ids.shape} {generated_ids}")
        print(f"output_text  : {output_text}")


if __name__ == "__main__":
    with torch.no_grad():
        infer()
