import os
gpus = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import time

def chat_with_model(input_text, history, max_new_tokens, temperature, top_p):
    global MODEL, TOKENIZER

    original_input = input_text
    # record history
    max_memory = 500
    if len(history) != 0:
        input_text = "".join(["### Instruction:" + i[0] +"\n\n" + "### Response: " + i[1] + "\n\n" for i in history]) + "### Instruction:" + input_text
        input_text = input_text[len("### Instruction:\n"):]
        if len(input_text) > max_memory:
            input_text = input_text[-max_memory:]

    prompt = generate_prompt(input_text)
    input_ids = TOKENIZER.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        output_ids = MODEL.generate(
            input_ids=input_ids.to(DEVICE),
            max_new_tokens = max_new_tokens if max_new_tokens else 500,
            temperature= temperature if temperature else 0.3,
            top_k = 50,
            top_p = top_p if top_p else 0.9,
            repetition_penalty = 1.0
        ).to(DEVICE)
    output = TOKENIZER.decode(output_ids[0], skip_special_tokens=True)
    output = output.split("### Response:")[-1].strip()
    history.append((original_input, output))
    
    torch_gc()

    return history


def generate_prompt(text):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:

{text}

### Response:"""


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
MODEL_REVISION_VERSION = 'v1.0'
DEVICE = ''

if torch.cuda.is_available():
            DEVICE = torch.device('cuda')
            device_map='auto'
else:
    DEVICE = torch.device('cpu')
    device_map={'':DEVICE}

print(f"device: {DEVICE}")
print(f"device_map: {device_map}")

print("Loading Model")
start = time.time()
TOKENIZER = LlamaTokenizer.from_pretrained("./hub", local_files_only=True, trust_remote_code=True,device_map='auto',device = DEVICE,torch_dtype = torch.float16)
MODEL = LlamaForCausalLM.from_pretrained('./hub', local_files_only=True, trust_remote_code=True,
                                        low_cpu_mem_usage=True, device_map='auto',torch_dtype = torch.float16)

end = time.time()

print("Loading Model takes：%f second" % (end - start))
MODEL.eval()