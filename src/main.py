from fastapi import FastAPI, Request, Response
import uvicorn, json
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM , AutoTokenizer
import nvidia_smi
import time

app = FastAPI()

@app.post("/")
async def predict(request: dict) :
    input_text = request.get('input')
    history = request['history']
    history = history if isinstance(history, list) else []
    max_new_tokens = request.get('max_length',512)
    top_p = request.get('top_p',0.9)
    temperature = request.get('temperature',40)

    print("chat_with_model")
    start = time.time()

    output = chat_with_model(input_text, history, max_new_tokens, temperature, top_p)
    end = time.time()

    # 輸出結果
    print("chat_with_model takes %f seconds" % (end - start))
    #return history
    return output

def chat_with_model(input_text, history, max_new_tokens, temperature, top_p):
    global MODEL, TOKENIZER

    original_input = input_text
    # record history
    print('## Now processing input_text ##')
    start = time.time()
    max_memory = 500
    if len(history) != 0:
        input_text = "".join(["### Instruction:\n" + i[0] +"\n\n" + "### Response: " + i[1] + "\n\n" for i in history]) + "### Instruction:\n" + input_text
        input_text = input_text[len("### Instruction:\n"):]
        if len(input_text) > max_memory:
            input_text = input_text[-max_memory:]

    prompt = generate_prompt(input_text)
    end = time.time()
    print("process input takes %f seconds" % (end - start))

    input_ids = TOKENIZER.encode(prompt, return_tensors='pt')
    with torch.no_grad():
        output_ids = MODEL.generate(
            input_ids=input_ids,
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
            

def get_available_gpu_with_most_memory() -> list[str]:
    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    devices = []
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        free_m = mem.free/1024**2
        devices.append((i, free_m))
    devices.sort(key=lambda x: x[1], reverse=True)
    return [i[0] for i in devices]


if __name__ == '__main__':
    MODEL_REVISION_VERSION = 'v1.0'
    DEVICE = ''
    if torch.cuda.is_available():
        DEVICE_LIST = get_available_gpu_with_most_memory()
        DEVICE = f'cuda:{DEVICE_LIST[1]}'
        print('[device] device_list:', DEVICE_LIST)
    else:
        DEVICE = 'cpu'
        print('[device] use cpu')
    print("Loading Model")
    start = time.time()
    TOKENIZER = LlamaTokenizer.from_pretrained("/home/sam/hub/Llama-2-13b-chat-hf", local_files_only=True, trust_remote_code=True,torch_dtype = torch.float16)
    MODEL = LlamaForCausalLM.from_pretrained('/home/sam/hub/Llama-2-13b-chat-hf', local_files_only=True, trust_remote_code=True,
                                            low_cpu_mem_usage=True, device_map='auto',torch_dtype = torch.float16)
    
    end = time.time()

    # 輸出結果
    print("執行時間：%f 秒" % (end - start))
    MODEL.eval()
        
    uvicorn.run(app, host='0.0.0.0', port=8050, workers=1)