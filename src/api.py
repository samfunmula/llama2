from fastapi import FastAPI
import uvicorn
from lib import *
from accelerate import infer_auto_device_map, init_empty_weights
from pprint import pprint

app = FastAPI()

@app.post("/")
async def predict(request: dict) :
    input_text = request.get('input')
    history = request.get('history',[])
    max_new_tokens = request.get('max_length',512)
    top_p = request.get('top_p',0.9)
    temperature = request.get('temperature',40)

    print("Now generating response.\n...")
    start = time.time()

    output = chat_with_model(input_text, history, max_new_tokens, temperature, top_p)
    end = time.time()

    print("Generate response takes %f seconds" % (end - start))
    print(f"##\ninput : {output[-1][0]}")
    print(f"result : {output[-1][1]}")
    return {"history" : output }, {"input" : output[-1][0] , "result" : output[-1][1]}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8050, workers=1)