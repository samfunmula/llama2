# Visual-Chinese-LLaMA-Alpaca
### Start Fast API
```
cd src/
python3 api.py
```

### Start docker
```
bash runDocker.sh
```


## Request
### 參數介紹
####  Request
* input : query => str
* max_new_token : int => default = 512 (可以不用) => 範圍 0 ~ 1024 
* top_p : float => default = 0.9 (可以不用) => 範圍 0 ~ 1
* top_k : int => default = 40 (可以不用) => 範圍 0 ~ 100
* temperature : float => default = 0.5 => 範圍 0 ~ 1

#### history
* 作用 : 模型會根據history(過往的聊天紀錄)，來進行回答
* 格式 : list ， 預設為空 (皆須為雙引號，單引號傳不進去)




### curl
```
curl -X 'POST' \
        'http://localhost:8050/' \
        -H 'accept: application/json' \
        -H 'Content-Type: application/json' \
        -d '{
            "input" : "你好，很高興認識你",
            "history" : []
        }'
```
### Response
```
[
  {
    "history": [
      [
        "你好，很高興認識你",
        "Hello! Nice to meet you too! 😊"
      ]
    ]
  },
  {
    "input": "你好，很高興認識你",
    "result": "Hello! Nice to meet you too! 😊"
  }
]
```
