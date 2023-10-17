docker build -t llama2_api .
docker run -it -v /home/sam/hub/Llama-2-7b-chat-hf:/app/hub --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -p 8050:8050 llama2_api
#docker run -it -v /home/sam/hub/Llama-2-13b-hf:/app/hub --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -p 8050:8050 llama2_api
#docker run -it -v /home/sam/hub/Llama-2-13b-chat-hf:/app/hub --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -p 8050:8050 llama2_api
