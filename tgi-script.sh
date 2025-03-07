model=Qwen/Qwen2.5-14B-Instruct-1M
volume=~/.cache/huggingface/hub # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:3.1.1 \
    --model-id $model