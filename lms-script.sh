# LM Studio local server launch script

model=bartowski/Qwen2.5-14B-Instruct-1M-GGUF/Qwen2.5-14B-Instruct-1M-f16.gguf

lms server start
lms, unload --all
lms load $model --gpu max --exact
