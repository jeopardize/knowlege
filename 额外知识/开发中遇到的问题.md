# 开发过程中遇到的坑

# Qwen和ds的部署形态不一致



# Kimi-k2 tokenizer多进程加载

## 问题背景：

同时会存在多个进程同时下去执行，Autokenizer.from_pretrained, 其他进程都没有什么问题，只有kimi在大集群下会出现竞态问题



