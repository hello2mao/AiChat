# AiChat

## 模型
百模大战排位：（1）[UC伯克利LLM排位赛](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) （2）[C-Eval排行榜](https://cevalbenchmark.com/static/leaderboard_zh.html)
| 模型 | 版本 | 支持进度 |
| :-- | :-- | :--: | 
| ChatGPT | GPT-4 <br> GPT-3.5-turbo <br> ChatGPT-免费版 | ♻️ <br> ♻️ <br> ✅  | 
| ChatGLM | THUDM/chatglm2-6b <br> THUDM/chatglm2-6b-int4 <br> THUDM/chatglm-6b <br> THUDM/chatglm-6b-int8 <br> THUDM/chatglm-6b-int4 <br> THUDM/chatglm-6b-int4-qe | ✅ <br> ✅ <br> ✅ <br> ✅ <br> ✅ <br> ✅| 
| Baichuan | 28 | ✅ |

## 插件支持
| 插件 | 功能 | 进度 |  
| :--: | :--: | :--: |  
| 张三 | 25 | 程序员 |  
| 李四 | 30 | 设计师 |  
| 王五 | 28 | 产品经理 |

## Run
```bash
# ubuntu 20.04
sudo apt install -y swig

conda create -n chatchat python=3.10
conda activate chatchat
pip install -r requirements.txt
python main.py
```

# Ref
https://github.com/SCIR-HI/Huatuo-Llama-Med-Chinese