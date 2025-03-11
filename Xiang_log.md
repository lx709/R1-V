
rft-37797928.out

rft-37798085.out


rft-37806655.out
QWen2-VL-2B, VQA, acc=44.4

rft-37808155.out
QWen2-VL-2B, REF, acc=

rft-37811856.out
Qwen2.5-VL-3B, VQA

rft-37811864.out
Qwen2.5-VL-3B, REF

torch type error
update transformer

[Bug]: Qwen2.5-VL broke due to transformers upstream changes #13285

pip install --upgrade git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef

rft-37811967.out
Qwen2.5-VL-3B, VQA

rft-37811968.out
Qwen2.5-VL-3B, VQA, vLLM

rft-37816723.out
Qwen2.5-VL-3B, VQA


37816812, QWen2-VL-2B, 1k

37816813, Qwen2.5-VL-3B, 1k
acc=7.2

37816814, Qwen2.5-VL-3B-vLLM, VQA
acc=42.8

37816879, QWen2-VL-2B-v2, 2k
acc=44.4

37816934, QWen2-VL-2B-v3, 1k
acc=37.4

-----------


Qwen2.5-VL-3B, 2k, constant lr, max_prompt_length=4096 (not 512)

QWen2-VL-2B-SFT, 2k
48.60

37829786, QWen2-VL-2B-SFT+RL
50.40%

37864431, QWen2.5-VL-3B-SFT, 2k

37864554, QWen2.5-VL-3B-SFT, 2k, epoch=5

Qwen2.5-VL-3B-v2, 2k


