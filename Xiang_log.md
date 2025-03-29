
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

 AssertionError: Input and cos/sin must have the same dtype, got torch.float32 and torch.bfloat16

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

QWen2-VL-2B-SFT, 2k
48.60

37829786, QWen2-VL-2B-SFT+RL
50.40%


-----------

37890516, QWen2.5-VL-3B-SFT, epoch=10
Accuracy: 47.60%

37890511, QWen2.5-VL-3B-SFT
Accuracy: 52.20%
GPT-EVAL: 60%
Accuracy: 46.82%

37890512, Qwen2.5-VL-3B-v2
Accuracy: 7.60%

37899099, VRSBench_Qwen2.5-VL-3B-v3

Qwen2.5-VL-3B-vllm-v2,

37901153/37908012, Qwen2.5-VL-3B-v2, bs=2, acc_grad=4, 
Accuracy: 7.60%

37900564/37908024, Qwen2.5-VL-3B-v3, bs=2, acc_grad=4, learning_rate: 2.0e-05, warmup_ratio=0.1

37900569, Qwen2.5-VL-7B, bs=1, acc_grad=4, learning_rate: 2.0e-05, warmup_ratio=0.1
Accuracy: 0.20%

----------------
37908117, Qwen2.5-VL-3B, outputs_num/VRSBench_Qwen2.5-VL-3B-SFT
Accuracy: 47.69%

37908332, Qwen2-VL-2B, outputs_num/VRSBench_Qwen2-VL-2B
Accuracy: 40.85%

-----------------
37920822, Dota-num-VRSBench_Qwen2-VL-2B, tested on DIOR
Accuracy: 48.97%

37920825,  Dota-num-VRSBench_Qwen2-VL-2B-SFT, tested on DIOR
Accuracy: 54.72%

