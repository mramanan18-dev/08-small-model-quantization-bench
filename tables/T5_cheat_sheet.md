| You have / want                | Use                            | Our numbers (Qwen3-0.6B)    |
|:-------------------------------|:-------------------------------|:----------------------------|
| GPU, serving with vLLM         | AWQ 4-bit                      | +12% ppl, 1.41× decode      |
| GPU, AWQ too slow to calibrate | GPTQ 4-bit                     | +20% ppl, 2.5 min vs 45 min |
| Fine-tuning on a small GPU     | NF4 (QLoRA)                    | +17% ppl, no calibration    |
| CPU / laptop                   | GGUF Q4_K_M + imatrix          | +4.9% ppl, 2.0× F16 speed   |
| Quality first, still small     | Q5_K_M or Q8_0                 | +1.9% / +0.3% ppl           |
| Must go to 3-bit               | QAT if you can train, else AWQ | +77% vs +115% ppl           |
| Never                          | Plain 3-bit rounding           | +1,317% ppl                 |