# Fine-Tuning LLaMA 3.1 8B-Instruct for Retrieval-Augmented Generation (RAG)

This repository presents a supervised fine-tuning (SFT) workflow for enhancing large language models (LLMs) in Retrieval-Augmented Generation (RAG) tasks using the **RAG-Instruct** dataset. The model fine-tuned in this project is **Meta-LLaMA 3.1 8B-Instruct**, and the training was conducted on a multi-GPU setup.

---

## Dataset

**RAG-Instruct** is a synthetic, high-quality dataset created using GPT-4o. It is based on the Wikipedia corpus and includes **40,000** QA pairs.

Each entry in the dataset includes:

- Instruction
- Retrieved context documents (positive and distractor documents)
- Ground truth answer

---

## Training Setup

- **Hardware**: 4 × A800 GPUs (80GB RAM each)
- **Duration**: \~24 hours
- **Parallelization**: DeepSpeed ZeRO-2
- **Precision**: FP16

### Training Configuration

- Framework: Hugging Face Transformers
- Optimizer: AdamW
- Scheduler: Cosine with warmup
- Batch size: per-GPU and gradient accumulation adjusted to memory limit

---

## How to Use

1. Clone the repo
2. Download the RAG-Instruct dataset and place in `./train_data`
3. Run training:

```bash
bash train.sh
```

---

## Next Steps

- Evaluate the fine-tuned model on standard RAG benchmarks (HotpotQA, NaturalQuestions, etc.)
- Compare to base LLaMA3.1-Instruct model in open/closed-book QA
- Explore quantization for deployment (GGUF, GPTQ, etc.)
- Add streaming RAG with LLM + retriever pipeline

---

## License

This repo is based on open-source LLaMA 3.1 and the open RAG-Instruct dataset. See `LICENSE` for details.

---
