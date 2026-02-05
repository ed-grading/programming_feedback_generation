# LLM Feedback Generation and Evaluation Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation for **"From Code to Rubrics: A Multimodal Evaluation of LLM Systems for Automated Feedback Generation"**.

## 🎯 Overview

This codebase provides a complete pipeline for:
- **Feedback Generation**: Comparing unified vs. multi-agent approaches for bug detection and feedback generation
- **Evaluation**: Implementing J1-style thinking judges for improved automated evaluation
- **Reproducibility**: All experimental conditions and data processing from our paper

## 📋 Table of Contents

- [Setup](#setup)
- [Dataset](#dataset)
- [Feedback Generation](#feedback-generation)
- [Judge Evaluation](#judge-evaluation)
- [File Structure](#file-structure)

## 🚀 Setup

### Prerequisites

- **GPU**: NVIDIA GPU with 32GB+ VRAM (A100 40GB or A100 80GB recommended)
- **Platform**: RunPod or similar GPU cloud provider
- **Python**: 3.8 or higher

### Environment Setup

1. **Launch RunPod Instance**
   ```bash
   # Recommended: A100 80GB with PyTorch template
   # Or use custom container: runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04
   ```

2. **Clone Repository**
   ```bash
   git clone https://github.com/ed-grading/programming_feedback_generation
   cd programming_feedback_generation
   ```
3. **Create Virtual Environment**
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Model Setup with vLLM

1. **Install vLLM**
   ```bash
   pip install vllm
   ```

2. **Start vLLM Server**
   ```bash
   # For 32B models (requires 80GB VRAM)
   vllm serve Qwen/Qwen2.5-Coder-32B-Instruct \
     --max-model-len 8192 \
     --gpu-memory-utilization 0.9 \
     --port 8000

   # For 14B models (requires 32GB VRAM)
   vllm serve Qwen/Qwen2.5-Coder-14B-Instruct \
     --max-model-len 4096 \
     --gpu-memory-utilization 0.9 \
     --port 8000
   ```

3. **Verify Server**
   ```bash
   curl http://localhost:8000/v1/models
   ```
    Should return JSON with model info

## 📊 Dataset

### Socratic Benchmark

We use the Socratic Benchmark for Python programming problems:

1. **Download Dataset**
   ```bash
   # Clone the Socratic Benchmark repository
   git clone https://github.com/taisazero/socratic-debugging-benchmark
   ```

2. **Dataset Locations**
   
   This repository uses the Socratic Debugging Benchmark. Two versions are available:
```bash
   # Version 1 (BEA@ACL'24)
   socratic_debugging_benchmark/v1_bea/final_dataset
   
   # Version 2 ( SIGCSE'24) 
   socratic_debugging_benchmark/v2_sigcse/final_dataset
```
   
   **Expected Format:**
```
   Each problem file contains:
   <problem>...</problem>
   <bug_code>...</bug_code>
   <unit_tests>...</unit_tests>
   ```

## 🔧 Feedback Generation

### Quick Start

1. **Configure Your LLM Connection**
   ```python
   # Edit feedback_script.py - Update these lines:
   llm = ChatOpenAI(
       model="Qwen/Qwen2.5-Coder-32B-Instruct",  # Match your vLLM model
       base_url="http://localhost:8000/v1",
       api_key="EMPTY",
       temperature=0.1
   )
   ```

2. **Run Feedback Generation**
   ```bash
   # Generate feedback with both approaches
   python feedback_script.py
   ```

### Configuration Options

**Model Selection:**
- `Qwen/Qwen2.5-Coder-14B-Instruct` (requires 32GB VRAM)
- `Qwen/Qwen2.5-Coder-32B-Instruct` (requires 80GB VRAM)
- `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` (16B, requires 32GB VRAM)

**Generation Approaches:**
- **Unified**: Single prompt for complete feedback
- **Multi-Agent**: Sequential bug detection → feedback generation

### Expected Output

```
generated_feedback/
├── model_name/
│   ├── unified/
│   │   ├── problem_001_feedback.txt
│   │   └── ...
│   └── multi_agent/
│       ├── problem_001_feedback.txt
│       └── ...
```

## ⚖️ Judge Evaluation

### Setup Judge Model

1. **Configure Judge in evaluation_script.py**
   ```python
   # Use your best model as judge (typically 32B)
   judge_llm = ChatOpenAI(
       model="Qwen/Qwen2.5-Coder-32B-Instruct",
       base_url="http://localhost:8000/v1",
       api_key="EMPTY",
       temperature=0.1
   )
   ```

2. **Run Evaluation**
   ```bash
   # Evaluate all generated feedback
   python evaluation_script.py
   ```

### Evaluation Approaches

**Standard Judge:**
- Direct evaluation without explicit reasoning
- Baseline performance

**J1-Thinking Judge:**  
- Chain-of-thought reasoning before evaluation
- Improved human agreement (our main contribution)

### Evaluation Criteria

| Criterion | Code | Description |
|-----------|------|-------------|
| Explanation Accurate | EA | Correctly identifies true bugs |
| Explanation Selective | ES | No false positives |
| Explanation Clear | EC | Understandable for novices |
| Fixes Accurate | FA | Fixes would work correctly |
| Fixes Selective | FS | No unnecessary modifications |
| Fixes Clear | FC | Specific and actionable |


## 📁 File Structure

```
llm-feedback-evaluation/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── feedback_script.py           # Main feedback generation pipeline
├── evaluation_script.py         # Judge evaluation pipeline  
├── unified_prompt.py           # Unified approach prompts
├── multi_agents_prompt.py      # Multi-agent approach prompts
├── single_judge_prompt.py      # Standard judge prompts
├── j1_judge_prompt.py          # J1-thinking judge prompts
└── grading_criteria.py         # Evaluation criteria definitions
```

## 🔧 Troubleshooting

### Common Issues

**vLLM Server Won't Start:**
```bash
# Check GPU memory
nvidia-smi

# Reduce memory utilization
vllm serve model-name --gpu-memory-utilization 0.8
```

**CUDA Out of Memory:**
```bash
# Use smaller model or reduce context length
vllm serve model-name --max-model-len 4096
```

**Connection Refused:**
```bash
# Check if vLLM server is running
curl http://localhost:8000/health

# Restart server if needed
pkill -f vllm
vllm serve your-model --port 8000
```

### RunPod Specific Tips

1. **Use Persistent Storage**: Attach network volume to avoid losing data
2. **Monitor GPU Usage**: Use `nvidia-smi` to track memory
3. **Save Progress**: Results are automatically saved to avoid loss
4. **Multiple Models**: Stop/start vLLM server to switch models


## 📧 Support

For issues with:
- **Code implementation**: Create issue with `[code]` tag
- **Reproduction**: Create issue with `[reproduction]` tag  
- **Dataset**: Check [Socratic Benchmark repository](https://github.com/socratic-dev/socratic-benchmark)



## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---