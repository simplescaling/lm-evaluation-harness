# HealthBench

## Overview

HealthBench is a comprehensive medical reasoning benchmark that evaluates language models on realistic healthcare scenarios. It uses LLM-as-judge evaluation with detailed rubrics to assess the quality, accuracy, and safety of medical advice.

**Paper:** [Introducing HealthBench: An evaluation for AI systems and human health](https://openai.com/index/healthbench/)  
**Dataset:** [openai/healthbench](https://huggingface.co/datasets/openai/healthbench)

## Implementation Details

### Evaluation Method

HealthBench uses **LLM-as-judge** evaluation where:
1. A model generates a response to a medical question
2. Each response is evaluated against multiple rubric items
3. GPT-4.1-mini judges whether each criterion is met
4. Scores are aggregated across rubric items

### Rubric Structure

Each example includes:
- **Prompt**: Medical question or scenario
- **Rubrics**: List of criteria with point values
- **Tags**: Categories (e.g., "diagnosis", "treatment", "safety")

Rubric items can have:
- **Positive points**: Desirable criteria (e.g., "recommends seeking medical attention")
- **Negative points**: Undesirable criteria (e.g., "provides dangerous advice")

### Metrics

- **overall_score**: Main metric - percentage of total possible points achieved
- **Tag-specific scores**: Scores for different medical categories
- **exact_match**: Alias for overall_score (for compatibility)

## Usage

```bash
lm_eval --model openai-chat-completions \
    --model_args model=gpt-4.1 \
    --tasks healthbench \
    --apply_chat_template \
    --gen_kwargs max_gen_toks=2048 \
    --batch_size 1 \
    --limit 3 \
    --output_path <ADD PATH> \
    --log_samples
```

## Citation

```bibtex
@article{arora2025healthbench,
  title={Healthbench: Evaluating large language models towards improved human health},
  author={Arora, Rahul K and Wei, Jason and Hicks, Rebecca Soskin and Bowman, Preston and Qui{\~n}onero-Candela, Joaquin and Tsimpourlas, Foivos and Sharman, Michael and Shah, Meghan and Vallone, Andrea and Beutel, Alex and others},
  journal={arXiv preprint arXiv:2505.08775},
  year={2025}
}
```

## Notes

- Grading model: `gpt-4.1-2024-07-18`
- Evaluation is done asynchronously for speed
- Results include detailed rubric-level feedback
