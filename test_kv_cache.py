import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def test_kv_cache():
    # Initialize model and tokenizer
    model_name = "meta-llama/Llama-2-7b-chat-hf"  # Replace with your model
    model = LLM(model=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create a long context that we'll reuse
    base_context = "Here is a long context that we'll use to test KV caching. " * 10
    base_tokens = tokenizer.encode(base_context)
    
    # Create different prompts that share the same prefix
    prompts = [
        base_context + "What is the capital of France?",
        base_context + "What is the capital of Germany?",
        base_context + "What is the capital of Italy?"
    ]
    
    # Test without KV cache
    print("\nTesting without KV cache:")
    start_time = time.time()
    outputs_no_cache = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        sampling_params = SamplingParams(max_tokens=50)
        output = model.generate(prompt_token_ids=[tokens], sampling_params=sampling_params)
        outputs_no_cache.append(output)
    time_without_cache = time.time() - start_time
    print(f"Time without KV cache: {time_without_cache:.2f} seconds")
    
    # Test with KV cache
    print("\nTesting with KV cache:")
    start_time = time.time()
    outputs_with_cache = []
    kv_cache = None
    
    # First generate with base context to get initial KV cache
    base_tokens = tokenizer.encode(base_context)
    sampling_params = SamplingParams(max_tokens=1)  # We only need 1 token to get the cache
    initial_output = model.generate(prompt_token_ids=[base_tokens], sampling_params=sampling_params)
    kv_cache = initial_output.kv_cache
    
    # Now generate with different prompts using the KV cache
    for prompt in prompts:
        # Only encode the new part of the prompt
        new_prompt = prompt[len(base_context):]
        new_tokens = tokenizer.encode(new_prompt)
        
        sampling_params = SamplingParams(max_tokens=50)
        output = model.generate(
            prompt_token_ids=[new_tokens],
            sampling_params=sampling_params,
            kv_cache=kv_cache
        )
        outputs_with_cache.append(output)
        kv_cache = output.kv_cache  # Update cache for next iteration
    
    time_with_cache = time.time() - start_time
    print(f"Time with KV cache: {time_with_cache:.2f} seconds")
    
    # Compare results
    print("\nResults comparison:")
    print(f"Time saved: {time_without_cache - time_with_cache:.2f} seconds")
    print(f"Speedup: {time_without_cache/time_with_cache:.2f}x")
    
    # Verify outputs are similar
    print("\nVerifying outputs:")
    for i, (output_no_cache, output_with_cache) in enumerate(zip(outputs_no_cache, outputs_with_cache)):
        text_no_cache = output_no_cache[0].outputs[0].text
        text_with_cache = output_with_cache[0].outputs[0].text
        print(f"\nPrompt {i+1}:")
        print(f"Without cache: {text_no_cache}")
        print(f"With cache: {text_with_cache}")
        print(f"Outputs match: {text_no_cache == text_with_cache}")

if __name__ == "__main__":
    test_kv_cache() 