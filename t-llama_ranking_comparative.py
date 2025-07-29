import json
import os
import sys
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from transformers import LlamaTokenizer, AutoModelForCausalLM

# Initialize T-LLaMA
model_path = "/ssd11/other/meiyy02/code_files/mt-corpus/T-LLaMA/T-LLaMA"
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

def get_comparative_prompt(source, translations, model_names):
    prompt = f"""Compare and score these {len(translations)} translations of the same source text:
    
Source: {source}

"""
    for i, (trans, name) in enumerate(zip(translations, model_names)):
        prompt += f"Translation {chr(65+i)} ({name}): {trans}\n\n"

    prompt += """Evaluate each translation on these criteria (1-5 scale):
1. Accuracy (faithfulness to source)
2. Fluency (naturalness in target language)
3. Style (appropriate tone/register)
4. Grammar (technical correctness)

SCORING KEY:
5 = Excellent, 4 = Good, 3 = Adequate, 2 = Poor, 1 = Very poor

Provide ONLY the scores in this exact format:
Translation A: [score]
Translation B: [score]
Translation C: [score]"""
    return prompt

def parse_scores(response_text, num_models):
    scores = {}
    patterns = [
        r'Translation ([A-Z]):\s*(\d)',
        r'([A-Z]):\s*(\d)',
        r'Translation ([A-Z])\s*\[(\d)\]'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response_text)
        if len(matches) >= num_models:
            for m in matches:
                scores[m[0]] = int(m[1])
            break
    
    # Fallback: find all numbers in order
    if len(scores) < num_models:
        numbers = re.findall(r'\d', response_text)
        if len(numbers) >= num_models:
            for i in range(num_models):
                scores[chr(65+i)] = int(numbers[i])
    
    # Validate scores
    valid_scores = {}
    for k, v in scores.items():
        if 1 <= v <= 5:
            valid_scores[k] = v
    
    return valid_scores

def score_translations(js_data):
    source = js_data['source']
    translations = js_data['translations']
    model_names = js_data['model_names']
    
    prompt = get_comparative_prompt(source, translations, model_names)
    inputs = tokenizer(prompt, return_tensors="pt")
    
    try:
        outputs = model.generate(
            **inputs,
            max_length=512,
            temperature=0.3,
            top_p=0.9,
            num_return_sequences=1
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        scores = parse_scores(response, len(translations))
        if len(scores) == len(translations):
            # Map scores back to original order
            ordered_scores = [scores.get(chr(65+i), 0) for i in range(len(translations))]
            return {
                'id': js_data['id'],
                'scores': ordered_scores,
                'raw_response': response,
                'status': 'success'
            }
        else:
            return {
                'id': js_data['id'],
                'scores': [0]*len(translations),
                'raw_response': response,
                'status': f'parse_error (got {len(scores)}/{len(translations)} scores)'
            }
    except Exception as e:
        return {
            'id': js_data['id'],
            'scores': [0]*len(translations),
            'raw_response': str(e),
            'status': 'generation_error'
        }

def main(src_file, *tgt_files):
    """
    usage:
    python3 T-LLaMA/t-llama_ranking_comparative.py tbt-cn-200/src_clean.txt tbt-cn-200/mt-hyps/hyp_deepseek-v3 tbt-cn-200/mt-hyps/hyp_google-translate tbt-cn-200/mt-hyps/hyp_qwen2.5_72b
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 T-LLaMA/t-llama_ranking_comparative.py tbt-cn-200/src_clean.txt tbt-cn-200/mt-hyps/hyp_deepseek-v3 tbt-cn-200/mt-hyps/hyp_google-translate tbt-cn-200/mt-hyps/hyp_qwen2.5_72b
    """
    # Load data
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = [line.strip() for line in f]
    
    model_names = [os.path.basename(f).replace('hyp_', '') for f in tgt_files]
    tgt_lines = []
    
    for f in tgt_files:
        with open(f, 'r', encoding='utf-8') as fin:
            tgt_lines.append([line.strip() for line in fin])
    
    # Verify lengths
    num_lines = len(src_lines)
    for lines in tgt_lines:
        if len(lines) != num_lines:
            print("Error: All files must have same number of lines")
            sys.exit(1)
    
    # Prepare jobs
    jobs = []
    for i in range(num_lines):
        jobs.append({
            'id': i,
            'source': src_lines[i],
            'translations': [tgt_lines[j][i] for j in range(len(tgt_files))],
            'model_names': model_names
        })
    
    # Process with threading
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(score_translations, job) for job in jobs]
        for future in tqdm(concurrent.futures.as_completed(futures), 
                          total=len(futures), 
                          desc="Scoring translations"):
            results.append(future.result())
    
    # Sort and save results
    results.sort(key=lambda x: x['id'])
    
    output_dir = "tbt-cn-200/T-LLaMA_mev_scores"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    with open(f"{output_dir}/detailed_results.jsonl", 'w') as fout:
        for res in results:
            fout.write(json.dumps(res, ensure_ascii=False) + '\n')
    
    # Save individual score files
    for i, name in enumerate(model_names):
        with open(f"{output_dir}/scores_{name}.txt", 'w') as fout:
            for res in results:
                fout.write(f"{res['scores'][i]}\n")
    
    print(f"\nResults saved to {output_dir}/")
    print(f"Average scores:")
    for i, name in enumerate(model_names):
        avg = sum(r['scores'][i] for r in results) / num_lines
        print(f"{name}: {avg:.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python score_comparative.py <source_file> <translation1> [translation2...]")
        sys.exit(1)
    
    main(*sys.argv[1:])