import json
import os
import glob
from openai import OpenAI


def load_api_config():
    """Load OpenAI API configuration from openai_key.txt file."""
    try:
        with open('openai_key.txt', 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
            base_url = lines[0].strip() if len(lines) > 0 else None
            api_key = lines[1].strip() if len(lines) > 1 else None
        return base_url, api_key
    except FileNotFoundError:
        print("Error: openai_key.txt file not found!")
        return None, None


def evaluate_response(client, question, baseline_answer, model_answer):
    """
    Use GPT to evaluate a model's answer against a baseline.
    
    Args:
        client: OpenAI client instance
        question: The original question
        baseline_answer: The baseline/reference answer
        model_answer: The model's generated answer
        
    Returns:
        dict: Contains score (0-10) and explanation
    """
    
    evaluation_prompt = f"""You are an expert evaluator tasked with scoring the quality of model-generated answers.

Given:
- Question: {question}
- Baseline Answer: {baseline_answer}
- Model Answer: {model_answer}

Please evaluate the Model Answer based on the following criteria:
1. **Relevance** (0-3 points): Does the answer address the question appropriately?
2. **Accuracy** (0-3 points): Is the answer factually correct and consistent with the baseline?
3. **Coherence** (0-2 points): Is the answer well-structured and logically organized?
4. **Completeness** (0-2 points): Does the answer provide sufficient detail?

Provide:
1. A total score from 0-10 (sum of all criteria)
2. A brief explanation (2-3 sentences) justifying the score

Format your response as:
Score: [number]
Explanation: [your explanation]
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert text evaluator who provides objective, consistent scores."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse the score and explanation
        lines = result_text.split('\n')
        score_line = [line for line in lines if line.startswith('Score:')]
        explanation_line = [line for line in lines if line.startswith('Explanation:')]
        
        score = None
        explanation = result_text
        
        if score_line:
            try:
                score_str = score_line[0].replace('Score:', '').strip()
                # Extract just the number (handles formats like "8/10" or "8")
                score = float(score_str.split('/')[0])
            except:
                score = None
        
        if explanation_line:
            explanation = explanation_line[0].replace('Explanation:', '').strip()
        
        return {
            'score': score,
            'explanation': explanation,
            'full_response': result_text
        }
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return {
            'score': None,
            'explanation': f"Evaluation failed: {str(e)}",
            'full_response': None
        }


def process_json_file(filepath, client):
    """
    Process a single JSON file and evaluate all entries.
    
    Args:
        filepath: Path to the JSON file
        client: OpenAI client instance
        
    Returns:
        dict: Evaluation results with scores and statistics
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\nProcessing: {filepath}")
        print(f"Total entries: {len(data)}")
        
        evaluations = []
        scores = []
        
        for idx, entry in enumerate(data):
            print(f"  Evaluating entry {idx + 1}/{len(data)}...", end=' ')
            
            # Each entry is a list: [question, baseline, model_answer]
            if isinstance(entry, list) and len(entry) >= 3:
                question = entry[0] if len(entry) > 0 else ''
                baseline = entry[1] if len(entry) > 1 else ''
                model_answer = entry[2] if len(entry) > 2 else ''
            else:
                print("Skipped (invalid format)")
                continue
            
            # Evaluate the response
            eval_result = evaluate_response(client, question, baseline, model_answer)
            
            evaluations.append({
                'entry_index': idx,
                'question': question[:100] + '...' if len(question) > 100 else question,
                'score': eval_result['score'],
                'explanation': eval_result['explanation']
            })
            
            if eval_result['score'] is not None:
                scores.append(eval_result['score'])
                print(f"Score: {eval_result['score']}/10")
            else:
                print("Failed")
        
        # Calculate statistics
        avg_score = sum(scores) / len(scores) if scores else 0
        min_score = min(scores) if scores else 0
        max_score = max(scores) if scores else 0
        
        results = {
            'file': os.path.basename(filepath),
            'total_entries': len(data),
            'evaluated_entries': len(scores),
            'average_score': round(avg_score, 2),
            'min_score': min_score,
            'max_score': max_score,
            'evaluations': evaluations
        }
        
        print(f"\n  Summary:")
        print(f"    Average Score: {results['average_score']}/10")
        print(f"    Score Range: {min_score} - {max_score}")
        
        return results
        
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON in {filepath}: {e}")
        return None
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None


def save_evaluation_results(all_results, output_file='evaluation_results.json'):
    """Save all evaluation results to a JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Evaluation results saved to: {output_file}")
    except Exception as e:
        print(f"\n✗ Error saving results: {e}")


def main():
    """Main function to evaluate all JSON files in the evaluate folder."""
    
    # Load API configuration
    base_url, api_key = load_api_config()
    if not api_key:
        print("Cannot proceed without API key.")
        return
    
    # Initialize OpenAI client with custom base URL
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        print("✓ OpenAI client initialized successfully")
        print(f"  Base URL: {base_url}")
    except Exception as e:
        print(f"✗ Error initializing OpenAI client: {e}")
        return
    
    # Find all JSON files in evaluate folder
    evaluate_folder = 'evaluate'
    if not os.path.exists(evaluate_folder):
        print(f"Error: '{evaluate_folder}' folder not found!")
        return
    
    json_files = glob.glob(os.path.join(evaluate_folder, '*.json'))
    
    if not json_files:
        print(f"No JSON files found in '{evaluate_folder}' folder.")
        return
    
    print(f"\nFound {len(json_files)} JSON file(s) to evaluate.")
    print("=" * 60)
    
    # Process each JSON file
    all_results = []
    for json_file in json_files:
        result = process_json_file(json_file, client)
        if result:
            all_results.append(result)
        print("-" * 60)
    
    # Save results
    if all_results:
        save_evaluation_results(all_results)
        
        # Print overall summary
        print("\n" + "=" * 60)
        print("OVERALL SUMMARY")
        print("=" * 60)
        for result in all_results:
            print(f"\n{result['file']}:")
            print(f"  Entries: {result['evaluated_entries']}/{result['total_entries']}")
            print(f"  Average Score: {result['average_score']}/10")
            print(f"  Range: {result['min_score']} - {result['max_score']}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
