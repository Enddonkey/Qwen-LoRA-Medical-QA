import json
import os
import glob
from openai import OpenAI
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


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
    Use GPT to evaluate a model's answer against a baseline using comparison approach.
    
    Args:
        client: OpenAI client instance
        question: The original question
        baseline_answer: The baseline/reference answer (Assistant 1)
        model_answer: The model's generated answer (Assistant 2)
        
    Returns:
        dict: Contains comparison result and explanation
    """
    
    evaluation_prompt = f"""We would like to request your feedback on the two AI assistants in response to the user question displayed above.

Please evaluate the helpfulness, relevance, accuracy, level of details of their responses. You should tell me whether Assistant 1 is 'better than', 'worse than', or 'equal to' Assistant 2.

Please first compare their responses and analyze which one is more in line with the given requirements.

In the last line, please output a single line containing only a single label selecting from 'Assistant 1 is better than Assistant 2', 'Assistant 1 is worse than Assistant 2', and 'Assistant 1 is equal to Assistant 2', avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

Question: {question}

Assistant 1: {baseline_answer}

Assistant 2: {model_answer}
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert evaluator who provides objective, unbiased comparisons between AI assistant responses."},
                {"role": "user", "content": evaluation_prompt}
            ],
            temperature=0.3,
            max_tokens=3000
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Parse the comparison result from the last line
        lines = result_text.split('\n')
        last_line = lines[-1].strip()
        
        # Determine the comparison result
        comparison = None
        if 'Assistant 1 is better than Assistant 2' in last_line:
            comparison = 'baseline_better'
        elif 'Assistant 1 is worse than Assistant 2' in last_line:
            comparison = 'model_better'
        elif 'Assistant 1 is equal to Assistant 2' in last_line:
            comparison = 'equal'
        
        return {
            'comparison': comparison,
            'explanation': result_text,
            'raw_judgment': last_line
        }
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return {
            'comparison': None,
            'explanation': f"Evaluation failed: {str(e)}",
            'raw_judgment': None
        }


def process_json_file(filepath, client):
    """
    Process a single JSON file and evaluate all entries.
    
    Args:
        filepath: Path to the JSON file
        client: OpenAI client instance
        
    Returns:
        dict: Evaluation results with comparison statistics
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\nProcessing: {filepath}")
        print(f"Total entries: {len(data)}")
        
        evaluations = []
        baseline_better_count = 0
        model_better_count = 0
        equal_count = 0
        failed_count = 0
        
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
                'comparison': eval_result['comparison'],
                'explanation': eval_result['explanation'],
                'raw_judgment': eval_result['raw_judgment']
            })
            
            # Count comparisons
            if eval_result['comparison'] == 'baseline_better':
                baseline_better_count += 1
                print("Baseline Better")
            elif eval_result['comparison'] == 'model_better':
                model_better_count += 1
                print("Model Better")
            elif eval_result['comparison'] == 'equal':
                equal_count += 1
                print("Equal")
            else:
                failed_count += 1
                print("Failed")
        
        results = {
            'file': os.path.basename(filepath),
            'total_entries': len(data),
            'evaluated_entries': len(data) - failed_count,
            'baseline_better': baseline_better_count,
            'model_better': model_better_count,
            'equal': equal_count,
            'failed': failed_count,
            'evaluations': evaluations
        }
        
        print(f"\n  Summary:")
        print(f"    Baseline Better: {baseline_better_count}")
        print(f"    Model Better: {model_better_count}")
        print(f"    Equal: {equal_count}")
        print(f"    Failed: {failed_count}")
        
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


def visualize_results(all_results, output_dir='./evaluate/evaluation_plots'):
    """
    Create visualizations for evaluation results.
    
    Args:
        all_results: List of evaluation results for each file
        output_dir: Directory to save plots
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Create comparison bar chart for each file
        for result in all_results:
            filename = result['file'].replace('.json', '')
            
            categories = ['Baseline Better', 'Model Better', 'Equal', 'Failed']
            counts = [
                result['baseline_better'],
                result['model_better'],
                result['equal'],
                result['failed']
            ]
            
            plt.figure(figsize=(10, 6))
            colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#CCCCCC']
            bars = plt.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.2)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            plt.xlabel('Comparison Result', fontsize=12, fontweight='bold')
            plt.ylabel('Count', fontsize=12, fontweight='bold')
            plt.title(f'Evaluation Results: {filename}', fontsize=14, fontweight='bold')
            plt.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            
            plot_file = os.path.join(output_dir, f'{filename}_comparison.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved plot: {plot_file}")
        
        # 2. Create overall comparison across all files
        if len(all_results) > 1:
            plt.figure(figsize=(14, 8))
            
            filenames = [r['file'].replace('.json', '') for r in all_results]
            baseline_better = [r['baseline_better'] for r in all_results]
            model_better = [r['model_better'] for r in all_results]
            equal = [r['equal'] for r in all_results]
            
            x = range(len(filenames))
            width = 0.25
            
            plt.bar([i - width for i in x], baseline_better, width, label='Baseline Better', color='#FF6B6B', edgecolor='black')
            plt.bar(x, model_better, width, label='Model Better', color='#4ECDC4', edgecolor='black')
            plt.bar([i + width for i in x], equal, width, label='Equal', color='#95E1D3', edgecolor='black')
            
            plt.xlabel('Model Files', fontsize=12, fontweight='bold')
            plt.ylabel('Count', fontsize=12, fontweight='bold')
            plt.title('Overall Comparison Across All Models', fontsize=14, fontweight='bold')
            plt.xticks(x, filenames, rotation=45, ha='right')
            plt.legend(fontsize=10)
            plt.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            
            overall_plot = os.path.join(output_dir, 'overall_comparison.png')
            plt.savefig(overall_plot, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved plot: {overall_plot}")
        
        # 3. Create pie chart for each file
        for result in all_results:
            filename = result['file'].replace('.json', '')
            
            # Only include non-zero categories
            categories = []
            counts = []
            colors = []
            color_map = {
                'Baseline Better': '#FF6B6B',
                'Model Better': '#4ECDC4',
                'Equal': '#95E1D3'
            }
            
            if result['baseline_better'] > 0:
                categories.append('Baseline Better')
                counts.append(result['baseline_better'])
                colors.append(color_map['Baseline Better'])
            if result['model_better'] > 0:
                categories.append('Model Better')
                counts.append(result['model_better'])
                colors.append(color_map['Model Better'])
            if result['equal'] > 0:
                categories.append('Equal')
                counts.append(result['equal'])
                colors.append(color_map['Equal'])
            
            if counts:
                plt.figure(figsize=(10, 8))
                wedges, texts, autotexts = plt.pie(counts, labels=categories, colors=colors,
                                                    autopct='%1.1f%%', startangle=90,
                                                    textprops={'fontsize': 12, 'fontweight': 'bold'})
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                
                plt.title(f'Distribution: {filename}', fontsize=14, fontweight='bold')
                plt.axis('equal')
                plt.tight_layout()
                
                pie_file = os.path.join(output_dir, f'{filename}_pie.png')
                plt.savefig(pie_file, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  Saved plot: {pie_file}")
        
        print(f"\n✓ All plots saved to: {output_dir}/")
        
    except Exception as e:
        print(f"\n✗ Error creating visualizations: {e}")


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
            print(f"  Total Entries: {result['total_entries']}")
            print(f"  Baseline Better: {result['baseline_better']}")
            print(f"  Model Better: {result['model_better']}")
            print(f"  Equal: {result['equal']}")
            print(f"  Failed: {result['failed']}")
            
            # Calculate percentages
            total_valid = result['baseline_better'] + result['model_better'] + result['equal']
            if total_valid > 0:
                model_win_rate = (result['model_better'] / total_valid) * 100
                print(f"  Model Win Rate: {model_win_rate:.1f}%")
        
        # Create visualizations
        print("\n" + "=" * 60)
        print("CREATING VISUALIZATIONS")
        print("=" * 60)
        visualize_results(all_results)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
