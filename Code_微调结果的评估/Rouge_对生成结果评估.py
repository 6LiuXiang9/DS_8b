from unsloth import FastLanguageModel
import torch
import os
import json
from rouge import Rouge
import pandas as pd
from tqdm import tqdm
import numpy as np

# Print current working directory to help locate model files
print(f"Current working directory: {os.getcwd()}")

# Model configuration
max_seq_length = 2048
load_in_4bit = True

# Path to your saved fine-tuned model and test data
model_path = "E:/DS_8b/DeepSeek-R1-Medical-COT_710"
test_data_path = "E:/Deepseek/data_sorce/HZ/Test_medical_data_with_cot_reasoner_100.json"

print(f"Loading model from: {model_path}")
print(f"Loading test data from: {test_data_path}")

# Load the fine-tuned model and tokenizer
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )
    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Set the model to inference mode
FastLanguageModel.for_inference(model)
print("Model prepared for inference")

# Setup prompt template similar to what was used during training
test_prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
{}

### Response:
"""


# ROUGE score calculation
def calculate_rouge(reference, candidate):
    """
    Calculate ROUGE-L F1 score between reference and candidate texts
    """
    try:
        rouge = Rouge()
        # Clean texts - remove extra whitespace and ensure non-empty
        reference_clean = ' '.join(reference.strip().split())
        candidate_clean = ' '.join(candidate.strip().split())

        if not reference_clean or not candidate_clean:
            return 0.0

        scores = rouge.get_scores(candidate_clean, reference_clean)
        return scores[0]['rouge-l']['f']
    except Exception as e:
        print(f"ROUGE calculation error: {e}")
        return 0.0


def extract_response_and_cot(full_response):
    """
    Extract both CoT and Response content from the full generated text
    Returns: (cot_content, response_content)
    """
    try:
        # Split by "### Response:" and take the part after it
        if "### Response:" in full_response:
            response_part = full_response.split("### Response:")[1].strip()
        else:
            response_part = full_response.strip()

        cot_content = ""
        response_content = ""

        # Check if thinking tags are present
        if "<think>" in response_part and "</think>" in response_part:
            # Extract CoT content between <think> and </think>
            think_start = response_part.find("<think>")
            think_end = response_part.find("</think>")

            if think_start != -1 and think_end != -1:
                cot_content = response_part[think_start + 7:think_end].strip()
                # Extract response content after </think>
                response_content = response_part[think_end + 8:].strip()
            else:
                # No proper thinking tags, treat all as response
                response_content = response_part
        else:
            # No thinking tags, treat all as response
            response_content = response_part

        return cot_content, response_content
    except:
        return "", full_response.strip()


# Load test data (separate from training data)
try:
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples from independent test dataset")
except Exception as e:
    print(f"Error loading test data: {e}")
    exit(1)

# Setup device
if torch.cuda.is_available():
    device = "cuda"
    model = model.to(device)
    print("Using CUDA for inference")
else:
    device = "cpu"
    print("Using CPU for inference (this might be slow)")

# Evaluation parameters - Use ALL test data
num_samples = len(test_data)  # Use all available test samples
sampled_data = test_data  # Use complete dataset

print(f"Using complete test dataset: {num_samples} samples")
print("Evaluating on ALL samples for comprehensive assessment")

# Store results
results = []
response_rouge_scores = []
cot_rouge_scores = []
combined_rouge_scores = []

print(f"\nStarting comprehensive evaluation on complete test dataset...")
print(f"Evaluating ALL {num_samples} samples for maximum statistical reliability")
print("Evaluating Response, CoT, and Combined ROUGE-L F1 scores...")
print("This avoids data leakage since test data is separate from training data.")
print("-" * 70)

for i, sample in enumerate(tqdm(sampled_data, desc="Evaluating")):
    try:
        # Get question and reference texts
        question = sample.get('Question', '')
        reference_response = sample.get('Response', '')
        reference_cot = sample.get('Complex_CoT', '')

        if not question or not reference_response:
            print(f"Skipping sample {i}: missing question or response")
            continue

        # Format the question
        formatted_input = test_prompt_template.format(question)

        # Tokenize input
        inputs = tokenizer([formatted_input], return_tensors="pt")
        if device == "cuda":
            inputs = inputs.to(device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=1000,  # Increased for CoT + Response
                temperature=0.7,
                top_p=0.9,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode the response
        full_response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        generated_cot, generated_response = extract_response_and_cot(full_response)

        # Calculate ROUGE scores for each component

        # 1. Response ROUGE-L F1
        response_rouge = calculate_rouge(reference_response, generated_response)
        response_rouge_scores.append(response_rouge)

        # 2. CoT ROUGE-L F1 (only if both reference and generated CoT exist)
        cot_rouge = 0.0
        if reference_cot and generated_cot:
            cot_rouge = calculate_rouge(reference_cot, generated_cot)
        cot_rouge_scores.append(cot_rouge)

        # 3. Combined ROUGE-L F1 (CoT + Response)
        reference_combined = f"{reference_cot} {reference_response}".strip()
        generated_combined = f"{generated_cot} {generated_response}".strip()
        combined_rouge = calculate_rouge(reference_combined, generated_combined)
        combined_rouge_scores.append(combined_rouge)

        # Store detailed result
        result = {
            'sample_id': i,
            'question': question,
            'reference_response': reference_response,
            'reference_cot': reference_cot,
            'generated_response': generated_response,
            'generated_cot': generated_cot,
            'response_rouge_l_f1': response_rouge,
            'cot_rouge_l_f1': cot_rouge,
            'combined_rouge_l_f1': combined_rouge
        }
        results.append(result)

        # Print sample results every 20 samples for complete dataset
        if (i + 1) % 20 == 0:
            avg_response_rouge = np.mean(response_rouge_scores)
            avg_cot_rouge = np.mean(cot_rouge_scores)
            avg_combined_rouge = np.mean(combined_rouge_scores)

            print(f"\nProgress: {i + 1}/{num_samples} samples completed ({((i + 1) / num_samples) * 100:.1f}%)")
            print(f"Current - Response: {response_rouge:.4f}, CoT: {cot_rouge:.4f}, Combined: {combined_rouge:.4f}")
            print(
                f"Running Average - Response: {avg_response_rouge:.4f}, CoT: {avg_cot_rouge:.4f}, Combined: {avg_combined_rouge:.4f}")
            print("-" * 70)

    except Exception as e:
        print(f"Error processing sample {i}: {e}")
        continue

# Calculate final statistics for all three metrics
if response_rouge_scores and cot_rouge_scores and combined_rouge_scores:
    final_stats = {
        # Response statistics
        'response_mean_rouge_l_f1': np.mean(response_rouge_scores),
        'response_std_rouge_l_f1': np.std(response_rouge_scores),
        'response_min_rouge_l_f1': np.min(response_rouge_scores),
        'response_max_rouge_l_f1': np.max(response_rouge_scores),
        'response_median_rouge_l_f1': np.median(response_rouge_scores),

        # CoT statistics
        'cot_mean_rouge_l_f1': np.mean(cot_rouge_scores),
        'cot_std_rouge_l_f1': np.std(cot_rouge_scores),
        'cot_min_rouge_l_f1': np.min(cot_rouge_scores),
        'cot_max_rouge_l_f1': np.max(cot_rouge_scores),
        'cot_median_rouge_l_f1': np.median(cot_rouge_scores),

        # Combined statistics
        'combined_mean_rouge_l_f1': np.mean(combined_rouge_scores),
        'combined_std_rouge_l_f1': np.std(combined_rouge_scores),
        'combined_min_rouge_l_f1': np.min(combined_rouge_scores),
        'combined_max_rouge_l_f1': np.max(combined_rouge_scores),
        'combined_median_rouge_l_f1': np.median(combined_rouge_scores),

        # General info
        'total_samples': len(response_rouge_scores),
        'test_dataset': 'Test_medical_data_with_cot_reasoner_100.json',
        'model_name': 'DeepSeek-R1-Medical-COT_710'
    }

    print("\n" + "=" * 80)
    print("FINAL COMPREHENSIVE EVALUATION RESULTS - COMPLETE DATASET ANALYSIS")
    print("=" * 80)
    print(f"Model: {final_stats['model_name']}")
    print(f"Test Dataset: {final_stats['test_dataset']}")
    print(f"Total samples evaluated: {final_stats['total_samples']} (COMPLETE DATASET)")
    print("=" * 80)

    # Display results in a formatted table
    print(f"{'Metric':<15} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Median':<8}")
    print("-" * 65)
    print(
        f"{'Response':<15} {final_stats['response_mean_rouge_l_f1']:<8.4f} {final_stats['response_std_rouge_l_f1']:<8.4f} {final_stats['response_min_rouge_l_f1']:<8.4f} {final_stats['response_max_rouge_l_f1']:<8.4f} {final_stats['response_median_rouge_l_f1']:<8.4f}")
    print(
        f"{'CoT':<15} {final_stats['cot_mean_rouge_l_f1']:<8.4f} {final_stats['cot_std_rouge_l_f1']:<8.4f} {final_stats['cot_min_rouge_l_f1']:<8.4f} {final_stats['cot_max_rouge_l_f1']:<8.4f} {final_stats['cot_median_rouge_l_f1']:<8.4f}")
    print(
        f"{'Combined':<15} {final_stats['combined_mean_rouge_l_f1']:<8.4f} {final_stats['combined_std_rouge_l_f1']:<8.4f} {final_stats['combined_min_rouge_l_f1']:<8.4f} {final_stats['combined_max_rouge_l_f1']:<8.4f} {final_stats['combined_median_rouge_l_f1']:<8.4f}")
    print("=" * 80)


    # Calculate confidence intervals for more robust statistics
    def calculate_confidence_interval(scores, confidence=0.95):
        """Calculate confidence interval for the mean"""
        try:
            from scipy import stats
            n = len(scores)
            if n < 2:
                return np.nan, np.nan

            mean = np.mean(scores)
            sem = stats.sem(scores)  # Standard error of the mean
            h = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
            return mean - h, mean + h
        except ImportError:
            return np.nan, np.nan
        except Exception:
            return np.nan, np.nan


    # Calculate 95% confidence intervals
    response_ci = calculate_confidence_interval(response_rouge_scores)
    cot_ci = calculate_confidence_interval(cot_rouge_scores)
    combined_ci = calculate_confidence_interval(combined_rouge_scores)

    if not np.isnan(response_ci[0]):
        print(f"\n📊 95% Confidence Intervals:")
        print(f"Response:  [{response_ci[0]:.4f}, {response_ci[1]:.4f}]")
        print(f"CoT:       [{cot_ci[0]:.4f}, {cot_ci[1]:.4f}]")
        print(f"Combined:  [{combined_ci[0]:.4f}, {combined_ci[1]:.4f}]")

        # Add to final stats
        final_stats.update({
            'response_ci_lower': response_ci[0],
            'response_ci_upper': response_ci[1],
            'cot_ci_lower': cot_ci[0],
            'cot_ci_upper': cot_ci[1],
            'combined_ci_lower': combined_ci[0],
            'combined_ci_upper': combined_ci[1]
        })
    else:
        print(f"\n📝 Note: Install scipy for confidence interval calculations")


    # Performance interpretation for each metric
    def interpret_performance(score, metric_name):
        if score >= 0.5:
            return f"🟢 {metric_name}: Excellent performance"
        elif score >= 0.3:
            return f"🟡 {metric_name}: Good performance"
        elif score >= 0.2:
            return f"🟠 {metric_name}: Fair performance"
        else:
            return f"🔴 {metric_name}: Needs improvement"


    print(f"\nPerformance Interpretation:")
    print(interpret_performance(final_stats['response_mean_rouge_l_f1'], "Response"))
    print(interpret_performance(final_stats['cot_mean_rouge_l_f1'], "CoT"))
    print(interpret_performance(final_stats['combined_mean_rouge_l_f1'], "Combined"))

    # Calculate overall average across all three metrics
    overall_average = (final_stats['response_mean_rouge_l_f1'] +
                       final_stats['cot_mean_rouge_l_f1'] +
                       final_stats['combined_mean_rouge_l_f1']) / 3

    print(f"\n🎯 Overall Average ROUGE-L F1: {overall_average:.4f}")
    print(interpret_performance(overall_average, "Overall Model"))

    # Additional comprehensive analysis
    print(f"\n📈 Detailed Analysis:")
    print(f"• Best performing aspect: Response" if final_stats['response_mean_rouge_l_f1'] == max(
        final_stats['response_mean_rouge_l_f1'], final_stats['cot_mean_rouge_l_f1'],
        final_stats['combined_mean_rouge_l_f1']) else
          f"• Best performing aspect: CoT" if final_stats['cot_mean_rouge_l_f1'] == max(
              final_stats['response_mean_rouge_l_f1'], final_stats['cot_mean_rouge_l_f1'],
              final_stats['combined_mean_rouge_l_f1']) else
          f"• Best performing aspect: Combined")

    print(f"• Most consistent performance: Response" if final_stats['response_std_rouge_l_f1'] == min(
        final_stats['response_std_rouge_l_f1'], final_stats['cot_std_rouge_l_f1'],
        final_stats['combined_std_rouge_l_f1']) else
          f"• Most consistent performance: CoT" if final_stats['cot_std_rouge_l_f1'] == min(
              final_stats['response_std_rouge_l_f1'], final_stats['cot_std_rouge_l_f1'],
              final_stats['combined_std_rouge_l_f1']) else
          f"• Most consistent performance: Combined")

    response_var_coeff = final_stats['response_std_rouge_l_f1'] / final_stats['response_mean_rouge_l_f1']
    cot_var_coeff = final_stats['cot_std_rouge_l_f1'] / final_stats['cot_mean_rouge_l_f1'] if final_stats[
                                                                                                  'cot_mean_rouge_l_f1'] > 0 else float(
        'inf')
    combined_var_coeff = final_stats['combined_std_rouge_l_f1'] / final_stats['combined_mean_rouge_l_f1']

    print(
        f"• Coefficient of Variation - Response: {response_var_coeff:.3f}, CoT: {cot_var_coeff:.3f}, Combined: {combined_var_coeff:.3f}")

    # Save results to files
    try:
        # Save detailed results
        output_file = f'comprehensive_evaluation_{final_stats["model_name"]}_complete_dataset.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'evaluation_info': {
                    'model': final_stats['model_name'],
                    'test_dataset': final_stats['test_dataset'],
                    'evaluation_date': str(pd.Timestamp.now()),
                    'evaluation_type': 'Comprehensive ROUGE-L F1 (Response + CoT + Combined)',
                    'dataset_size': final_stats['total_samples'],
                    'note': 'Evaluated on complete independent test dataset to avoid data leakage'
                },
                'statistics': final_stats,
                'overall_average': overall_average,
                'detailed_results': results
            }, f, ensure_ascii=False, indent=2)

        # Save summary CSV
        df = pd.DataFrame(results)
        csv_file = f'comprehensive_rouge_evaluation_{final_stats["model_name"]}_complete_dataset.csv'
        df.to_csv(csv_file, index=False, encoding='utf-8')

        # Save statistics summary
        stats_df = pd.DataFrame([{
            'Metric': 'Response',
            'Mean': final_stats['response_mean_rouge_l_f1'],
            'Std': final_stats['response_std_rouge_l_f1'],
            'Min': final_stats['response_min_rouge_l_f1'],
            'Max': final_stats['response_max_rouge_l_f1'],
            'Median': final_stats['response_median_rouge_l_f1'],
            'CI_Lower': final_stats.get('response_ci_lower', np.nan),
            'CI_Upper': final_stats.get('response_ci_upper', np.nan)
        }, {
            'Metric': 'CoT',
            'Mean': final_stats['cot_mean_rouge_l_f1'],
            'Std': final_stats['cot_std_rouge_l_f1'],
            'Min': final_stats['cot_min_rouge_l_f1'],
            'Max': final_stats['cot_max_rouge_l_f1'],
            'Median': final_stats['cot_median_rouge_l_f1'],
            'CI_Lower': final_stats.get('cot_ci_lower', np.nan),
            'CI_Upper': final_stats.get('cot_ci_upper', np.nan)
        }, {
            'Metric': 'Combined',
            'Mean': final_stats['combined_mean_rouge_l_f1'],
            'Std': final_stats['combined_std_rouge_l_f1'],
            'Min': final_stats['combined_min_rouge_l_f1'],
            'Max': final_stats['combined_max_rouge_l_f1'],
            'Median': final_stats['combined_median_rouge_l_f1'],
            'CI_Lower': final_stats.get('combined_ci_lower', np.nan),
            'CI_Upper': final_stats.get('combined_ci_upper', np.nan)
        }, {
            'Metric': 'Overall_Average',
            'Mean': overall_average,
            'Std': np.nan,
            'Min': np.nan,
            'Max': np.nan,
            'Median': np.nan,
            'CI_Lower': np.nan,
            'CI_Upper': np.nan
        }])

        stats_csv = f'rouge_statistics_summary_{final_stats["model_name"]}_complete_dataset.csv'
        stats_df.to_csv(stats_csv, index=False)

        print(f"\nResults saved to:")
        print(f"- {output_file} (comprehensive detailed results)")
        print(f"- {csv_file} (detailed evaluation data)")
        print(f"- {stats_csv} (statistics summary)")

    except Exception as e:
        print(f"Error saving results: {e}")

    # Show sample comparisons for each metric
    print("\n" + "=" * 80)
    print("SAMPLE COMPARISONS (Best and Worst Cases for Each Metric)")
    print("=" * 80)

    # Sort results by each metric
    sorted_by_response = sorted(results, key=lambda x: x['response_rouge_l_f1'])
    sorted_by_cot = sorted(results, key=lambda x: x['cot_rouge_l_f1'])
    sorted_by_combined = sorted(results, key=lambda x: x['combined_rouge_l_f1'])


    def show_case(case_list, metric_name, case_type, metric_key):
        if not case_list:
            return

        case = case_list[-1] if case_type == "BEST" else case_list[0]
        icon = "🟢" if case_type == "BEST" else "🔴"

        print(f"\n{icon} {case_type} {metric_name} CASE (ROUGE-L F1: {case[metric_key]:.4f})")
        print(f"Question: {case['question'][:80]}...")

        if metric_name == "Response":
            print(f"Reference: {case['reference_response'][:100]}...")
            print(f"Generated: {case['generated_response'][:100]}...")
        elif metric_name == "CoT":
            print(f"Reference: {case['reference_cot'][:100]}...")
            print(f"Generated: {case['generated_cot'][:100]}...")
        else:  # Combined
            ref_combined = f"{case['reference_cot']} {case['reference_response']}"
            gen_combined = f"{case['generated_cot']} {case['generated_response']}"
            print(f"Reference: {ref_combined[:100]}...")
            print(f"Generated: {gen_combined[:100]}...")
        print("-" * 60)


    # Show best and worst cases for each metric
    show_case(sorted_by_response, "Response", "WORST", "response_rouge_l_f1")
    show_case(sorted_by_response, "Response", "BEST", "response_rouge_l_f1")

    show_case(sorted_by_cot, "CoT", "WORST", "cot_rouge_l_f1")
    show_case(sorted_by_cot, "CoT", "BEST", "cot_rouge_l_f1")

    show_case(sorted_by_combined, "Combined", "WORST", "combined_rouge_l_f1")
    show_case(sorted_by_combined, "Combined", "BEST", "combined_rouge_l_f1")

else:
    print("❌ No valid evaluations completed!")

print(f"\n✅ Comprehensive evaluation completed!")
print("📊 Evaluated Response, CoT, and Combined ROUGE-L F1 scores on complete dataset")
print("🛡️ Used independent test dataset to avoid data leakage")
print("📈 Generated comprehensive statistics, confidence intervals, and sample comparisons")
print("🎯 Analyzed ALL samples for maximum statistical reliability")