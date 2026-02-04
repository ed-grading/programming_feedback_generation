"""
Judge Evaluation Pipeline
Evaluates all generated feedback using both Standard and J1-thinking judges
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from langchain_core.prompts import ChatPromptTemplate
from grading_criteria import GRADING_CRITERIA, GRADING_GUIDELINES
from single_judge_prompt import standard_judge_prompt
from j1_judge_prompt import j1_thinking_judge_prompt
from unified_prompt import unified_feedback_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda

@dataclass
class EvaluationResult:
    """Structure for storing evaluation results"""
    problem_id: str
    model_name: str
    approach: str  # 'unified' or 'multi_agent'
    judge_type: str  # 'standard' or 'j1_thinking'
    
    # Evaluation scores
    EA: bool
    ES: bool
    EC: bool
    FA: bool
    FS: bool
    FC: bool
    
    # Metadata
    thinking_trace: str = ""
    timestamp: str = ""
    raw_output: str = ""
    
    def to_dict(self):
        return asdict(self)

def strip_deepseek_think_block(text: str) -> str:
    """
    For DeepSeek-style outputs where the file contains a long 'thinking'
    section followed by a closing </think> tag and then the actual feedback.
    """
    marker = "</think>"
    idx = text.find(marker)
    if idx != -1:
        after = text[idx + len(marker):].strip()
        # If there's something after </think>, use it; otherwise fall back.
        return after if after else text.strip()
    return text.strip()
    

class EvaluationParser:
    """Parser for extracting evaluation results from judge outputs"""
    
    @staticmethod
    def parse_criteria(text: str) -> Dict[str, bool]:
        """Extract criteria evaluations from judge output"""
        criteria = {}
        
        # Look for patterns like "EA: yes" or "EA: no"
        for criterion in ['EA', 'ES', 'EC', 'FA', 'FS', 'FC']:
            pattern = rf'{criterion}:\s*(yes|no)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                criteria[criterion] = match.group(1).lower() == 'yes'
            else:
                # Default to False if not found
                criteria[criterion] = False
                # print(f"Warning: Could not find evaluation for {criterion}")
        
        return criteria
    
    @staticmethod
    def extract_thinking(text: str) -> str:
        """Extract thinking trace from J1 output"""
        match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        return match.group(1).strip() if match else ""

def truncate_text(text: str, max_chars: int = 3000) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > int(0.7 * max_chars):
        return truncated[:last_period + 1]
    return truncated

class JudgeEvaluator:
    """Handles evaluation of feedback using both standard and J1-thinking judges"""
    
    def __init__(self, llm, output_dir: str = "judge_evaluations_qwen"):
        self.llm = llm
        self.output_dir = Path(output_dir)
        self.parser = EvaluationParser()
        self.setup_chains()
        
    def setup_chains(self):
        """Setup evaluation chains for both approaches"""
        
        # Standard judge chain
        self.feedback_generation_chain = (
            unified_feedback_prompt   # or whatever prompt you use for judge-generated feedback
            | self.llm.bind(max_tokens=700) 
            | StrOutputParser()
        )
    
        # 2) SAG-style standard judge chain
        self.standard_chain = (
            RunnableMap(
                problem_description=RunnablePassthrough(),
                test_cases=RunnablePassthrough(),
                student_code=RunnablePassthrough(),
                ta_feedback=RunnablePassthrough(),
                grading_criteria=RunnablePassthrough(),
                grading_guidelines=RunnablePassthrough(),
                # Step 1: judge generates its own bugs+fixes
                judge_generated_bugs_and_fixes=self.feedback_generation_chain
            )
            # Step 2: plug everything into the standard judge prompt
            | standard_judge_prompt
            | self.llm
            | StrOutputParser()
        )
        # J1-thinking judge chain
        self.j1_chain = (
            j1_thinking_judge_prompt 
            | self.llm 
            | StrOutputParser()
        )
    
    def evaluate_feedback(
        self,
        problem_data: Dict,
        feedback: str,
        judge_type: str = "standard"
    ) -> Dict[str, Any]:
        """
        Evaluate a single feedback instance
        
        Args:
            problem_data: Dict with problem_description, test_cases, student_code
            feedback: The generated feedback to evaluate
            judge_type: "standard" or "j1_thinking"
        """

        # Debug show token-ish sizes (character counts)
        print("\n--- INPUT SIZE DEBUG ---")
        print("TA feedback chars:", len(feedback))
        print("Problem description chars:", len(problem_data['problem_description']))
        print("Student code chars:", len(problem_data['student_code']))
        print("Test cases chars:", len(problem_data['test_cases']))
        
        # For SAG judge: judge-generated baseline might be huge
        if judge_type == "standard":
            # We can't compute judge_generated_bugs_and_fixes yet (it gets computed inside the chain)
            # So just warn if TA feedback alone is large
            if len(feedback) > 3000:
                print("⚠️ WARNING: TA feedback is unusually large")
        
        print("------------------------\n")
        
        # Prepare inputs
        inputs = {
            "problem_description": problem_data['problem_description'],
            "test_cases": problem_data['test_cases'],
            "student_code": problem_data['student_code'],
            "ta_feedback": feedback,
            "grading_criteria": GRADING_CRITERIA,
            "grading_guidelines": GRADING_GUIDELINES
        }
        
        # Run appropriate chain
        if judge_type == "standard":
            output = self.standard_chain.invoke(inputs)
            thinking_trace = ""
        else:  # j1_thinking
            output = self.j1_chain.invoke(inputs)
            thinking_trace = self.parser.extract_thinking(output)
        
        # Parse criteria evaluations
        criteria = self.parser.parse_criteria(output)
        
        # Add metadata
        criteria['thinking_trace'] = thinking_trace
        criteria['raw_output'] = output
        criteria['judge_type'] = judge_type
        
        return criteria

class EvaluationPipeline:
    """Main pipeline for evaluating all generated feedback"""
    
    def __init__(self, llm, dataset_dir: str = "Dataset",
                 feedback_dir: str = "generated_feedback",
                 output_dir: str = "judge_evaluations_qwen"):
        self.evaluator = JudgeEvaluator(llm, output_dir)
        self.dataset_dir = Path(dataset_dir)
        self.feedback_dir = Path(feedback_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_problem_data(self, problem_id: str) -> Dict:
        """Load problem data from dataset"""
        problem_file = self.dataset_dir / f"{problem_id}.txt"
        
        with open(problem_file, 'r') as f:
            content = f.read()
        
        # Parse problem components (adjust based on your format)
        problem_match = re.search(r'<problem>(.*?)</problem>', content, re.DOTALL)
        bug_code_match = re.search(r'<bug_code>(.*?)</bug_code>', content, re.DOTALL)
        tests_match = re.search(r'<unit_tests>(.*?)</unit_tests>', content, re.DOTALL)
        
        return {
            'problem_description': problem_match.group(1).strip() if problem_match else "",
            'student_code': bug_code_match.group(1).strip() if bug_code_match else "",
            'test_cases': tests_match.group(1).strip() if tests_match else ""
        }
    
    def load_feedback(self, model_name: str, approach: str, problem_id: str) -> str:
        """Load generated feedback"""
        feedback_file = self.feedback_dir / model_name / approach / f"problem_{problem_id}_feedback.txt"
        
        if not feedback_file.exists():
            print(f"Warning: Feedback file not found: {feedback_file} for model: {model_name}")
            return None
            
        with open(feedback_file, 'r', encoding="utf-8") as f:
            text = f.read()
    
        # If this is the DeepSeek model, strip the thinking part
        if "deepseek" in model_name.lower():
            text = strip_deepseek_think_block(text)
    
        return text
    
    def evaluate_all_feedback(self, models: List[str] = None, max_files: int = None):
        """
        Evaluate all generated feedback with both judge types
        
        Args:
            models: List of model names to evaluate (None for all)
            max_files: Maximum number of files to evaluate per model/approach (None for all)
        """
        
        # Get all model directories if not specified
        if models is None:
            models = [d.name for d in self.feedback_dir.iterdir() if d.is_dir()]
        
        all_results = []
        
        for model_name in models:
            print(f"\n{'='*60}")
            print(f"Evaluating feedback from: {model_name}")
            if max_files:
                print(f"Limiting to {max_files} files per approach")
            print(f"{'='*60}")
            
            for approach in ['unified', 'multi_agent']:
                approach_dir = self.feedback_dir / model_name / approach
                
                if not approach_dir.exists():
                    print(f"Skipping {approach} - directory not found", approach_dir)
                    continue
                
                # Get all feedback files
                feedback_files = sorted(approach_dir.glob("problem_*_feedback.txt"))
                
                # Limit number of files if specified
                if max_files is not None:
                    feedback_files = feedback_files[:max_files]
                    print(f"  Processing {len(feedback_files)} files for {approach}")
                
                for feedback_file in feedback_files:
                    # Extract problem ID from filename
                    problem_id = feedback_file.stem.replace('problem_', '').replace('_feedback', '')
                    
                    print(f"  Evaluating problem {problem_id} ({approach})...")
                    
                    # Load problem data and feedback
                    problem_data = self.load_problem_data(problem_id)
                    feedback = self.load_feedback(model_name, approach, problem_id)
                    
                    if feedback is None:
                        continue
                    
                    # Evaluate with both judges
                    for judge_type in ['standard', 'j1_thinking']:
                        judge_save_dir = (
                            self.output_dir
                            / model_name
                            / approach
                            / judge_type
                        )
                        judge_save_dir.mkdir(parents=True, exist_ok=True)
                        judge_save_path = judge_save_dir / f"problem_{problem_id}.txt"
                    
                        if judge_save_path.exists():
                            print(f"    Skipping {judge_type} for problem {problem_id} - evaluation file already exists")
                            continue
                        try:
                            result = self.evaluator.evaluate_feedback(
                                problem_data, feedback, judge_type
                            )
                            
                            # Create evaluation result
                            eval_result = EvaluationResult(
                                problem_id=problem_id,
                                model_name=model_name,
                                approach=approach,
                                judge_type=judge_type,
                                EA=result.get('EA', False),
                                ES=result.get('ES', False),
                                EC=result.get('EC', False),
                                FA=result.get('FA', False),
                                FS=result.get('FS', False),
                                FC=result.get('FC', False),
                                thinking_trace=result.get('thinking_trace', ''),
                                timestamp=datetime.now().isoformat(),
                                raw_output=result.get('raw_output', '')
                            )
                            
                            all_results.append(eval_result)
                            
                            # Save individual result
                            self.save_individual_result(eval_result)
                            
                        except Exception as e:
                            print(f"    Error evaluating with {judge_type}: {e}")
                            self.log_error(model_name, approach, problem_id, judge_type, e)
            
            # Save progress after each model
            self.save_batch_results(all_results, model_name)
        
        # Save final aggregated results
        self.save_final_results(all_results)
        
        return all_results
    
    def save_individual_result(self, result: EvaluationResult):
        """Save individual evaluation result"""
        # Create directory structure
        save_dir = (
            self.output_dir
            / result.model_name
            / result.approach
            / result.judge_type
        )
        save_dir.mkdir(parents=True, exist_ok=True)
    
        # Save the judge's feedback as plain text
        save_path = save_dir / f"problem_{result.problem_id}.txt"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(result.raw_output)
    
    def save_batch_results(self, results: List[EvaluationResult], model_name: str):
        """Save batch results for a model"""
        save_path = self.output_dir / f"{model_name}_evaluations.json"
        
        with open(save_path, 'w') as f:
            json.dump([r.to_dict() for r in results if r.model_name == model_name], f, indent=2)
    
    def save_final_results(self, results: List[EvaluationResult]):
        """Save all results in various formats"""
        
        # JSON format
        json_path = self.output_dir / "all_evaluations.json"
        with open(json_path, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        
        # CSV format for easy analysis
        import csv
        csv_path = self.output_dir / "all_evaluations.csv"
        
        if results:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].to_dict().keys())
                writer.writeheader()
                for result in results:
                    writer.writerow(result.to_dict())
        
        print(f"\nResults saved to {self.output_dir}")

    def log_error(self, model_name: str, approach: str, problem_id: str, judge_type: str, error: Exception):
        """Append error info to a log file for later inspection."""
        log_path = self.output_dir / "errors.log"
        msg = (
            f"[{datetime.now().isoformat()}] "
            f"Model={model_name} | Approach={approach} | Problem={problem_id} | Judge={judge_type}\n"
            f"Error: {error}\n"
            f"{'-'*80}\n"
        )
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg)


def run_evaluation(llm, models: List[str] = None, max_files: int = None):
    """
    Main function to run the complete evaluation pipeline
    
    Args:
        llm: Language model instance
        models: List of model names to evaluate
        max_files: Maximum number of files to evaluate per model/approach (None for all)
    """
    
    # Create pipeline
    pipeline = EvaluationPipeline(llm)
    
    # Run evaluations
    print("Starting judge evaluations...")
    if max_files:
        print(f"Limiting to {max_files} files per model/approach")
    results = pipeline.evaluate_all_feedback(models, max_files=max_files)
    
    
    return results

def main():
    
    # Example usage (you'll need to replace with your actual LLM setup)
    
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
    model="Qwen/Qwen2.5-Coder-32B-Instruct",   # must match the vLLM model name
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",            
    temperature=0.0
    )

    models_to_evaluate = [
        "Llama-33_70B_awq",
        "gpt-oss-20b",
        # "qwen25_32B",
        "deepseek_r1_qwen_32B",
         "qwen25_14b",
        # "gpt_oss_20b",
        # "deepseek_r1_distill_32b"
    ]
    
    print(f"Models to evaluate: {models_to_evaluate}")
    print("\nStarting evaluation pipeline...")
    print("This will evaluate each feedback with both standard and J1-thinking judges")
    
    # Run the evaluation
    results = run_evaluation(
        llm=llm,
        models=models_to_evaluate,
        # max_files=3,
    )
    
    print(f"\nCompleted {len(results)} evaluations")
    


if __name__ == "__main__":
    main()
    print("Judge Evaluation Pipeline")
    print("="*60)
    print("\nUsage:")
    print("1. Initialize your LLM")
    print("2. Call: results = run_evaluation(llm)")
    print("3. Results saved to 'judge_evaluations/' directory")
    
    




