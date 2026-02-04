import os
import re
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Tuple
from multi_agents_prompt import feedback_generation_prompt_ag, bug_detection_prompt_ag
from unified_prompt import unified_feedback_prompt
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda


def parse_problem_file(file_path: str) -> Dict[str, str]:

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract each section using regex
    problem_match = re.search(r'<problem>(.*?)</problem>', content, re.DOTALL)
    bug_code_match = re.search(r'<bug_code>(.*?)</bug_code>', content, re.DOTALL)
    bug_desc_match = re.search(r'<bug_desc>(.*?)</bug_desc>', content, re.DOTALL)
    bug_fixes_match = re.search(r'<bug_fixes>(.*?)</bug_fixes>', content, re.DOTALL)
    unit_tests_match = re.search(r'<unit_tests>(.*?)</unit_tests>', content, re.DOTALL)
    correct_code_match = re.search(r'<code>(.*?)</code>', content, re.DOTALL)
    
    return {
        'problem_description': problem_match.group(1).strip() if problem_match else "",
        'student_code': bug_code_match.group(1).strip() if bug_code_match else "",
        'bug_description': bug_desc_match.group(1).strip() if bug_desc_match else "",
        'bug_fixes': bug_fixes_match.group(1).strip() if bug_fixes_match else "",
        'test_cases': unit_tests_match.group(1).strip() if unit_tests_match else "",
        'correct_code': correct_code_match.group(1).strip() if correct_code_match else ""
    }


def strip_deepseek_think(text: str) -> str:
    marker = "</think>"
    idx = text.find(marker)
    if idx != -1:
        print("Think block detected")
        # Take everything AFTER the marker
        after = text[idx + len(marker):]
        cleaned = after.strip()
        return cleaned if cleaned else text.strip()
    # Fallback: no marker, just strip whitespace
    print("Did not detect think block")
    return text.strip()
    
class FeedbackGenerator:
    """
    Handles feedback generation for both unified and multi-agent approaches
    """
    
    def __init__(self, llm, output_dir: str = "generated_feedback", strip_think_block: bool = False):
        """
        Initialize the feedback generator
        
        Args:
            llm: Language model instance
            output_dir: Directory to save generated feedback
            strip_think_block: If True, remove <think>...</think> from bug_analysis
                               (useful for DeepSeek-R1 style models)
        """
        self.llm = llm
        self.output_dir = Path(output_dir)
        self.strip_think_block = strip_think_block
        self.setup_chains()
        
    def setup_chains(self):
        """Setup the chains for both approaches"""
        
        # Unified approach chain (unchanged)
        self.unified_chain = (
            unified_feedback_prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        bug_analysis_chain = (
            bug_detection_prompt_ag
            | self.llm
            | StrOutputParser()
        )
        
        if self.strip_think_block:
            print("Came in If block")
            bug_analysis_chain = bug_analysis_chain | RunnableLambda(strip_deepseek_think)
        
        # Multi-agent approach chain
        self.multi_agent_chain = (
            RunnableMap(
                problem_description=RunnablePassthrough(),
                test_cases=RunnablePassthrough(),
                student_code=RunnablePassthrough(),
                bug_analysis=bug_analysis_chain,
            )
            | feedback_generation_prompt_ag
            | self.llm
            | StrOutputParser()
        )

    
    def generate_feedback_unified(self, problem_data: Dict) -> str:
        """Generate feedback using unified approach"""
        return self.unified_chain.invoke({
            "problem_description": problem_data['problem_description'],
            "test_cases": problem_data['test_cases'],
            "student_code": problem_data['student_code']
        })
    
    def generate_feedback_multi_agent(self, problem_data: Dict) -> str:
        """Generate feedback using multi-agent approach"""
        return self.multi_agent_chain.invoke({
            "problem_description": problem_data['problem_description'],
            "test_cases": problem_data['test_cases'],
            "student_code": problem_data['student_code']
        })
    
    def save_feedback(self, feedback: str, model_name: str, approach: str, 
                     problem_id: str):
        """
        Save feedback to appropriate file
        
        Args:
            feedback: Generated feedback text
            model_name: Name of the model used
            approach: 'unified' or 'multi_agent'
            problem_id: Problem identifier
        """
        # Create directory structure
        save_dir = self.output_dir / model_name / approach
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save feedback
        file_path = save_dir / f"problem_{problem_id}_feedback.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(feedback)
        
        return file_path
    
    def process_single_problem(self, file_path: str, problem_id: str, 
                              model_name: str, skip_existing: bool = True) -> Dict:
        """
        Process a single problem with both approaches
        
        Args:
            file_path: Path to problem file
            problem_id: Problem identifier
            model_name: Name of the model being used
            
        Returns:
            Dictionary with results and metadata
        """

        if skip_existing:
            unified_path = self.output_dir / model_name / 'unified' / f"problem_{problem_id}_feedback.txt"
            multi_agent_path = self.output_dir / model_name / 'multi_agent' / f"problem_{problem_id}_feedback.txt"
            
            if unified_path.exists() or multi_agent_path.exists():
                print(f"  - Skipping (already processed)")
                return {
                    'problem_id': problem_id,
                    'file_path': file_path,
                    'status': 'skipped',
                    'unified': {'status': 'skipped', 'output_path': str(unified_path)},
                    'multi_agent': {'status': 'skipped', 'output_path': str(multi_agent_path)}
                }
        print(f"Processing problem {problem_id}...")
        
        # Parse problem file
        problem_data = parse_problem_file(file_path)
        
        results = {
            'problem_id': problem_id,
            'file_path': file_path,
            'timestamp': datetime.now().isoformat()
        }
        
        # Generate feedback with unified approach
        try:
            print(f"  - Generating unified feedback...")
            unified_feedback = self.generate_feedback_unified(problem_data)
            unified_path = self.save_feedback(
                unified_feedback, model_name, 'unified', problem_id
            )
            results['unified'] = {
                'status': 'success',
                'output_path': str(unified_path),
                'feedback': unified_feedback
            }
        except Exception as e:
            print(f"  - Error in unified approach: {e}")
            results['unified'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Generate feedback with multi-agent approach
        try:
            print(f"  - Generating multi-agent feedback...")
            multi_agent_feedback = self.generate_feedback_multi_agent(problem_data)
            multi_agent_path = self.save_feedback(
                multi_agent_feedback, model_name, 'multi_agent', problem_id
            )
            results['multi_agent'] = {
                'status': 'success',
                'output_path': str(multi_agent_path),
                'feedback': multi_agent_feedback
            }
        except Exception as e:
            print(f"  - Error in multi-agent approach: {e}")
            results['multi_agent'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Store ground truth for reference
        results['ground_truth'] = {
            'bug_description': problem_data['bug_description'],
            'bug_fixes': problem_data['bug_fixes']
        }
        
        return results


class FeedbackPipeline:
    """
    Main pipeline for batch processing all problems
    """
    
    def __init__(self, llm, model_name: str, dataset_dir: str = "Datasets",
                 output_dir: str = "generated_feedback"):
        """
        Initialize the pipeline
        
        Args:
            llm: Language model instance
            model_name: Name of the model (e.g., 'qwen25_14b')
            dataset_dir: Directory containing problem files
            output_dir: Directory for output
        """
        if "deepseek" in model_name.lower():
            print("Think stripping ENABLED for model:", model_name)
            self.generator = FeedbackGenerator(llm, output_dir, strip_think_block=True)
        else:
            self.generator = FeedbackGenerator(llm, output_dir)
            
        self.model_name = model_name
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        
    def get_problem_files(self) -> List[Tuple[str, str]]:
        """
        Get all problem files from dataset directory
        
        Returns:
            List of (file_path, problem_id) tuples
        """
        problem_files = []
        
        # Look for .txt or .xml files in dataset directory
        for file_path in sorted(self.dataset_dir.glob("*.txt")):
            # Extract problem ID from filename (e.g., 'problem_001.txt' -> '001')
            problem_id = file_path.stem.replace('problem_', '')
            if not problem_id:
                problem_id = file_path.stem
            problem_files.append((str(file_path), problem_id))
        
        return problem_files
    
    def run_pipeline(self, max_problems: int = None) -> Dict:
        """
        Run the complete pipeline
        
        Args:
            max_problems: Maximum number of problems to process (None for all)
            
        Returns:
            Dictionary with all results
        """
        print(f"="*60)
        print(f"FEEDBACK GENERATION PIPELINE")
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.dataset_dir}")
        print(f"Output: {self.output_dir}")
        print(f"="*60)
        
        # Get all problem files
        problem_files = self.get_problem_files()
        
        if max_problems:
            problem_files = problem_files[:max_problems]
        
        print(f"\nFound {len(problem_files)} problems to process")
        
        # Process each problem
        all_results = []
        for idx, (file_path, problem_id) in enumerate(problem_files, 1):
            print(f"\n[{idx}/{len(problem_files)}] ", end="")
            
            result = self.generator.process_single_problem(
                file_path, problem_id, self.model_name, skip_existing=True
            )
            all_results.append(result)
            
            # Save intermediate progress
            if idx % 10 == 0:
                self.save_progress(all_results)
        
        # Save final results
        self.save_progress(all_results)
        
        # Generate summary
        summary = self.generate_summary(all_results)
        print(f"\n{summary}")
        
        return {
            'model_name': self.model_name,
            'total_problems': len(all_results),
            'results': all_results,
            'summary': summary
        }
    
    def save_progress(self, results: List[Dict]):
        """Save intermediate results to JSON"""
        progress_file = self.output_dir / self.model_name / "progress.json"
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(progress_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def generate_summary(self, results: List[Dict]) -> str:
        """Generate summary statistics"""
        unified_success = sum(1 for r in results 
                             if r.get('unified', {}).get('status') == 'success')
        multi_agent_success = sum(1 for r in results 
                                 if r.get('multi_agent', {}).get('status') == 'success')
        
        summary = f"""
PIPELINE SUMMARY
================
Total Problems: {len(results)}
Unified Approach: {unified_success}/{len(results)} successful
Multi-Agent Approach: {multi_agent_success}/{len(results)} successful

Output saved to: {self.output_dir / self.model_name}
"""
        return summary

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run the pipeline
    
    Usage:
        1. Set up your LLM
        2. Call this function with appropriate parameters
    """
    
    # Example usage (you'll need to replace with your actual LLM setup)
    
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
    model="casperhansen/llama-3.3-70b-instruct-awq",   # must match the vLLM model name
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",            # vLLM ignores api keys — put anything
    temperature=0.0
    )
    
    # Create and run pipeline
    pipeline = FeedbackPipeline(
        llm=llm,
        model_name="Llama-33_70B_awq",  # Change based on your model
        dataset_dir="Dataset",    # Your dataset directory
        output_dir="generated_feedback"
    )
    
    # Run for all problems (or specify max_problems for testing)
    results = pipeline.run_pipeline(max_problems= None)
    

    
    print("Pipeline setup complete!")
    print("\nTo run:")
    print("1. Initialize your LLM")
    print("2. Create FeedbackPipeline instance")
    print("3. Call run_pipeline()")
    
if __name__ == "__main__":
    main()