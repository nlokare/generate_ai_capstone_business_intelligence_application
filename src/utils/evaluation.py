"""
Model evaluation module using QAEvalChain and custom metrics.
"""

from typing import List, Dict, Any, Tuple
import pandas as pd
from langchain_openai import ChatOpenAI

try:
    from langchain.evaluation import QAEvalChain
    EVAL_AVAILABLE = True
except ImportError:
    EVAL_AVAILABLE = False


class ModelEvaluator:
    """Evaluates the RAG system performance."""

    def __init__(self, llm: ChatOpenAI = None):
        """
        Initialize evaluator.

        Args:
            llm: LLM to use for evaluation (if None, creates default)
        """
        self.llm = llm or ChatOpenAI(model="gpt-4", temperature=0)
        if EVAL_AVAILABLE:
            self.eval_chain = QAEvalChain.from_llm(self.llm)
        else:
            self.eval_chain = None

    def evaluate_predictions(
        self,
        questions: List[str],
        predictions: List[str],
        ground_truths: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate model predictions against ground truth.

        Args:
            questions: List of questions
            predictions: List of model predictions
            ground_truths: List of ground truth answers

        Returns:
            Dictionary with evaluation metrics
        """
        # Simple evaluation without QAEvalChain
        # Compare predictions with ground truth using basic string matching
        results = []
        correct = 0

        for q, p, gt in zip(questions, predictions, ground_truths):
            # Simple heuristic: check if key numbers/words from ground truth appear in prediction
            is_correct = self._simple_match(p, gt)
            if is_correct:
                correct += 1
            results.append({
                "text": "CORRECT" if is_correct else "INCORRECT",
                "question": q,
                "prediction": p,
                "ground_truth": gt
            })

        total = len(results)
        accuracy = (correct / total) * 100 if total > 0 else 0

        return {
            "total_examples": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": accuracy,
            "results": results
        }

    def _simple_match(self, prediction: str, ground_truth: str) -> bool:
        """Simple matching heuristic."""
        # Extract numbers from both strings
        import re
        pred_numbers = set(re.findall(r'\d+\.?\d*', prediction))
        gt_numbers = set(re.findall(r'\d+\.?\d*', ground_truth))

        # If there are numbers, check if they match
        if gt_numbers:
            return len(pred_numbers.intersection(gt_numbers)) > 0

        # Otherwise, check for key word overlap
        pred_words = set(prediction.lower().split())
        gt_words = set(ground_truth.lower().split())
        common_words = pred_words.intersection(gt_words)

        # Consider correct if more than 30% of words match
        return len(common_words) / len(gt_words) > 0.3 if gt_words else False

    def create_test_cases(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """
        Create test cases from the data.

        Args:
            df: Business data DataFrame

        Returns:
            List of test cases with questions and expected answers
        """
        test_cases = []

        # Total records test
        test_cases.append({
            "question": "How many total records are in the dataset?",
            "expected_answer": f"There are {len(df)} records in the dataset."
        })

        # Sales statistics tests (if sales column exists)
        sales_cols = [col for col in df.columns if 'sales' in col.lower() or 'revenue' in col.lower()]
        if sales_cols:
            sales_col = sales_cols[0]
            total_sales = df[sales_col].sum()
            avg_sales = df[sales_col].mean()

            test_cases.append({
                "question": "What is the total sales?",
                "expected_answer": f"The total sales is ${total_sales:,.2f}."
            })

            test_cases.append({
                "question": "What is the average sale amount?",
                "expected_answer": f"The average sale amount is ${avg_sales:,.2f}."
            })

        # Product tests (if product column exists)
        product_cols = [col for col in df.columns if 'product' in col.lower()]
        if product_cols:
            product_col = product_cols[0]
            num_products = df[product_col].nunique()

            test_cases.append({
                "question": "How many unique products are there?",
                "expected_answer": f"There are {num_products} unique products."
            })

            if sales_cols:
                top_product = df.groupby(product_col)[sales_cols[0]].sum().idxmax()
                test_cases.append({
                    "question": "What is the top-selling product?",
                    "expected_answer": f"The top-selling product is {top_product}."
                })

        # Regional tests (if region column exists)
        region_cols = [col for col in df.columns if 'region' in col.lower()]
        if region_cols:
            region_col = region_cols[0]
            num_regions = df[region_col].nunique()

            test_cases.append({
                "question": "How many regions are covered?",
                "expected_answer": f"There are {num_regions} regions covered."
            })

        return test_cases

    def run_evaluation(
        self,
        rag_system: Any,
        test_cases: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Run evaluation on the RAG system.

        Args:
            rag_system: RAG system to evaluate
            test_cases: List of test cases

        Returns:
            Evaluation results
        """
        questions = [tc["question"] for tc in test_cases]
        expected_answers = [tc["expected_answer"] for tc in test_cases]

        # Get predictions
        predictions = []
        for question in questions:
            answer = rag_system.query(question)
            predictions.append(answer)

        # Evaluate
        results = self.evaluate_predictions(questions, predictions, expected_answers)

        # Add detailed comparison
        results["detailed_results"] = [
            {
                "question": q,
                "expected": e,
                "predicted": p,
                "evaluation": r.get("text", "unknown")
            }
            for q, e, p, r in zip(questions, expected_answers, predictions, results["results"])
        ]

        return results

    def calculate_response_metrics(
        self,
        responses: List[str]
    ) -> Dict[str, float]:
        """
        Calculate metrics about response quality.

        Args:
            responses: List of response texts

        Returns:
            Dictionary with response metrics
        """
        if not responses:
            return {}

        lengths = [len(r) for r in responses]
        word_counts = [len(r.split()) for r in responses]

        return {
            "avg_response_length": sum(lengths) / len(lengths),
            "avg_word_count": sum(word_counts) / len(word_counts),
            "min_response_length": min(lengths),
            "max_response_length": max(lengths),
            "total_responses": len(responses)
        }


def create_evaluation_report(
    evaluation_results: Dict[str, Any],
    save_path: str = None
) -> pd.DataFrame:
    """
    Create a detailed evaluation report.

    Args:
        evaluation_results: Results from evaluation
        save_path: Optional path to save CSV report

    Returns:
        DataFrame with evaluation results
    """
    detailed_results = evaluation_results.get("detailed_results", [])

    if not detailed_results:
        return pd.DataFrame()

    df = pd.DataFrame(detailed_results)

    if save_path:
        df.to_csv(save_path, index=False)

    return df


def generate_test_questions(df: pd.DataFrame) -> List[str]:
    """
    Generate diverse test questions based on the data.

    Args:
        df: Business data DataFrame

    Returns:
        List of test questions
    """
    questions = [
        "What is the overview of the dataset?",
        "How many records are in the data?",
        "What are the main columns in the dataset?"
    ]

    # Sales questions
    sales_cols = [col for col in df.columns if 'sales' in col.lower() or 'revenue' in col.lower()]
    if sales_cols:
        questions.extend([
            "What is the total sales?",
            "What is the average sale amount?",
            "What is the sales trend?",
            "Which period had the highest sales?"
        ])

    # Product questions
    product_cols = [col for col in df.columns if 'product' in col.lower()]
    if product_cols:
        questions.extend([
            "How many products are there?",
            "What are the top products?",
            "Which product has the best performance?",
            "Compare the top 3 products"
        ])

    # Regional questions
    region_cols = [col for col in df.columns if 'region' in col.lower()]
    if region_cols:
        questions.extend([
            "How many regions are covered?",
            "Which region has the best performance?",
            "Compare sales across regions"
        ])

    # Customer questions
    age_cols = [col for col in df.columns if 'age' in col.lower()]
    if age_cols:
        questions.extend([
            "What is the customer age distribution?",
            "What is the average customer age?"
        ])

    return questions
