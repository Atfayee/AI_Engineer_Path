import json
from datasets import Dataset
from pathlib import Path
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import OpenAI as RagasOpenAI
import os
from main import build_pipeline, retrieve_with_context, generate_answer


evaluator_llm = RagasOpenAI(
    model="gpt-3.5-turbo",
    api_key=os.getenv("OPENAI_API_KEY")
)

def load_golden(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def main():
    golden_path = Path(__file__).parent / "golden_set.json"
    golden_set = load_golden(golden_path)

    print("Building pipeline...")
    pipeline = build_pipeline()

    questions = []
    contexts_list = []
    answers = []
    ground_truths = []

    print("Generating answers...")

    for sample in golden_set:
        question = sample["question"]
        ground_truth = sample["golden_answer"]

        _, contexts = retrieve_with_context(pipeline=pipeline, query=question)

        answer = generate_answer(pipeline=pipeline, query=question)

        questions.append(question)
        contexts_list.append(contexts)
        answers.append(answer)
        ground_truths.append(ground_truth)
    
    dataset = Dataset.from_dict(
        {
            "question": questions,
            "contexts": contexts_list,
            "answer": answers,
            "ground_truth": ground_truths
        }
    )
    print("Running RASAS evaluation...")
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision
        ],
        llm=evaluator_llm
    )
    print("\n=== RAGAS Results ===")
    print(result)

if __name__ == "__main__":
    main()