import os
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

from ragas_typhoon.rag import RAG
from ragas_typhoon.model import init_llm, init_embed
from ragas_typhoon.settings import RAGAS_APP_TOKEN

def generate_eval_dataset(querys, references):
    dataset = []

    for query,reference in zip(querys,references):

        relevant_docs = rag.get_most_relevant_docs(query)
        response = rag.generate_answer(query, relevant_docs)
        
        dataset.append(
            {
                "user_input":query,
                "retrieved_contexts":relevant_docs,
                "response":response,
                "reference":reference
            }
        )

    evaluation_dataset = EvaluationDataset.from_list(dataset)
    
    return evaluation_dataset


if __name__ == "__main__":
    os.environ["RAGAS_APP_TOKEN"] = RAGAS_APP_TOKEN
    
    sample_docs = [
    "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
    "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
    "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
    "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'.",
    "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine."
    ]
    
    sample_queries = [
    "Who introduced the theory of relativity?",
    "Who was the first computer programmer?",
    "What did Isaac Newton contribute to science?",
    "Who won two Nobel Prizes for research on radioactivity?",
    "What is the theory of evolution by natural selection?"
    ]

    expected_responses = [
        "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
        "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine.",
        "Isaac Newton formulated the laws of motion and universal gravitation, laying the foundation for classical mechanics.",
        "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity and won two Nobel Prizes.",
        "Charles Darwin introduced the theory of evolution by natural selection in his book 'On the Origin of Species'."
    ]
    
    rag = RAG()
    rag.load_documents(sample_docs)
    evaluation_dataset = generate_eval_dataset(sample_queries, expected_responses)
    evaluator_llm = LangchainLLMWrapper(init_llm())
    metrics = [LLMContextRecall(), Faithfulness(), FactualCorrectness()]
    result = evaluate(dataset=evaluation_dataset,metrics=metrics,llm=evaluator_llm)
    result.upload()
    