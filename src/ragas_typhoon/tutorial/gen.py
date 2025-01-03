from ragas_typhoon.model import init_llm, init_embed
llm = init_llm()
embed = init_embed()

from langchain_community.document_loaders import DirectoryLoader

path = "src/ragas_typhoon/data/sample/Sample_Docs_Markdown"
loader = DirectoryLoader(path, glob="**/*.md")
docs = loader.load()

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
# generator_llm = LangchainLLMWrapper(llm)
# generator_embeddings = LangchainEmbeddingsWrapper(embed)
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas_typhoon.settings import OPENAI_KEY, RAGAS_API_KEY
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_KEY))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=OPENAI_KEY))

from ragas.testset import TestsetGenerator

generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
dataset = generator.generate_with_langchain_docs(docs, testset_size=10)

dataset.to_pandas()
import os
os.environ["RAGAS_APP_TOKEN"] = RAGAS_API_KEY

dataset.upload()

from ragas.testset.graph import KnowledgeGraph

kg = KnowledgeGraph()

from ragas.testset.graph import Node, NodeType

for doc in docs:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
        )
    )

from ragas.testset.transforms import default_transforms, apply_transforms


# define your LLM and Embedding Model
# here we are using the same LLM and Embedding Model that we used to generate the testset
transformer_llm = generator_llm
embedding_model = generator_embeddings

trans = default_transforms(documents=docs, llm=transformer_llm, embedding_model=embedding_model)
apply_transforms(kg, trans)

kg.save("knowledge_graph.json")
loaded_kg = KnowledgeGraph.load("knowledge_graph.json")

from ragas.testset import TestsetGenerator

generator = TestsetGenerator(llm=generator_llm, embedding_model=embedding_model, knowledge_graph=loaded_kg)

from ragas.testset.synthesizers import default_query_distribution

query_distribution = default_query_distribution(generator_llm)

testset = generator.generate(testset_size=10, query_distribution=query_distribution)
testset.to_pandas()

testset.upload()

