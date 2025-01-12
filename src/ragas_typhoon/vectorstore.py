from ragas_typhoon.model import init_embed, init_llm

# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from tqdm import tqdm
import json
import os 


def vs(DATA_DIR, METADATA_DIR, model=None):
    data = []
    
    for file in os.listdir(DATA_DIR): 
        if file.endswith(".txt"):
            FILE_DIR = os.path.join(DATA_DIR, file)
            with open(FILE_DIR, "r", encoding="utf-8") as file:
                content = file.read()
            data.append(content)
            
    with open(METADATA_DIR, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    
    print("Data Loaded")   

    metadata = [
        {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}
        for meta in metadata
    ]
    
    embeddings = init_embed(model)
    
    print("Creating vectorstore...")
    
    Chroma.from_texts(
        texts=tqdm(data, desc="Processing texts"),
        metadatas=metadata,
        embedding=embeddings,
        collection_name="chroma_db",
        persist_directory="./chroma_db"
    )
    
    print("Vectorstore created successfully")

def retrieve_context(query, k=3, model=None):
    # document_content_description = "Example conversation"
    # metadata_field_info = [
    # AttributeInfo(
    #     name="type",
    #     description="The type of the question",
    #     type="string",
    # ),
    # AttributeInfo(
    #     name="question",
    #     description="The question asked by the user",
    #     type="string",
    # ),
    # ]
    # llm = init_llm(model)
    embeddings = init_embed(model)
    vectorstore = Chroma(
        collection_name="chroma_db",
        persist_directory="./chroma_db",
        embedding_function=embeddings)
    # retriever = vectorstore.as_retriever()
    # documents = retriever.invoke(query)
    # return documents
    
    # retriever = SelfQueryRetriever.from_llm(
    # llm,
    # vectorstore,
    # document_content_description,
    # metadata_field_info,
    # )
    # return retriever.invoke(query)
    return vectorstore.similarity_search(query=query, k=k)
        
if __name__ == "__main__":
    if not os.path.exists("./chroma_db"):
        DATA_DIR = "src/ragas_typhoon/data/invx-general-finance/finance_invx/data/text"
        METADATA_DIR = "src/ragas_typhoon/data/invx-general-finance/finance_invx/finance_metadata.json"
        
        vs(DATA_DIR, METADATA_DIR)
        
    context = retrieve_context("Alphabet Inc.")
    print(context[0])
    

