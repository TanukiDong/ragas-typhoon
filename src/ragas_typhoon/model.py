from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.messages import HumanMessage

from ragas_typhoon.settings import TYPHOON_API_KEY, OPENAI_KEY
from functools import cache

@cache
def init_llm(t=None):
    if t == "o":
        return ChatOpenAI(model="gpt-4o-mini",
                          temperature=0,
                          api_key=OPENAI_KEY,
                          max_tokens=None,
                          timeout=None,
                          max_retries=2,
                          )
        
    return ChatOpenAI(base_url='https://api.opentyphoon.ai/v1',
                    model='typhoon-v2-8b-instruct',
                    api_key=TYPHOON_API_KEY,
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                    )
@cache
def init_embed(t=None):
    if t == "o":
        return OpenAIEmbeddings(api_key=OPENAI_KEY)
    
    # return SentenceTransformer('all-MiniLM-L6-v2')
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if __name__ == "__main__":
    llm = init_llm()
    resp = llm.invoke([HumanMessage(content="สวัสดี คุณเป็นโมเดลอะไร")])
    print(resp.content)
