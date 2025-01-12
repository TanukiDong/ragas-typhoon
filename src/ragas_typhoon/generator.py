from ragas_typhoon.model import init_llm
from ragas_typhoon.vectorstore import retrieve_context

from pathlib import Path
import json
import re

def load_data(DIR):
    with open(DIR, 'r') as file:
        data = json.load(file)
    return data

def extract_messages(question):
    chat_history = []
    keywords = ["customer :", "customer:", "user:", "user :", "agent :", "agent:"]
    
    if any(keyword in question.lower() for keyword in keywords):
        msgs = question.lower().split("\n")
        for msg in msgs:
            if "agent" in msg:
                chat_history.append({
                    "role": "assistant",
                    "content" : msg.split(":")[1].strip()
                    })
            elif "customer" in msg or "user" in msg:
                chat_history.append({
                    "role": "user",
                    "content" : msg.split(":")[1].strip()
                    })
            else:
                msg_ = chat_history.pop(-1)
                chat_history.append(msg_ | {"content" : "\n".join([msg_["content"], msg])})
    else:
        chat_history.append({
            "role": "user",
            "content" : question.strip()
        })
    return chat_history
    
    

def call_llm(question, chat_history, model=None):
    llm = init_llm(model)
    context = retrieve_context(question)
    SYSTEM_PROMPT = f"""\
    # Instruction
    You are a helpful assistant that answers the question based on a given context. If you can't answer, reply that you don't know. Always answer in Thai.
    
    # Context
    {context}
    """
    
    # print(SYSTEM_PROMPT)
    # print(question)
    # print()
    # import time
    # time.sleep(5)
    messages = [{"role" : "system", "content" : SYSTEM_PROMPT}]
    messages.extend(chat_history)
    llm_response = llm.invoke(messages)
    return llm_response.content
    



if __name__ == "__main__":

    data = load_data("src/ragas_typhoon/data/input/finance_cleaned.json")

    for record in data:
        question = record.get("question")
        chat_history = extract_messages(question)
        llm_response = call_llm(question, chat_history)
        
        # record = record | {"response" : llm_response}
            


