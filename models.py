import asyncio
from indexing import load_vector_store
import langchain
from langchain.callbacks import StreamingStdOutCallbackHandler,  AsyncIteratorCallbackHandler
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter
import os
from pydantic import BaseModel
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# langchain.verbose = True

#message and question basemodels

class Query(BaseModel):
    content: str

class Answer(BaseModel):
    content: str

LLAMA_7B = 'models/llama-2-7b-chat.Q4_K_M.gguf'
LLAMA_13B = 'models/llama-2-13b-chat.Q4_K_M.gguf'
MISTRAL_MODEL_Q4 = 'models/mistral-7b-instruct-v0.2.Q4_K_M.gguf'
MISTRAL_MODEL_Q5 = 'models/mistral-7b-instruct-v0.2.Q5_0.gguf'

n_gpu_layers = 33
n_batch = 128
context_window = 4192

def initialize_chain():
    callback = AsyncIteratorCallbackHandler()

    llm = LlamaCpp(model_path = MISTRAL_MODEL_Q4,
                    streaming=True,
                    callbacks=[callback, StreamingStdOutCallbackHandler(),ConsoleCallbackHandler()],
                    n_gpu_layers = n_gpu_layers,
                    n_batch = n_batch,
                    n_ctx = context_window,
                    stop = ['\nAssistant:', '\nUser:', 'Assistant:', '---', 'Context(s)', 'User:', '[INST]', '[/INST]', '#', '[', ']' ,'[]'],
                    temperature = 0.1)

    template = """<s>[INST] Assistant is an expert, created to answer questions related to the use of the "UWB Tuning Tool" app at Continental.
    Assistant adheres strictly to the context provided and doesn't add or make up any extra information.
    If no context is provided, assistant doesn't answer the question and asks the user to rephrase their question.
    ------------------------------
    Context :
    {context}
    \n----------------------------
    #The following is a conversation between Assistant and a user:
    # [/INST] </s>
    # {chat_history}
    [INST]User: {question}[/INST]
    Assistant: """ 

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template=template)

    vector_store = load_vector_store()

    #define retriever
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold",
                                        search_kwargs={'score_threshold': 0.63})

    #define memory
    memory = ConversationBufferWindowMemory(
        output_key="answer",
        input_key="question",
        human_prefix="[INST]User: ",
        ai_prefix="Assistant: ",
        k = 3
    )

    #step to load memory

    chain = (
        {"context": retriever, 
        "question": RunnablePassthrough(),
        "chat_history": RunnableLambda(memory.load_memory_variables) | itemgetter("history")}
        | QA_CHAIN_PROMPT
        | llm
    )

    # return chain, memory, callback
    return chain, memory, callback
