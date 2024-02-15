
from langchain_community.llms import LlamaCpp
from langchain.callbacks import StreamingStdOutCallbackHandler
from operator import itemgetter
from langchain.prompts import PromptTemplate
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.memory import ConversationBufferWindowMemory
from indexing import load_vector_store
import langchain
langchain.verbose = True

MISTRAL_MODEL = 'models/mistral-7b-instruct-v0.2.Q4_K_M.gguf'
n_gpu_layers = 16 
n_batch = 256
context_window = 4200

llm = LlamaCpp(model_path = MISTRAL_MODEL,
                n_threads= 4,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],
                n_gpu_layers = n_gpu_layers,
                n_batch = n_batch,
                n_ctx = context_window,
                stop = ['\n\n', '\nAssistant:', '\nUser:', '[INST]', '[/INST]'],
                temperature = 0.1)

template = """<s>[INST] Assistant is an expert, created to answer questions related to the use of the "UWB Tuning Tool" app at Continental.
When asked questions about how to use the application, Assistant answers based on the context provided, if it is relevant.
Assistant doesn't not mention the use of the context, doesn't make up false information and says when it doesn't know the answer.
When no context is provided, Assistant says it doesn't know the answer and asks the user to rephrase their question.
----------------------------
Context(s) :
{context}
\n----------------------------
The following is a conversation between Assistant and a user:
[/INST] </s>
{chat_history}
[INST]User: {question}[/INST]
Assistant: """ 

QA_CHAIN_PROMPT = PromptTemplate.from_template(template=template)

vector_store = load_vector_store()

#define retriever
retriever = vector_store.as_retriever(search_type="similarity_score_threshold",
                                      search_kwargs={'score_threshold': 0.6})

#define memory
memory = ConversationBufferWindowMemory(
    output_key="answer",
    input_key="question",
    human_prefix="[INST]User: ",
    ai_prefix="Assistant: ",
    k = 2
)

#step to load memory

chain = (
    {"context": retriever, 
     "question": RunnablePassthrough(),
     "chat_history": RunnableLambda(memory.load_memory_variables) | itemgetter("history")}
    | QA_CHAIN_PROMPT
    | llm
)

def run_chain(query):
    result = chain.invoke(query, config = {'callbacks' : [ConsoleCallbackHandler()]})
    memory.chat_memory.add_user_message(query + "[/INST]")
    memory.chat_memory.add_ai_message(result)