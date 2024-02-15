import asyncio
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from model import initialize_chain
from typing import AsyncIterable, List
from pydantic import BaseModel

class StreamChatRequest(BaseModel):
    content: str
    queries: List[str]
    answers: List[str]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

inference_queue = asyncio.Queue()

chain, memory, async_callback = initialize_chain()

async def add_to_queue(query, response_queue, queries_list, answers_list):
    memory.chat_memory.clear()
    for zip_query, zip_answer in zip(queries_list, answers_list):
        memory.chat_memory.add_user_message(zip_query +'[/INST]')
        memory.chat_memory.add_ai_message(zip_answer)
    print(query)
    print(memory.chat_memory)
    await inference_queue.put((query, response_queue))

async def worker():
    while True:
        query, response_queue = await inference_queue.get()  # Expect a tuple of (query, response_queue)
        async_callback.done.clear()  # Clear the done flag for the new request
        try:
            await send_message(query, response_queue)
        except Exception as e:
            await response_queue.put(f"Error: {e}")  # Send error to the response queue
        finally:
            await response_queue.put(None)  # Indicate the end of the response
            inference_queue.task_done() 

async def send_message(query: str, response_queue: asyncio.Queue) -> None:
    task = asyncio.create_task(
        chain.ainvoke(query)
    )

    try:
        async for token in async_callback.aiter():
            await response_queue.put(token)  # Put each token in the response queue
    except Exception as e:
        raise e
    finally:
        async_callback.done.set()

    await task

async def stream_response(response_queue: asyncio.Queue) -> AsyncIterable[str]:
    while True:
        response = await response_queue.get()
        if response is None:  # Check for end of the response
            break
        yield response

@app.get("/health")
async def health_check():
    try:
         return JSONResponse(content={"status": "alive"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream_chat")
async def stream_chat(request: StreamChatRequest):
    response_queue = asyncio.Queue()
    print(request.content)
    await add_to_queue(query = request.content,
                       response_queue=response_queue,
                       queries_list=request.queries,
                       answers_list=request.answers)
    return StreamingResponse(stream_response(response_queue), media_type="text/event-stream")

# Start the worker in the background
asyncio.create_task(worker())