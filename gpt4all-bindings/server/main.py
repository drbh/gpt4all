from fastapi import FastAPI
from typing import List, Optional
from gpt4all import GPT4All
from typing_extensions import Annotated
from pydantic import BaseModel

class Message(BaseModel):
    role: str
    content: str

class UserInput(BaseModel):
    message: str

class ChatCompletionResponse(BaseModel):
    messages: List[Message]

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello there."},
    {"role": "assistant", "content": "Hi, how can I help you?"},
]


MODEL: str = "ggml-gpt4all-j-v1.3-groovy"
N_THREADS: int = 6

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global gpt4all_instance
    gpt4all_instance = GPT4All(MODEL)

    # if threads are passed, set them
    if N_THREADS != 4:
        # set number of threads
        gpt4all_instance.model.set_thread_count(N_THREADS)

@app.post("/message", response_model=ChatCompletionResponse)
async def chat(input: UserInput):
    MESSAGES.append({"role": "user", "content": input.message})

    # execute chat completion
    full_response = gpt4all_instance.chat_completion(
        MESSAGES,
        # preferential kwargs for chat ux
        logits_size=0,
        tokens_size=0,
        n_past=0,
        n_ctx=0,
        n_predict=100,
        top_k=5,
        top_p=0.9,
        temp=0.2,
        n_batch=4,
        repeat_penalty=1.1,
        repeat_last_n=64,
        context_erase=0.0,
        verbose=False,
    )

    # record assistant's response to messages
    MESSAGES.append(full_response.get("choices")[0].get("message"))
    
    return {"messages": MESSAGES}

@app.get("/messages", response_model=List[Message])
async def get_messages():
    return MESSAGES
