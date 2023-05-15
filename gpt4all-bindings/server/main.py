from fastapi import FastAPI
from typing import List, Optional
from gpt4all import GPT4All
from typing_extensions import Annotated
from pydantic import BaseModel
import sqlite3
from sqlite3 import Error
import os


class Message(BaseModel):
    role: str
    content: str
    chat_id: int
    timestamp: Optional[str]


class UserInput(BaseModel):
    message: str
    chat_id: int


class ChatCompletionResponse(BaseModel):
    messages: List[Message]


INIT_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello there."},
    {"role": "assistant", "content": "Hi, how can I help you?"},
]

MODEL: str = "ggml-gpt4all-j-v1.3-groovy"
N_THREADS: int = 6

DATABASE = "chat_history.db"

app = FastAPI()


def create_connection():
    conn = None
    try:
        conn = sqlite3.connect(DATABASE)
        return conn
    except Error as e:
        print(e)
    return conn


def create_table(conn):
    try:
        sql_create_table = """CREATE TABLE IF NOT EXISTS messages (
                                        id integer PRIMARY KEY AUTOINCREMENT,
                                        role text NOT NULL,
                                        content text NOT NULL,
                                        chat_id integer NOT NULL,
                                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                                    ); """
        c = conn.cursor()
        c.execute(sql_create_table)
    except Error as e:
        print(e)


@app.on_event("startup")
async def startup_event():
    global gpt4all_instance
    gpt4all_instance = GPT4All(MODEL)

    # if threads are passed, set them
    if N_THREADS != 4:
        # set number of threads
        gpt4all_instance.model.set_thread_count(N_THREADS)

    conn = create_connection()
    if conn is not None:
        create_table(conn)
    else:
        print("Error! Cannot create the database connection.")


@app.post("/message", response_model=ChatCompletionResponse)
async def chat(input: UserInput):
    conn = create_connection()
    cursor = conn.cursor()

    # check if there is any message in the db with the chat id
    is_new_chat = cursor.execute(
        "SELECT * FROM messages WHERE chat_id=?", (input.chat_id,)
    ).fetchone()

    if is_new_chat is None:
        # add all init messages to the db
        for message in INIT_MESSAGES:
            cursor.execute(
                "INSERT INTO messages(role, content, chat_id) VALUES(?, ?, ?)",
                (message["role"], message["content"], input.chat_id),
            )
            conn.commit()

    # Insert user's message to the database
    cursor.execute(
        "INSERT INTO messages(role, content, chat_id) VALUES(?, ?, ?)",
        ("user", input.message, input.chat_id),
    )
    conn.commit()

    # Fetch all messages for the chat id
    cursor.execute("SELECT * FROM messages WHERE chat_id=?", (input.chat_id,))
    MESSAGES = [
        {"role": row[1], "content": row[2], "chat_id": row[3]}
        for row in cursor.fetchall()
    ]

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

    # Add assistant's response to the database
    cursor.execute(
        "INSERT INTO messages(role, content, chat_id) VALUES(?, ?, ?)",
        (
            "assistant",
            full_response.get("choices")[0].get("message").get("content"),
            input.chat_id,
        ),
    )
    conn.commit()

    # Fetch all messages for the chat id
    cursor.execute("SELECT * FROM messages WHERE chat_id=?", (input.chat_id,))
    MESSAGES = [
        {"role": row[1], "content": row[2], "chat_id": row[3]}
        for row in cursor.fetchall()
    ]
    conn.close()

    return {"messages": MESSAGES}


@app.get("/messages/{chat_id}", response_model=List[Message])
async def get_messages(chat_id: int):
    conn = create_connection()
    cursor = conn.cursor()

    # Fetch all messages for the chat id
    cursor.execute("SELECT * FROM messages WHERE chat_id=?", (chat_id,))
    MESSAGES = [
        {"role": row[1], "content": row[2], "chat_id": row[3]}
        for row in cursor.fetchall()
    ]

    conn.close()

    return MESSAGES


# List all chat ids and their last message and time timestamp
@app.get("/chats", response_model=List[Message])
async def get_chats():
    conn = create_connection()
    cursor = conn.cursor()

    # Fetch all messages for the chat id
    cursor.execute(
        "SELECT chat_id, role, content, timestamp, MAX(id) FROM messages GROUP BY chat_id"
    )
    chats = [
        {
            "chat_id": row[0], 
            "content": row[2],
            "role": row[1],
            # convert timestamp to string
            "timestamp": row[3]
        } for row in cursor.fetchall()
    ]

    print(chats)
    conn.close()

    return chats

# Search messages for text string
@app.get("/search/{text}", response_model=List[Message])
async def search_messages(text: str):
    conn = create_connection()
    cursor = conn.cursor()

    # Fetch all messages for the chat id
    cursor.execute(
        "SELECT * FROM messages WHERE content LIKE ?", ('%' + text + '%',)
    )
    searchResults = [
        {"role": row[1], "content": row[2], "chat_id": row[3]}
        for row in cursor.fetchall()
    ]

    conn.close()

    return searchResults