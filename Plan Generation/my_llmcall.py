import os

from groq import Groq

def call_groq(messages, model="llama-3.3-70b-versatile"):
    client = Groq(
        api_key="",
    )

    chat_completion = client.chat.completions.create(
        messages=messages,
        model = model,
    )

    return (chat_completion.choices[0].message.content)    