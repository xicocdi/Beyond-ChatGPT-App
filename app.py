# flake8: noqa: E501
# type: ignore
# You can find this code for Chainlit python streaming here (https://docs.chainlit.io/concepts/streaming/python)

# OpenAI Chat completion
import os
from openai import AsyncOpenAI  # importing openai for API usage
import chainlit as cl  # importing chainlit for our app
from chainlit.prompt import Prompt, PromptMessage  # importing prompt tools
from chainlit.playground.providers.openai import (
    ChatOpenAI,
)  # importing ChatOpenAI tools
from dotenv import load_dotenv

load_dotenv()

# ChatOpenAI Templates
system_template = """You are a helpful assistant who specializes in assisting conversation designers generate training data for intents and bot responses that are appropriate for the given intent. When providing your recommendation of the bot reponse you should take on the persona of a gender-neutral chatbot that is helpful, empathetic, and knowledgeable. Also, your responses should be short and concise, no more than 2 sentences. And use emojis sparingly, only when appropriate.
"""

user_template = """{input}
Provide 10 diverse training phrases for each intent given and explain why the training phrases provided are appropriate. Then provide one bot response for the intent that is an appropriate response to each of the training phrases that could possibly key off the intent.
"""


@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    settings = {
        "model": "gpt-4",
        "temperature": 0,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    cl.user_session.set("settings", settings)
    cl.user_session.set("chat_history", [])


@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):
    settings = cl.user_session.get("settings")
    chat_history = cl.user_session.get("chat_history")

    client = AsyncOpenAI()

    print(message.content)

    # Append the new user message to the chat history
    chat_history.append({"role": "user", "content": message.content})

    prompt = Prompt(
        provider=ChatOpenAI.id,
        messages=[
            PromptMessage(
                role="system",
                template=system_template,
                formatted=system_template,
            ),
            *[
                PromptMessage(
                    role=msg["role"], template=msg["content"], formatted=msg["content"]
                )
                for msg in chat_history
            ],
            PromptMessage(
                role="user",
                template=user_template,
                formatted=user_template.format(input=message.content),
            ),
        ],
        inputs={"input": message.content},
        settings=settings,
    )

    if prompt.messages is not None:
        print([m.to_openai() for m in prompt.messages])

    msg = cl.Message(content="")

    try:
        # Call OpenAI
        async for stream_resp in await client.chat.completions.create(
            messages=[m.to_openai() for m in prompt.messages], stream=True, **settings
        ):
            token = stream_resp.choices[0].delta.content
            if not token:
                token = ""
            await msg.stream_token(token)

        # Update the prompt object with the completion
        prompt.completion = msg.content
        msg.prompt = prompt

        # Append the assistant's response to the chat history
        chat_history.append({"role": "assistant", "content": msg.content})
        cl.user_session.set("chat_history", chat_history)

        # Send and close the message stream
        await msg.send()
    except Exception as e:
        print(f"An error occurred: {e}")
        await cl.Message(
            content="An error occurred while processing your request. Please try again."
        ).send()
