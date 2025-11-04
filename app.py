"""
🤖 DialoGPT Meme-Style Chatbot
Author: Charan Siddarth
Year: 2025

A fun meme-style AI chatbot built using Microsoft's DialoGPT-medium model.
Runs as a Gradio web app. Replies with playful, emoji-rich responses.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr

# ---- Load model and tokenizer ----
MODEL_NAME = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# ---- Meme personality wrapper ----
def meme_response(text):
    meme_suffixes = [
        "😂", "🔥", "💀", "🤣", "😎", "🤖", "✨", 
        "lol", "bruh", "no cap", "fr fr", "💯", "ngl"
    ]
    import random
    return text + " " + random.choice(meme_suffixes)

# ---- Chat handler ----
def chat_fn(message, history):
    history_text = ""
    for human, bot in history:
        history_text += f"User: {human}\nBot: {bot}\n"

    prompt = history_text + f"User: {message}\nBot:"

    input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')

    output = model.generate(
        input_ids,
        max_length=250,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.9,
        top_p=0.92
    )

    reply = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return meme_response(reply)

# ---- Gradio UI ----
gr.ChatInterface(
    fn=chat_fn,
    title="DialoGPT Meme Chatbot 🤖🔥",
    description="A fun meme-style AI chatbot powered by Hugging Face Transformers.",
    type="messages"
).launch()
