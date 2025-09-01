# import streamlit as st
# import threading
# from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
# import torch

# # Load model & tokenizer (cache will speed this up after first run)
# @st.cache_resource
# def load_model():
#     model_name = "Qwen/Qwen2.5-0.5B-Instruct"
#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         torch_dtype="auto",
#         device_map="auto"
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     if not tokenizer.pad_token:
#         tokenizer.pad_token = tokenizer.eos_token
#     return model, tokenizer

# model, tokenizer = load_model()

# # Session state for conversation history
# if "messages" not in st.session_state:
#     st.session_state.messages = [
#         {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}
#     ]

# st.title("💬 Chat with Qwen")
# st.markdown("Type your message and press **Enter** to chat. Messages stream live.")

# # Chat UI: show past messages
# for msg in st.session_state.messages:
#     if msg["role"] == "user":
#         with st.chat_message("user"):
#             st.markdown(msg["content"])
#     elif msg["role"] == "assistant":
#         with st.chat_message("assistant"):
#             st.markdown(msg["content"])

# # Input box (press Enter to send)
# if prompt := st.chat_input("Type your message..."):
#     # Add user message
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Prepare input for model
#     chat_prompt = tokenizer.apply_chat_template(
#         st.session_state.messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#     inputs = tokenizer([chat_prompt], return_tensors="pt").to(model.device)

#     # Stream response
#     streamer = TextIteratorStreamer(
#         tokenizer,
#         skip_prompt=True,
#         skip_special_tokens=True
#     )

#     def generate():
#         model.generate(
#             **inputs,
#             streamer=streamer,
#             max_new_tokens=1000,   # adjust as needed
#             pad_token_id=tokenizer.eos_token_id,
#             do_sample=True,
#             temperature=0.7,
#         )

#     thread = threading.Thread(target=generate)
#     thread.start()

#     # Display streaming output
#     response_container = st.chat_message("assistant")
#     response_text = st.empty()
#     partial_output = ""

#     for new_text in streamer:
#         partial_output += new_text
#         response_text.markdown(partial_output)

#     # Add assistant message to history
#     st.session_state.messages.append({"role": "assistant", "content": partial_output})


import streamlit as st
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from duckduckgo_search import DDGS

# ---------------------------
# Load Qwen Model
# ---------------------------
@st.cache_resource
def load_model():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

model, tokenizer = load_model()

# ---------------------------
# DuckDuckGo search helper
# ---------------------------
def web_search(query, max_results=5):
    ddgs = DDGS()
    results = ddgs.text(query, max_results=max_results)
    return results  # dicts with {title, href, body}

# ---------------------------
# Streamlit App
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are Qwen, a helpful assistant with access to live web search results."}
    ]

st.title("🔎 Chat with Qwen + Live Web Search (Perplexity-style)")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Fetch search results ---
    search_results = web_search(prompt, max_results=5)
    snippets = [f"{r['title']} ({r['href']}): {r['body']}" for r in search_results]

    # Show retrieved sources
    with st.expander("📚 Sources from the web"):
        for r in search_results:
            st.markdown(f"- [{r['title']}]({r['href']})  \n  {r['body']}")

    # --- Build augmented prompt for Qwen ---
    search_context = "\n".join(snippets)
    augmented_prompt = f"""
    Use the following search results to answer the question.
    Be concise, factual, and cite useful information.

    Search results:
    {search_context}

    Question: {prompt}
    """

    chat_prompt = tokenizer.apply_chat_template(
        st.session_state.messages + [{"role": "user", "content": augmented_prompt}],
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([chat_prompt], return_tensors="pt").to(model.device)

    # --- Stream response ---
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    def generate():
        model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=800,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
        )

    thread = threading.Thread(target=generate)
    thread.start()

    response_container = st.chat_message("assistant")
    response_text = st.empty()
    partial_output = ""

    for new_text in streamer:
        partial_output += new_text
        response_text.markdown(partial_output)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": partial_output})
