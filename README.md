# 🔎 Chat with Qwen + Live Web Search

This project explores how to run **Hugging Face models locally** and build a simple **LLM-powered chatbot** with **live web search integration**.  
It uses [Qwen](https://huggingface.co/Qwen) (by Alibaba) for natural language generation and [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/) to fetch fresh information from the web.  
The interface is built with [Streamlit](https://streamlit.io/).

---

## ✨ Features
- ✅ Run **Qwen LLM locally** via Hugging Face Transformers  
- ✅ **Chat UI** with Streamlit  
- ✅ **DuckDuckGo-powered live search**  
- ✅ Conversational history stored in session state  
- ✅ Real-time **streaming responses**  

---

## 🚀 Getting Started

### 1️⃣ Clone this repo
```bash
git clone https://github.com/birijepallisiva/qwen3_local_llm.git
cd hugging_face
```

### 2️⃣ Create a virtual environment

On **Windows**:
```bash
python -m venv venv
.
env\Scripts ctivate
```

On **Linux / macOS**:
```bash
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies
Install everything with:
```bash
pip install streamlit transformers torch duckduckgo-search
```

### 4️⃣ Run the app
```bash
streamlit run qwen3_local_llm.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📂 Project Structure
```
.
├── huggingface/qwen3_local_llm.py              # Main Streamlit app
└── README.md           # Project documentation
```

---

## 🛠 How It Works
1. User asks a question in the Streamlit chat UI.  
2. The app fetches **live search results** from DuckDuckGo.  
3. The search results are appended to the user’s query.  
4. Qwen LLM generates a **context-aware answer** using Hugging Face Transformers.  
5. The response is streamed back to the chat interface in real-time.  

---

## 📌 Example
**User:**  
> Who won the FIFA World Cup in 2022?  

**Assistant:**  
> According to recent search results, **Argentina** won the FIFA World Cup 2022 by defeating **France** in the final. 🏆  

(Sources are expandable under **📚 Sources from the web**.)

---

## 🎯 Goal
The main goal of this project is to **explore Hugging Face models** and learn how to **run a local LLM** with external knowledge (via web search).  
It can be extended into a **personal research assistant** or **Perplexity-style chatbot**.

---

## 📖 References
- [Qwen models on Hugging Face](https://huggingface.co/Qwen)  
- [Streamlit Documentation](https://docs.streamlit.io/)  
- [DuckDuckGo Search API](https://pypi.org/project/duckduckgo-search/)  
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)  

---
