# Brainlox course Q&A

A small Flask app that answers questions about the public [Brainlox technical courses](https://brainlox.com/courses/category/technical) page.

It downloads that page, splits the text into chunks, and ranks them by word overlap. With no API key, `/chat` returns the best excerpt. Set `GROQ_API_KEY` or `HF_API_TOKEN` in a `.env` file and the same excerpts are sent to Groq or the Hugging Face inference API. If that call fails, the excerpt is still returned.

This replaced an earlier LangChain and Chroma build that refused to start without a Hugging Face token and pinned libraries that no longer match current LangChain.

## Run

```bash
pip install -r requirements.txt
python check_bot.py
python chatbot.py
```

Open http://127.0.0.1:5000/ui

`POST /chat` with `{"query": "python"}` returns `response`, `sources`, and `mode` (`retrieve` or `model`). Five questions a minute per address.

The app binds to localhost only.
