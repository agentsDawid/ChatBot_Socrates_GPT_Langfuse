# NaszGPT — Sokrates AI

A Streamlit chatbot inspired by the Socratic method. It guides users through questions and hints, supports conversation history and editable personality instructions, allows model selection, estimates token costs in USD/PLN, and can send traces to Langfuse.

## Requirements

- Python 3.10 or newer
- An OpenAI API key
- Optional: Langfuse project keys to enable tracing

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

If `OPENAI_API_KEY` is not set, the app asks for it in the Streamlit interface. You may save it to `.env` next to `app.py` or use it only for the current session. To configure keys manually, copy `.env.example` to `.env` and fill in the values.

## Optional Langfuse tracing

The current code intentionally uses the Langfuse 2.x integration pattern:

```python
from langfuse.decorators import observe
from langfuse.openai import OpenAI
```

Set both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`. Set `LANGFUSE_HOST` to the endpoint for your Langfuse region. The code also accepts `LANGFUSE_BASE_URL` as an alias. Without both Langfuse keys, the chatbot falls back to the regular OpenAI client.

**Compatibility note:** `requirements.txt` pins Langfuse below version 3 to match the current imports.
## Local data and secrets

- `.env` contains secrets and must not be committed.
- `db/` stores conversation history and is excluded by `.gitignore`.
- Displayed token costs are estimates based on the price table in `app.py`; actual invoices may differ.
- USD→PLN uses the NBP exchange-rate API, with a labelled fallback if the API cannot be reached.

## Screenshot

![NaszGPT chat interface](files/chatbot_socrates.png)
