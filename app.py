from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st
st.set_page_config(page_title="NaszGPT — Sokrates AI", page_icon="🏛️", layout="wide")

from dotenv import dotenv_values, load_dotenv, set_key
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI as StandardOpenAI,
    RateLimitError,
)

APP_DIR = Path(__file__).resolve().parent
ENV_PATH = APP_DIR / ".env"
DB_PATH = APP_DIR / "db"
DB_CONVERSATIONS_PATH = DB_PATH / "conversations"
CURRENT_CONVERSATION_PATH = DB_PATH / "current.json"

# Load settings from the .env file located beside app.py.
# Keep non-empty process environment values, but fill missing/empty values from .env.
load_dotenv(ENV_PATH, override=False)
_DOTENV_VALUES = dotenv_values(ENV_PATH)
for _name in (
    "OPENAI_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "LANGFUSE_BASE_URL",
):
    _current = os.environ.get(_name, "").strip()
    _from_file = _DOTENV_VALUES.get(_name)
    if not _current and isinstance(_from_file, str) and _from_file.strip():
        os.environ[_name] = _from_file.strip()

# Use the same Langfuse integration as in the user's original working example:
# `from langfuse.decorators import observe` and `from langfuse.openai import OpenAI`.
# Langfuse 2.x reads LANGFUSE_HOST; also accept LANGFUSE_BASE_URL for compatibility.
LANGFUSE_HOST = (
    os.environ.get("LANGFUSE_HOST", "").strip()
    or os.environ.get("LANGFUSE_BASE_URL", "").strip()
    or "https://cloud.langfuse.com"
)
os.environ["LANGFUSE_HOST"] = LANGFUSE_HOST
LANGFUSE_BASE_URL = LANGFUSE_HOST  # kept as the display variable used by the sidebar

LANGFUSE_AVAILABLE = False
LANGFUSE_IMPORT_ERROR: str | None = None
LangfuseSDKClient = None
LangfuseOpenAI = StandardOpenAI
_langfuse_observe = None
langfuse_context = None
_langfuse_diagnostic_client = None
try:
    from langfuse import Langfuse as LangfuseSDKClient
    from langfuse.decorators import observe as _langfuse_observe, langfuse_context
    from langfuse.openai import OpenAI as LangfuseOpenAI

    LANGFUSE_AVAILABLE = True
except Exception as exc: 
    LANGFUSE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

LANGFUSE_PUBLIC_KEY_PRESENT = bool(os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip())
LANGFUSE_SECRET_KEY_PRESENT = bool(os.environ.get("LANGFUSE_SECRET_KEY", "").strip())
LANGFUSE_CONFIGURED = LANGFUSE_PUBLIC_KEY_PRESENT and LANGFUSE_SECRET_KEY_PRESENT
LANGFUSE_ACTIVE = LANGFUSE_AVAILABLE and LANGFUSE_CONFIGURED

if LANGFUSE_ACTIVE:
    OpenAIClient = LangfuseOpenAI
    observe = _langfuse_observe
else:
    OpenAIClient = StandardOpenAI

    def observe(*decorator_args: Any, **decorator_kwargs: Any):
        def decorate(function):
            return function

        if decorator_args and callable(decorator_args[0]) and len(decorator_args) == 1:
            return decorator_args[0]
        return decorate


def get_langfuse_client() -> Any:
    """Create the Langfuse v2 SDK client only for the sidebar auth check."""
    global _langfuse_diagnostic_client
    if not LANGFUSE_ACTIVE or LangfuseSDKClient is None:
        raise RuntimeError(LANGFUSE_IMPORT_ERROR or "Langfuse is not configured")
    if _langfuse_diagnostic_client is None:
        _langfuse_diagnostic_client = LangfuseSDKClient(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"].strip(),
            secret_key=os.environ["LANGFUSE_SECRET_KEY"].strip(),
            host=LANGFUSE_HOST,
        )
    return _langfuse_diagnostic_client


def flush_langfuse() -> None:
    """Flush buffered decorator/integration events using the Langfuse v2 context."""
    if not LANGFUSE_ACTIVE or langfuse_context is None:
        return
    try:
        langfuse_context.flush()
    except Exception as exc:  # telemetry failures must not interrupt the chatbot
        print(f"Langfuse flush failed: {type(exc).__name__}: {exc}")


MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {
        "input_per_million": 0.15,
        "cached_input_per_million": 0.075,
        "output_per_million": 0.60,
    },
    "gpt-4o": {
        "input_per_million": 2.50,
        "cached_input_per_million": 1.25,
        "output_per_million": 10.00,
    },
    "gpt-4.1-mini": {
        "input_per_million": 0.40,
        "cached_input_per_million": 0.10,
        "output_per_million": 1.60,
    },
    "gpt-4.1": {
        "input_per_million": 2.00,
        "cached_input_per_million": 0.50,
        "output_per_million": 8.00,
    },
    "gpt-5.4-mini": {
        "input_per_million": 0.75,
        "cached_input_per_million": 0.075,
        "output_per_million": 4.50,
    },
    "gpt-5.4": {
        "input_per_million": 2.50,
        "cached_input_per_million": 0.25,
        "output_per_million": 15.00,
    },
    "gpt-5.6-sol": {
        "input_per_million": 4.00,
        "cached_input_per_million": 0.40,
        "output_per_million": 20.00,
    },
    "gpt-6-luna": {
        "input_per_million": 0.10,
        "cached_input_per_million": 0.01,
        "output_per_million": 0.50,
    },
    "gpt-6.1-sol": {
        "input_per_million": 2.00,
        "cached_input_per_million": 0.10,
        "output_per_million": 10.00,
    },
    "gpt-6-astra": {
        "input_per_million": 10.00,
        "cached_input_per_million": 1.00,
        "output_per_million": 50.00,
    },
}

MODEL_LABELS = {
    "gpt-4o-mini": "GPT-4o mini — ekonomiczny, starsza generacja (wycofanie planowane: 14.01.2027)",
    "gpt-4o": "GPT-4o — starsza generacja (wycofanie planowane: 14.01.2027)",
    "gpt-4.1-mini": "GPT-4.1 mini — szybki, starsza generacja (wycofanie planowane: 14.01.2027)",
    "gpt-4.1": "GPT-4.1 — starsza generacja (wycofanie planowane: 14.01.2027)",
    "gpt-5.4-mini": "GPT-5.4 mini — ekonomiczny",
    "gpt-5.4": "GPT-5.4 — zaawansowany",
    "gpt-5.6-sol": "GPT-5.6 Sol — profesjonalny",
    "gpt-6-luna": "GPT-6 Luna — ekonomiczny",
    "gpt-6.1-sol": "GPT-6.1 Sol — balans możliwości i kosztu",
    "gpt-6-astra": "GPT-6 Astra — najbardziej zaawansowany, droższy",
}
DEFAULT_MODEL = "gpt-6-luna"
MAX_CONTEXT_MESSAGES = 10
USD_TO_PLN_FALLBACK = 3.97

DEFAULT_PERSONALITY = """
Jesteś Sokratesem — cierpliwym przewodnikiem, który pomaga użytkownikowi
samodzielnie dojść do odpowiedzi. Nie podawaj od razu gotowego rozwiązania:
zadawaj jedno trafne pytanie naraz i dawaj wskazówki prowadzące do wniosku.
Za każdym razem, gdy użytkownik napisze „nie wiem”, podaj mu dodatkową wskazówkę.
Po trzecim „nie wiem” wyjaśnij rozwiązanie jasno i zwięźle.
""".strip()


def get_openai_api_key() -> str:
    """Read the API key from the process environment, .env, or this session."""
    file_values = dotenv_values(ENV_PATH)
    candidates = (
        os.environ.get("OPENAI_API_KEY"),
        file_values.get("OPENAI_API_KEY"),
        st.session_state.get("openai_api_key"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def get_openai_client() -> Any:
    """Create a client, asking for a key in Streamlit when none is configured."""
    api_key = get_openai_api_key()
    if not api_key:
        st.title(":classical_building: NaszGPT — Sokrates AI")
        st.warning("Nie znaleziono klucza API OpenAI. Podaj klucz, aby uruchomić chatbota.")

        with st.form("openai_api_key_form"):
            entered_key = st.text_input(
                "Klucz API OpenAI",
                type="password",
                help="Klucz jest ukryty na ekranie. Nie udostępniaj go publicznie.",
            ).strip()
            save_key = st.checkbox(
                "Zapisz klucz w pliku .env obok app.py",
                value=True,
            )
            submitted = st.form_submit_button("Zapisz klucz i uruchom")

        if submitted:
            if not entered_key:
                st.error("Wprowadź klucz API OpenAI.")
                st.stop()

            if save_key:
                try:
                    ENV_PATH.touch(exist_ok=True)
                    set_key(
                        str(ENV_PATH),
                        "OPENAI_API_KEY",
                        entered_key,
                        quote_mode="always",
                    )
                except (OSError, PermissionError) as exc:
                    st.error(f"Nie udało się zapisać pliku .env: {exc}")
                    st.stop()
            else:
                st.session_state["openai_api_key"] = entered_key

            st.rerun()

        st.stop()

    return OpenAIClient(api_key=api_key)


@st.cache_data(ttl=3600, show_spinner=False)
def get_usd_to_pln() -> tuple[float, bool]:
    """Return the current USD/PLN mid-rate from NBP; flag use of the fallback rate."""
    url = "https://api.nbp.pl/api/exchangerates/rates/A/USD/?format=json"
    try:
        response = requests.get(url, timeout=4)
        response.raise_for_status()
        rate = float(response.json()["rates"][0]["mid"])
        if rate <= 0:
            raise ValueError("Invalid USD/PLN rate returned by NBP")
        return rate, False
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return USD_TO_PLN_FALLBACK, True


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def new_conversation(conversation_id: int, personality: str | None = None) -> dict[str, Any]:
    return {
        "id": conversation_id,
        "name": f"Konwersacja {conversation_id}",
        "chatbot_personality": personality or DEFAULT_PERSONALITY,
        "messages": [],
    }


def load_conversation_to_state(conversation: dict[str, Any]) -> None:
    st.session_state["id"] = conversation["id"]
    st.session_state["name"] = conversation["name"]
    st.session_state["messages"] = conversation["messages"]
    st.session_state["chatbot_personality"] = conversation["chatbot_personality"]


def load_current_conversation() -> None:
    """Recover/create the conversation database and load the selected conversation."""
    DB_CONVERSATIONS_PATH.mkdir(parents=True, exist_ok=True)

    conversation_files = list(DB_CONVERSATIONS_PATH.glob("*.json"))
    known_ids: list[int] = []
    for path in conversation_files:
        try:
            known_ids.append(int(path.stem))
        except ValueError:
            continue

    current_id: int | None = None
    if CURRENT_CONVERSATION_PATH.is_file():
        try:
            current_data = json.loads(CURRENT_CONVERSATION_PATH.read_text(encoding="utf-8"))
            candidate = int(current_data["current_conversation_id"])
            if (DB_CONVERSATIONS_PATH / f"{candidate}.json").is_file():
                current_id = candidate
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            current_id = None

    if current_id is None and known_ids:
        current_id = max(known_ids)

    if current_id is None:
        current_id = 1
        conversation = new_conversation(current_id)
        write_json(DB_CONVERSATIONS_PATH / f"{current_id}.json", conversation)
    else:
        path = DB_CONVERSATIONS_PATH / f"{current_id}.json"
        try:
            conversation = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(conversation, dict):
                raise ValueError("Conversation data must be a JSON object")
            conversation.setdefault("id", current_id)
            conversation.setdefault("name", f"Konwersacja {current_id}")
            conversation.setdefault("chatbot_personality", DEFAULT_PERSONALITY)
            conversation.setdefault("messages", [])
            if not isinstance(conversation["messages"], list):
                conversation["messages"] = []
        except (OSError, json.JSONDecodeError, ValueError):
            # Do not overwrite the bad file; start a fresh conversation with a new ID.
            current_id = max(known_ids, default=0) + 1
            conversation = new_conversation(current_id)
            write_json(DB_CONVERSATIONS_PATH / f"{current_id}.json", conversation)

    write_json(CURRENT_CONVERSATION_PATH, {"current_conversation_id": current_id})
    load_conversation_to_state(conversation)


def get_current_conversation() -> dict[str, Any]:
    conversation_id = int(st.session_state["id"])
    path = DB_CONVERSATIONS_PATH / f"{conversation_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def save_current_conversation_messages() -> None:
    conversation = get_current_conversation()
    conversation["messages"] = st.session_state["messages"]
    write_json(DB_CONVERSATIONS_PATH / f"{conversation['id']}.json", conversation)


def save_current_conversation_name(widget_key: str) -> None:
    conversation = get_current_conversation()
    conversation["name"] = st.session_state.get(widget_key, "").strip() or f"Konwersacja {conversation['id']}"
    write_json(DB_CONVERSATIONS_PATH / f"{conversation['id']}.json", conversation)
    st.session_state["name"] = conversation["name"]


def save_current_conversation_personality(widget_key: str) -> None:
    conversation = get_current_conversation()
    conversation["chatbot_personality"] = (
        st.session_state.get(widget_key, "").strip() or DEFAULT_PERSONALITY
    )
    write_json(DB_CONVERSATIONS_PATH / f"{conversation['id']}.json", conversation)
    st.session_state["chatbot_personality"] = conversation["chatbot_personality"]


def create_new_conversation() -> None:
    ids: list[int] = []
    for path in DB_CONVERSATIONS_PATH.glob("*.json"):
        try:
            ids.append(int(path.stem))
        except ValueError:
            continue

    conversation_id = max(ids, default=0) + 1
    personality = st.session_state.get("chatbot_personality") or DEFAULT_PERSONALITY
    conversation = new_conversation(conversation_id, personality)
    write_json(DB_CONVERSATIONS_PATH / f"{conversation_id}.json", conversation)
    write_json(CURRENT_CONVERSATION_PATH, {"current_conversation_id": conversation_id})
    load_conversation_to_state(conversation)
    st.rerun()


def switch_conversation(conversation_id: int) -> None:
    path = DB_CONVERSATIONS_PATH / f"{int(conversation_id)}.json"
    conversation = json.loads(path.read_text(encoding="utf-8"))
    write_json(CURRENT_CONVERSATION_PATH, {"current_conversation_id": int(conversation_id)})
    load_conversation_to_state(conversation)
    st.rerun()


def list_conversations() -> list[dict[str, Any]]:
    conversations: list[dict[str, Any]] = []
    for path in DB_CONVERSATIONS_PATH.glob("*.json"):
        try:
            conversation = json.loads(path.read_text(encoding="utf-8"))
            conversations.append({"id": int(conversation["id"]), "name": str(conversation["name"])})
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return conversations


def estimate_message_cost(message: dict[str, Any]) -> float | None:
    """Estimate one assistant response using the model saved with that response."""
    usage = message.get("usage")
    model_id = message.get("model")
    if not isinstance(usage, dict) or model_id not in MODEL_PRICING:
        return None

    try:
        input_tokens = max(0, int(usage.get("prompt_tokens", 0)))
        output_tokens = max(0, int(usage.get("completion_tokens", 0)))
        cached_tokens = min(input_tokens, max(0, int(usage.get("cached_prompt_tokens", 0))))
    except (TypeError, ValueError):
        return None

    pricing = MODEL_PRICING[model_id]
    uncached_input_tokens = input_tokens - cached_tokens
    input_cost = (
        uncached_input_tokens * pricing["input_per_million"]
        + cached_tokens * pricing["cached_input_per_million"]
    ) / 1_000_000
    output_cost = output_tokens * pricing["output_per_million"] / 1_000_000
    return input_cost + output_cost


openai_client = get_openai_client()
USD_TO_PLN, USING_FALLBACK_EXCHANGE_RATE = get_usd_to_pln()


@observe()
def chatbot_reply(user_prompt: str, memory: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    """Call the selected chat model; memory must NOT include user_prompt yet."""
    messages = [{
        "role": "system",
        "content": st.session_state["chatbot_personality"],
    }]
    messages.extend(
        {"role": message["role"], "content": message["content"]}
        for message in memory
        if message.get("role") in {"user", "assistant"} and isinstance(message.get("content"), str)
    )
    messages.append({"role": "user", "content": user_prompt})

    response = openai_client.chat.completions.create(
        model=model_id,
        messages=messages,
    )
    usage: dict[str, int] = {}
    if response.usage:
        usage = {
            "completion_tokens": int(response.usage.completion_tokens or 0),
            "prompt_tokens": int(response.usage.prompt_tokens or 0),
            "total_tokens": int(response.usage.total_tokens or 0),
        }
        prompt_details = getattr(response.usage, "prompt_tokens_details", None)
        cached_tokens = getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0
        usage["cached_prompt_tokens"] = int(cached_tokens or 0)

    return {
        "role": "assistant",
        "content": response.choices[0].message.content or "",
        "usage": usage,
        "model": model_id,
    }


def format_api_error(exc: Exception) -> str:
    if isinstance(exc, AuthenticationError):
        return "OpenAI odrzuciło klucz API. Sprawdź, czy klucz jest poprawny i aktywny."
    if isinstance(exc, RateLimitError):
        return "Przekroczono limit API lub dostępny budżet. Sprawdź limity i rozliczenia konta OpenAI."
    if isinstance(exc, APIConnectionError):
        return "Nie można połączyć się z API OpenAI. Sprawdź połączenie internetowe."
    if isinstance(exc, APIStatusError):
        return f"API OpenAI zwróciło błąd HTTP {exc.status_code}. Spróbuj ponownie później."
    return "Nie udało się uzyskać odpowiedzi. Sprawdź konfigurację API i spróbuj ponownie."


# MAIN PROGRAM
load_current_conversation()
st.title(":classical_building: NaszGPT — Sokrates AI")
st.caption("Chatbot inspirowany metodą sokratejską. Pytania prowadzą do samodzielnego odkrywania odpowiedzi.")

for message in st.session_state["messages"]:
    if message.get("role") in {"user", "assistant"}:
        with st.chat_message(message["role"]):
            st.markdown(message.get("content", ""))

prompt = st.chat_input("O co chcesz spytać?")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            previous_messages = st.session_state["messages"][:-1][-MAX_CONTEXT_MESSAGES:]
            selected_model_id = st.session_state.get("model", DEFAULT_MODEL)
            response = chatbot_reply(prompt, memory=previous_messages, model_id=selected_model_id)
            flush_langfuse()
            st.markdown(response["content"])
            st.session_state["messages"].append(response)
        except Exception as exc:
            st.error(format_api_error(exc))
        finally:
            save_current_conversation_messages()

with st.sidebar:
    st.subheader("Ustawienia modelu")
    model_ids = list(MODEL_LABELS.keys())
    if "model" not in st.session_state or st.session_state["model"] not in model_ids:
        st.session_state["model"] = DEFAULT_MODEL
    current_model_label = MODEL_LABELS[st.session_state["model"]]
    selected_model_label = st.selectbox(
        "Wybrany model",
        options=[MODEL_LABELS[model_id] for model_id in model_ids],
        index=[MODEL_LABELS[model_id] for model_id in model_ids].index(current_model_label),
        key="selected_model_label",
    )
    st.session_state["model"] = next(
        model_id for model_id, label in MODEL_LABELS.items() if label == selected_model_label
    )
    st.caption(f"Model API: `{st.session_state['model']}`")

    st.subheader("Aktualna konwersacja")
    total_cost = 0.0
    unpriced_legacy_messages = 0
    for message in st.session_state.get("messages", []):
        if message.get("role") != "assistant" or "usage" not in message:
            continue
        message_cost = estimate_message_cost(message)
        if message_cost is None:
            unpriced_legacy_messages += 1
        else:
            total_cost += message_cost

    col_usd, col_pln = st.columns(2)
    with col_usd:
        st.metric("Koszt rozmowy (USD)", f"${total_cost:.6f}")
    with col_pln:
        st.metric("Koszt rozmowy (PLN)", f"{total_cost * USD_TO_PLN:.4f}")

    if USING_FALLBACK_EXCHANGE_RATE:
        st.caption(f"Brak aktualnego kursu NBP; PLN wyliczono orientacyjnie przy kursie {USD_TO_PLN:.2f}.")
    else:
        st.caption(f"Kurs referencyjny NBP: 1 USD = {USD_TO_PLN:.4f} PLN (odświeżany co godzinę).")
    if unpriced_legacy_messages:
        st.warning(
            f"Nie uwzględniono kosztu {unpriced_legacy_messages} starszych odpowiedzi, "
            "ponieważ historia nie zapisywała modelu użytego do ich wygenerowania."
        )

    name_widget_key = f"conversation_name_{st.session_state['id']}"
    personality_widget_key = f"conversation_personality_{st.session_state['id']}"
    st.session_state[name_widget_key] = st.session_state["name"]
    st.session_state[personality_widget_key] = st.session_state["chatbot_personality"]
    st.text_input(
        "Nazwa konwersacji",
        key=name_widget_key,
        on_change=save_current_conversation_name,
        args=(name_widget_key,),
    )
    st.text_area(
        "Osobowość chatbota",
        max_chars=2000,
        height=200,
        key=personality_widget_key,
        on_change=save_current_conversation_personality,
        args=(personality_widget_key,),
    )

    if st.button("Nowa konwersacja", use_container_width=True):
        create_new_conversation()

    st.subheader("Konwersacje")
    conversations = sorted(list_conversations(), key=lambda item: item["id"], reverse=True)
    for conversation in conversations[:5]:
        columns = st.columns([3, 1])
        with columns[0]:
            st.write(conversation["name"])
        with columns[1]:
            if st.button(
                "Załaduj",
                key=f"load_conversation_{conversation['id']}",
                disabled=conversation["id"] == st.session_state["id"],
            ):
                switch_conversation(conversation["id"])

    st.divider()
    st.subheader("Diagnostyka Langfuse")
    if not LANGFUSE_AVAILABLE:
        st.error("SDK Langfuse nie zostało załadowane.")
        st.caption(f"Szczegóły: {LANGFUSE_IMPORT_ERROR or 'nieznany błąd importu'}")
        st.caption("Zainstaluj zależności poleceniem: python -m pip install -U langfuse openai")
    elif not LANGFUSE_CONFIGURED:
        missing = []
        if not LANGFUSE_PUBLIC_KEY_PRESENT:
            missing.append("LANGFUSE_PUBLIC_KEY")
        if not LANGFUSE_SECRET_KEY_PRESENT:
            missing.append("LANGFUSE_SECRET_KEY")
        st.warning("Langfuse jest wyłączony. Brak zmiennych: " + ", ".join(missing))
        st.caption(f"Sprawdzony plik .env: `{ENV_PATH}`")
    else:
        st.success("Klucze Langfuse wykryte; aktywna jest integracja `langfuse.openai.OpenAI`.")
        st.caption(f"Endpoint: `{LANGFUSE_BASE_URL}`")
        st.caption("Po wysłaniu wiadomości aplikacja próbuje opróżnić kolejkę zdarzeń Langfuse.")
        if st.button("Sprawdź połączenie z Langfuse", use_container_width=True):
            try:
                client = get_langfuse_client()
                authenticated = client.auth_check()
                if authenticated:
                    st.success("Langfuse potwierdził połączenie i autoryzację.")
                else:
                    st.error("Langfuse nie potwierdził autoryzacji. Sprawdź klucze oraz endpoint.")
            except Exception as exc:
                st.error(f"Test Langfuse nie powiódł się: {type(exc).__name__}: {exc}")
                st.caption("Sprawdź, czy klucze pochodzą z tego samego projektu i czy LANGFUSE_BASE_URL wskazuje właściwy region.")
