from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic
from anthropic.types import Message
import os

SPACER ="--------------------------"
client = Anthropic()

# USD per million tokens. Estimates only - check against https://claude.com/pricing
MODEL_PRICING = {
    "claude-fable-5-1":  {"input": 10.00, "output": 50.00, "cache_read": 0.25},
    "claude-opus-5-5":   {"input": 4.00,  "output": 20.00, "cache_read": 0.20},
    "claude-opus-5":     {"input": 5.00,  "output": 25.00, "cache_read": 0.50},
    "claude-sonnet-5-5": {"input": 2.00,  "output": 10.00, "cache_read": 0.20},
    "claude-sonnet-5":   {"input": 2.00,  "output": 10.00, "cache_read": 0.20},
    "claude-sonnet-4-6": {"input": 3.00,  "output": 15.00, "cache_read": 0.30},
    "claude-haiku-5-5":  {"input": 0.10,  "output": 0.50,  "cache_read": 0.01},
    "claude-haiku-4-5":  {"input": 1.00,  "output": 5.00,  "cache_read": 0.10},
}

# Haiku 5.5 charges more once the prompt goes over 100K tokens
LONG_CONTEXT_PRICING = {
    "claude-haiku-5-5": {"threshold": 100_000, "input": 0.50, "output": 2.50, "cache_read": 0.05},
}

CACHE_WRITE_5M_MULTIPLIER = 1.25
CACHE_WRITE_1H_MULTIPLIER = 2.0

def list_models():
    client = Anthropic(
        api_key=os.environ.get(
            "ANTHROPIC_API_KEY"
        ),  # This is the default and can be omitted
    )
    page = client.models.list()
    #page = page.data[0]

    for model in page.data:
        print(model.id)

def make_request(role,content,model="claude-haiku-5-5"):
    #client = Anthropic()

    message = client.messages.create(
        model=model,
        max_tokens=4000,
        messages=[
            {
                "role": role,
                "content": content
            }
        ]
    )

    return message

#list_models()
#print(message.content[0].text)

def add_user_message(messages, text):
    user_message = {"role": "user", "content": text}
    messages.append(user_message)

def add_user_message_multi(messages, message):
    user_message = {
        "role": "user",
        "content": message.content if isinstance(message, Message) else message
    }
    messages.append(user_message)

def add_assistant_message(messages, text):
    assistant_message = {"role": "assistant", "content": text}
    messages.append(assistant_message)

def add_assistant_message_multi(messages, message):
    assistant_message = {
        "role": "assistant",
        "content": message.content if isinstance(message, Message) else message
    }
    messages.append(assistant_message)

def build_tooluse_json(messages, text, is_error=False):
    json=[{
        "type": "tool_result",
        "tool_use_id": messages.content[0].id,
        "content": text,
        "is_error": is_error
    }]
    return json


def chat_old(messages,model="claude-haiku-5-5"):
    #client = Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=4000,
        messages=messages,
    )
    return message.content[0].text

def text_from_message(message):
    return "\n".join(
        [block.text for block in message.content if block.type == "text"]
    )

# text_from_response and stop_reason match OpenAIAPI's, so evaluation.py can treat both providers alike
def text_from_response(response):
    return text_from_message(response)

def stop_reason(response):
    return response.stop_reason

def estimate_cost(model, usage):
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        print(f"WARNING: no pricing for model '{model}', cost not estimated")
        return None

    input_tokens = usage.input_tokens or 0
    output_tokens = usage.output_tokens or 0
    cache_read_tokens = usage.cache_read_input_tokens or 0
    cache_write_tokens = usage.cache_creation_input_tokens or 0

    long_context = LONG_CONTEXT_PRICING.get(model)
    prompt_tokens = input_tokens + cache_read_tokens + cache_write_tokens
    if long_context and prompt_tokens > long_context["threshold"]:
        pricing = long_context

    # Split cache writes by TTL when the API reports it, otherwise assume the 5 minute TTL
    cache_creation = getattr(usage, "cache_creation", None)
    if cache_creation is not None:
        write_5m = cache_creation.ephemeral_5m_input_tokens or 0
        write_1h = cache_creation.ephemeral_1h_input_tokens or 0
    else:
        write_5m, write_1h = cache_write_tokens, 0

    per_token = 1_000_000
    input_cost = input_tokens * pricing["input"] / per_token
    output_cost = output_tokens * pricing["output"] / per_token
    cache_read_cost = cache_read_tokens * pricing["cache_read"] / per_token
    cache_write_cost = (
        write_5m * pricing["input"] * CACHE_WRITE_5M_MULTIPLIER
        + write_1h * pricing["input"] * CACHE_WRITE_1H_MULTIPLIER
    ) / per_token

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "cache_write_cost": cache_write_cost,
        "cache_read_cost": cache_read_cost,
        "total_cost": input_cost + output_cost + cache_write_cost + cache_read_cost,
    }

def usage_summary(model, usage):
    summary = {
        "input_tokens": usage.input_tokens or 0,
        "output_tokens": usage.output_tokens or 0,
        "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
        "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
    }
    summary["total_tokens"] = sum(summary.values())
    summary["cost"] = estimate_cost(model, usage)
    return summary

def chat(messages,model="claude-haiku-5-5", system=None,temperature=1.0, stream=True, output_config=None, tools=None, return_full_response=False, fine_grained=False, return_usage=False):
    #client = Anthropic()
    # temperature is accepted but no longer sent: API version 2026-10-21 rejects sampling parameters with a 400
    params = {
        "model": model,
        "max_tokens": 4000,
        "messages": messages
    }

    if output_config:
        params["output_config"] = output_config

    if system:
        params["system"] = system

    if tools:
        if fine_grained:
            tools = [
                {**tool, "eager_input_streaming": True} if "input_schema" in tool else tool
                for tool in tools
            ]
        params["tools"] = tools

    if stream:
        with client.messages.stream(**params) as message:
            # One pass over the events: a second loop (e.g. text_stream) would find the stream already used up
            for event in message:
                if event.type == "text":
                    print(event.text, end="", flush=True)
                elif event.type == "input_json":
                    print(event.partial_json, end="", flush=True)
            print()
            final = message.get_final_message()
    else:
        final = client.messages.create(**params)

    if return_full_response:
        result = final
    else:
        result = "".join(block.text for block in final.content if block.type == "text")

    if return_usage:
        return result, usage_summary(model, final.usage)
    return result

def show_web_search_response(response):
    print(f"stop_reason: {response.stop_reason}")
    print(f"Searches used: {response.usage.server_tool_use.web_search_requests}\n")

    for block in response.content:
        if block.type == "server_tool_use":
            print(f"SERVER TOOL: {block.name}")
            print(f"  input: {block.input}")
            query = block.input.get("query") if isinstance(block.input, dict) else None
            if query:
                print(f"SEARCH QUERY: {query}")
        elif block.type == "web_search_tool_result":
            if isinstance(block.content, list):
                for result in block.content:
                    print(f"  RESULT: {result.title}\n          {result.url}")
            else:
                print(f"  SEARCH ERROR: {block.content.error_code}")

        elif block.type == "text":
            print(f"\nTEXT: {block.text}")
            for c in (block.citations or []):
                print(f"   CITED: {c.title} ({c.url})")
                print(f"          \"{c.cited_text[:100]}...\"")

        print()

def tester():
    #list_models()
    #role= "user"

    content= "What is quantum computing? Answer in one sentence"

    # Start with an empty message list
    messages = []

    # Add the initial user question
    add_user_message(messages, content)

    # Get Claude's response
    answer = chat(messages)

    print(answer)
    print(SPACER)

    # Add Claude's response to the conversation history
    add_assistant_message(messages, answer)

    # Add a follow-up question
    add_user_message(messages, "Write another sentence")

    # Get the follow-up response with full context
    final_answer = chat(messages)

    #result= make_request(role, content)
    #print(result.content[0].text)
    print(final_answer)


def tester_with_system():

    content = "How do I solve 5x + 2 = 3 for x?"

    # Start with an empty message list
    messages = []

    # Add the initial user question
    add_user_message(messages, content)

    # Without system prompt
    answer = chat(messages)

    print(answer)
    print(SPACER)

    # With system prompt
    system = """
    You are a patient math tutor.
    Do not directly answer a student's questions.
    Guide them to a solution step by step.
    """
    answer = chat(messages, system=system)
    print(answer)
    print(SPACER)

def tester_with_temp():
    model = "claude-sonnet-4-6"

    prompt = "Suggest a name for a new pizzeria. Reply with the name only, nothing else"

    # Start with an empty message list
    messages = []

    # Add the initial user question
    add_user_message(messages, prompt)

    # Without system prompt
    answer = chat(messages,model=model, temperature=0)

    print(answer)
    print(SPACER)

    temperature=0.0

    answer = chat(messages, model=model, temperature=0.9)
    print(answer)
    print(SPACER)

def tester_with_stream():
    model = "claude-sonnet-4-6"

    prompt = "What is the weather forecast for Miami over the next 3 days?"

    # Start with an empty message list
    messages = []

    # Add the initial user question
    add_user_message(messages, prompt)

    # Without system prompt
    answer = chat(messages,model=model, temperature=0, stream=True)
    print(answer)
    print(SPACER)

def structured_data_tester():
    #model = "claude-sonnet-4-6"
    messages = []

    add_user_message(messages, "Give me 3 AWS services and what they do")

    output_config = {
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "services": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"}
                            },
                            "required": ["name", "description"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["services"],
                "additionalProperties": False
            }
        }
    }

    text = chat(messages, output_config=output_config)
    print(text)

#tester_with_system()
#tester_with_temp()
#tester_with_stream()
#structured_data_tester()
