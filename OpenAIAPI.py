from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from openai.types.responses import Response

SPACER = "--------------------------"
client = OpenAI()

# USD per million tokens, standard tier. Estimates only, from https://developers.openai.com/api/docs/pricing (checked 2026-10-25)
MODEL_PRICING = {
    "gpt-6-astra": {"input": 10.00, "output": 50.00, "cache_read": 1.00},
    "gpt-6.1-sol": {"input": 2.00,  "output": 10.00, "cache_read": 0.10},
    "gpt-6-sol":   {"input": 2.00,  "output": 10.00, "cache_read": 0.20},
    "gpt-6-luna":  {"input": 0.10,  "output": 0.50,  "cache_read": 0.01},
}

# Every GPT-6 model charges more once the prompt goes over 272K tokens
LONG_CONTEXT_THRESHOLD = 272_000
LONG_CONTEXT_PRICING = {
    "gpt-6-astra": {"input": 20.00, "output": 75.00, "cache_read": 2.00},
    "gpt-6.1-sol": {"input": 4.00,  "output": 15.00, "cache_read": 0.20},
    "gpt-6-sol":   {"input": 4.00,  "output": 15.00, "cache_read": 0.40},
    "gpt-6-luna":  {"input": 0.20,  "output": 0.75,  "cache_read": 0.02},
}

# The 30 minute cache rate. 12 hour writes cost 2x, but the usage doesn't say which TTL was used, so this assumes 30 minutes
CACHE_WRITE_MULTIPLIER = 1.25

def list_models():
    for model in client.models.list():
        print(model.id)

# The Responses API takes the same {"role", "content"} items as the Anthropic messages list
def add_user_message(messages, text):
    messages.append({"role": "user", "content": text})

def add_assistant_message(messages, text):
    messages.append({"role": "assistant", "content": text})

def add_response_output(messages, response):
    # Copies every output item (reasoning, function calls, messages) so a tool-use loop keeps them in the history
    messages.extend(item.model_dump(exclude_none=True) for item in response.output)

def build_tool_result(call_id, output):
    return {"type": "function_call_output", "call_id": call_id, "output": output}

def text_from_response(response):
    return response.output_text

def stop_reason(response):
    # OpenAI has no stop_reason: "completed", or why the response is incomplete (e.g. max_output_tokens)
    if response.status == "incomplete" and response.incomplete_details:
        return response.incomplete_details.reason
    return response.status

def _token_counts(usage):
    input_details = usage.input_tokens_details
    output_details = usage.output_tokens_details
    cache_read_tokens = (input_details.cached_tokens or 0) if input_details else 0
    cache_write_tokens = (getattr(input_details, "cache_write_tokens", 0) or 0) if input_details else 0
    return {
        # OpenAI's input_tokens includes cached tokens; report uncached input to match the Anthropic numbers
        "input_tokens": max((usage.input_tokens or 0) - cache_read_tokens - cache_write_tokens, 0),
        # Includes reasoning tokens, which are billed as output
        "output_tokens": usage.output_tokens or 0,
        "cache_read_input_tokens": cache_read_tokens,
        "cache_creation_input_tokens": cache_write_tokens,
        "reasoning_tokens": (output_details.reasoning_tokens or 0) if output_details else 0,
    }

def estimate_cost(model, usage):
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        print(f"WARNING: no pricing for model '{model}', cost not estimated")
        return None

    counts = _token_counts(usage)
    # OpenAI's input_tokens is the whole prompt, cached tokens and cache writes included
    long_context = LONG_CONTEXT_PRICING.get(model)
    if long_context and (usage.input_tokens or 0) > LONG_CONTEXT_THRESHOLD:
        pricing = long_context

    per_token = 1_000_000
    input_cost = counts["input_tokens"] * pricing["input"] / per_token
    output_cost = counts["output_tokens"] * pricing["output"] / per_token
    cache_read_cost = counts["cache_read_input_tokens"] * pricing["cache_read"] / per_token
    cache_write_cost = counts["cache_creation_input_tokens"] * pricing["input"] * CACHE_WRITE_MULTIPLIER / per_token

    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "cache_write_cost": cache_write_cost,
        "cache_read_cost": cache_read_cost,
        "total_cost": input_cost + output_cost + cache_write_cost + cache_read_cost,
    }

def usage_summary(model, usage):
    # Same keys as AnthropicAPI.usage_summary, plus reasoning_tokens, so both share the log format
    summary = _token_counts(usage)
    summary["total_tokens"] = (
        summary["input_tokens"] + summary["output_tokens"]
        + summary["cache_read_input_tokens"] + summary["cache_creation_input_tokens"]
    )
    summary["cost"] = estimate_cost(model, usage)
    return summary

def chat(messages, model="gpt-6-luna", system=None, temperature=1.0, stream=True, output_config=None, tools=None, return_full_response=False, return_usage=False, max_tokens=4000):
    # temperature is accepted for parity with AnthropicAPI.chat but not sent
    params = {
        "model": model,
        "max_output_tokens": max_tokens,
        "input": messages,
    }

    if system:
        params["instructions"] = system

    if output_config:
        # Accepts the Anthropic-style {"format": {"type": "json_schema", "schema": ...}}; OpenAI also wants a name
        params["text"] = {"format": {"name": "output", "strict": True, **output_config["format"]}}

    if tools:
        params["tools"] = tools

    if stream:
        with client.responses.stream(**params) as response_stream:
            # One pass over the events, as in AnthropicAPI.chat
            for event in response_stream:
                if event.type == "response.output_text.delta":
                    print(event.delta, end="", flush=True)
                elif event.type == "response.function_call_arguments.delta":
                    print(event.delta, end="", flush=True)
            print()
            final = response_stream.get_final_response()
    else:
        final = client.responses.create(**params)

    result = final if return_full_response else text_from_response(final)

    if return_usage:
        return result, usage_summary(model, final.usage)
    return result

def tester():
    messages = []
    add_user_message(messages, "What is quantum computing? Answer in one sentence")

    answer = chat(messages)
    print(SPACER)

    add_assistant_message(messages, answer)
    add_user_message(messages, "Write another sentence")

    chat(messages)

def tester_with_system():
    messages = []
    add_user_message(messages, "How do I solve 5x + 2 = 3 for x?")

    system = """
    You are a patient math tutor.
    Do not directly answer a student's questions.
    Guide them to a solution step by step.
    """
    chat(messages, system=system)

def structured_data_tester():
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

    chat(messages, output_config=output_config)

#list_models()
#tester()
#tester_with_system()
#structured_data_tester()