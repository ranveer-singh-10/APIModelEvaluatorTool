from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic
from anthropic.types import Message
import os

SPACER ="--------------------------"
client = Anthropic()

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

def chat(messages,model="claude-haiku-5-5", system=None,temperature=1.0, stream=True, output_config=None, tools=None, return_full_response=False, fine_grained=False):
    #client = Anthropic()
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

    extra = {"temperature": temperature} if temperature is not None else {}

    if stream:
        with client.messages.stream(**params, extra_body=extra) as message:
            for event in message:
                if event.type == "input_json":
                    print(event.partial_json, end="", flush=True)
            for text in message.text_stream:
                print(text, end="")
            final = message.get_final_message()
        if return_full_response:
            return final
        else:
            return "".join(block.text for block in final.content if block.type == "text")
    else:
        message = client.messages.create(**params, extra_body={"temperature": temperature})
        if return_full_response:
            return message
        else:
            return "".join(block.text for block in message.content if block.type == "text")

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