"""A two-turn conversation and a system prompt, with streaming, through each provider module directly.

Makes real (billed) API calls. Run with:  python examples/conversation.py [anthropic|openai|openrouter]
"""

import sys

from model_evaluator import PROVIDERS

provider = PROVIDERS[sys.argv[1] if len(sys.argv) > 1 else "anthropic"]

# Two turns: the second question only makes sense with the first answer in the history
messages = []
provider.add_user_message(messages, "What is quantum computing? Answer in one sentence")
answer = provider.chat(messages)  # streams the reply to the terminal and returns its text
provider.add_assistant_message(messages, answer)
provider.add_user_message(messages, "Write another sentence")
provider.chat(messages)

print("-" * 26)

# A system prompt changes how the model answers
messages = []
provider.add_user_message(messages, "How do I solve 5x + 2 = 3 for x?")
provider.chat(
    messages,
    system="You are a patient math tutor. Do not directly answer a student's questions. "
    "Guide them to a solution step by step.",
)
