"""Structured JSON output. Every provider takes the same Anthropic-style ``output_config``.

Makes real (billed) API calls. Run with:  python examples/structured_output.py [anthropic|openai|openrouter]
"""

import json
import sys

from model_evaluator import PROVIDERS

provider = PROVIDERS[sys.argv[1] if len(sys.argv) > 1 else "anthropic"]

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
                        "properties": {"name": {"type": "string"}, "description": {"type": "string"}},
                        "required": ["name", "description"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["services"],
            "additionalProperties": False,
        },
    }
}

messages = []
provider.add_user_message(messages, "Give me 3 AWS services and what they do")
text = provider.chat(messages, output_config=output_config, stream=False)
for service in json.loads(text)["services"]:
    print(f"{service['name']}: {service['description']}")
