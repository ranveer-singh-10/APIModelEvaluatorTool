from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def list_models():
    for m in client.models.list():
        print(m)

def test_call():
    r = client.responses.create(model="gpt-6-luna", input="Say hello")
    print(r.output_text)