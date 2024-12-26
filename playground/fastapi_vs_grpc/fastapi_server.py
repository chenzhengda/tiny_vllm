from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Literal

app = FastAPI()

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatInput(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.8
    max_tokens: int = 1024

class ChatOutput(BaseModel):
    choices: List[dict] = [{"message": {"role": "assistant", "content": ""}}]

from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


@app.post("/v1/chat/completions", response_model=ChatOutput)
def chat_completion(request: Request, chat_input: ChatInput):
    text = tokenizer.apply_chat_template(
        chat_input.messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return ChatOutput(choices=[{
        "message": {
            "role": "assistant",
            "content": response
        }
    }])
    

if __name__ == "__main__":
    import uvicorn

    print("Starting FastAPI server...")
    print("curl -X POST 'http://localhost:8000/v1/chat/completions' -H 'Content-Type: application/json' -d '{\"messages\": [{\"role\": \"user\", \"content\": \"你好，请介绍一下你自己\"}], \"temperature\": 0.8, \"max_tokens\": 1024}'")

    # run uvicorn with reload
    uvicorn.run("fastapi_server:app", port=8000, reload=True)
