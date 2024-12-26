import grpc
from concurrent import futures
import chatcompletion_pb2
import chatcompletion_pb2_grpc
from pydantic import BaseModel
from typing import List, Literal
from transformers import AutoModelForCausalLM, AutoTokenizer

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatInput(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.8
    max_tokens: int = 1024

class ChatOutput(BaseModel):
    choices: List[dict] = [{"message": {"role": "assistant", "content": ""}}]

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

class ChatCompletionService(chatcompletion_pb2_grpc.ChatCompletionServiceServicer):
    def ChatCompletion(self, request, context):
        chat_input = ChatInput(messages=[ChatMessage(role="user", content=request.name)])
        
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
        
        return chatcompletion_pb2.ChatCompletionResponse(message=response)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    chatcompletion_pb2_grpc.add_ChatCompletionServiceServicer_to_server(ChatCompletionService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Starting gRPC server on port 50051")
    print("grpc_cli call localhost:50051 ChatCompletion '你好，请介绍一下你自己'")
    server.wait_for_termination()

if __name__ == '__main__':
    serve() 