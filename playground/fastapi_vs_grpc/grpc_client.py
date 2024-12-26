import grpc
import chatcompletion_pb2
import chatcompletion_pb2_grpc

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = chatcompletion_pb2_grpc.ChatCompletionServiceStub(channel)
        response = stub.ChatCompletion(chatcompletion_pb2.ChatCompletionRequest(name="你好，请介绍一下你自己"))
        print("Client received: " + response.message)

if __name__ == '__main__':
    run()
