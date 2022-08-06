import grpc
from ..mygrpc import chat_pb2, chat_pb2_grpc
import grpc
from concurrent import futures
# from src.mygrpc import chat_pb2_grpc


class Server:
    def __init__(self, chatter):
        self.chatter = chatter


server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
chat_pb2_grpc.add_ChatServicer_to_server(Server, server)
server.add_insecure_port('[::]3000')
server.start()
server.wait_for_termination()
