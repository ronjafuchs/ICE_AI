import service_pb2
import service_pb2_grpc
import message_pb2
import grpc
from typing import List


class GrpcController:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.channel = None
        self.stub = None
        self.KEY_MAP = {
            "A": "A",
            "B": "B",
            "C": "C",
            "U": "U",
            "R": "R",
            "D": "D",
            "L": "L",
        }

    def connect(self):
        self.channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        self.stub = service_pb2_grpc.ServiceStub(self.channel)

    def initialize(self, player_name: str, player_number: bool, is_blind: bool):
        request = service_pb2.InitializeRequest()
        request.player_number = player_number
        request.player_name = player_name
        request.is_blind = is_blind

        response = self.stub.Initialize(request)
        return response.player_uuid

    def participate(self, player_uuid: str):
        request = service_pb2.ParticipateRequest()
        request.player_uuid = player_uuid

        response = self.stub.Participate(request)
        self.frameStream = response
        return response

    def input(self, player_uuid: str, keys: List[str] = []):
        if player_uuid is None:
            raise ValueError("player_uuid cannot be None")
        request = service_pb2.PlayerInput()
        request.player_uuid = player_uuid
        grpc_key = message_pb2.GrpcKey(
            A=False, B=True, C=False, U=False, R=False, D=False, L=False
        )

        # Check if the key string is valid and set the corresponding field to True
        for key in keys:
            if key in ["A", "B", "C", "U", "R", "D", "L"]:
                setattr(request.input_key, key, True)
            else:
                raise ValueError(f"Invalid key: {key}")
        print("Input for player_uuid: ", player_uuid, " is: ", request.input_key)
        self.stub.Input(
            service_pb2.PlayerInput(player_uuid=player_uuid, input_key=grpc_key)
        )
        return

    def spectate(self):
        request = service_pb2.SpectateRequest()

        response = self.stub.Spectate(request)
        return response

    def runGame(self):
        request = service_pb2.RunGameRequest()
        request.character_1 = "ZEN"
        request.character_2 = "ZEN"
        request.player_1 = "Player 1"
        request.player_2 = "Player 2"
        request.game_number = 1

        self.stub.RunGame(request)
        return

    def get_next_frame(self):
        for frame in self.frameStream:
            yield frame
