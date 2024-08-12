from pyftg.ai_interface import AIInterface
from pyftg.struct import (
    Key,
    FrameData,
    GameData,
    RoundResult,
    ScreenData,
    AudioData,
    CommandCenter,
)
import pandas as pd
import util
from multiprocessing import Pipe


class AgentRiver(AIInterface):
    def __init__(self, pipe: Pipe):
        super().__init__()
        self.pipe = pipe
        self.blind_flag = False
        self.frame_data: FrameData = None
        self.allowed_values = [
            "STAND",
            "FORWARD_WALK",
            "DASH",
            "BACK_STEP",
            "CROUCH",
            "JUMP",
            "FOR_JUMP",
            "BACK_JUMP",
            "STAND_GUARD",
            "CROUCH_GUARD",
            "AIR_GUARD",
            "THROW_A",
            "THROW_B",
            "STAND_A",
            "STAND_B",
            "CROUCH_A",
            "CROUCH_B",
            "AIR_A",
            "AIR_B",
            "AIR_DA",
            "AIR_DB",
            "STAND_FA",
            "STAND_FB",
            "CROUCH_FA",
            "CROUCH_FB",
            "AIR_FA",
            "AIR_FB",
            "AIR_UA",
            "AIR_UB",
            "STAND_D_DF_FA",
            "STAND_D_DF_FB",
            "STAND_F_D_DFA",
            "STAND_F_D_DFB",
            "STAND_D_DB_BA",
            "STAND_D_DB_BB",
            "AIR_D_DF_FA",
            "AIR_D_DF_FB",
            "AIR_F_D_DFA",
            "AIR_F_D_DFB",
            "AIR_D_DB_BA",
            "AIR_D_DB_BB",
            "STAND_D_DF_FC",
        ]
        self.desired_columns = pd.read_csv("desired_columns.csv")
        """
        self.save_frame = pd.DataFrame(columns=self.desired_columns.columns)
        self.huge_list = []
        """

    def name(self) -> str:
        return self.__class__.__name__

    def is_blind(self) -> bool:
        return self.blind_flag

    def input(self) -> Key:
        return self.key

    def initialize(self, game_data: GameData, player_number: int):
        self.cc = CommandCenter()
        self.key = Key()

    def get_information(
        self, frame_data: FrameData, is_control: bool, non_delay_frame_data: FrameData
    ):
        self.frame_data = frame_data

    def get_screen_data(self, screen_data: ScreenData):
        self.screen_data = screen_data

    def get_audio_data(self, audio_data: AudioData):
        self.audio_data = audio_data

    def processing(self):
        if self.frame_data.empty_flag or self.frame_data.current_frame_number <= 0:
            return

        if self.cc.get_skill_flag():
            self.key = self.cc.get_skill_key()
        else:
            flattened_data = util.flatten_dict(self.frame_data.as_dict())
            # print(flattened_data)
            # self.huge_list.append(flattened_data)

            df = pd.DataFrame(
                flattened_data,
                index=[0],
                columns=self.desired_columns.columns,
                dtype="str",
            )
            # print(df)
            # print(df)
            # df.to_csv("bla.csv")
            self.key.empty()
            self.cc.skill_cancel()
            self.pipe.send(["PREDICT", flattened_data])
            response = self.pipe.recv()
            message, input = response[0], response[1]
            assert message == "OUTPUT_PREDICTION"
            self.cc.command_call(input[0])

    def round_end(self, round_result: RoundResult):
        print(
            round_result.remaining_hps[0],
            round_result.remaining_hps[1],
            round_result.elapsed_frame,
        )

    def game_end(self):
        print("game end")
        """
        self.save_frame = pd.DataFrame(self.huge_list)
        self.save_frame.to_csv("game_data.csv")
        """

