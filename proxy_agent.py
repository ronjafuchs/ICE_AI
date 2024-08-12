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


class ProxyAgent(AIInterface):
    def __init__(self):
        super().__init__()
        self.blind_flag = False
        self.frame_data: FrameData = None

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
            self.key.empty()
            self.cc.skill_cancel()

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

