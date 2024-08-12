from multiprocessing import Pipe

import numpy as np
from pyftg.ai_interface import AIInterface
from pyftg.struct import (
    AudioData,
    CommandCenter,
    FrameData,
    GameData,
    Key,
    RoundResult,
    ScreenData,
)


class AgentRL(AIInterface):
    """ The reinforcement Learning agent.
    As an AI interface it directly receives information from the current game
    and sends it to the river agent and the RL agent. It uses a single pipe for this.
    It receives the action as a response from the RL agent and applies it
    """
    def __init__(self, pipe: Pipe):
        super().__init__()
        self.pipe = pipe
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

            # input time
            # send image over queue to predictor
            # wait for output
            # send output to command center

            # get screen data
            image_bytes = self.screen_data.display_bytes
            image_bytes = np.frombuffer(image_bytes, dtype=np.uint8)

            img = image_bytes.reshape((64, 96))

            if self.frame_data is not None:
                img = self.include_hp_and_energy(img.copy())

            # send information to RL model
            self.pipe.send(["OBSERVATION", img])

            # send information to River model
            self.pipe.send(["FRAME_DATA", self.frame_data.as_dict()])

            message, data = self.pipe.recv()
            assert message == "ACTION"
            action = data
            self.cc.command_call(self.convert_action(action))

    def round_end(self, round_result: RoundResult):
        #print(
        #    round_result.remaining_hps[0],
        #    round_result.remaining_hps[1],
        #    round_result.elapsed_frame,
        #)
        pass

    def game_end(self):
        self.pipe.send(["TERMINATED", None])
        print("game end")

    def convert_action(self, output):
        allowed_values = [
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
        return allowed_values[output]

    def include_hp_and_energy(self, image: np.ndarray):
        """ adds information on the players hp and energy levels in the image as separate pixels
        player 1 = top left
        player 2 = top right

        Parameters
        ----------
        image the image to be modified

        Returns
        -------
        the modified image
        """
        hp_character_1 = self.frame_data.character_data[0].hp
        hp_character_2 = self.frame_data.character_data[1].hp
        energy_character_1 = self.frame_data.character_data[0].energy
        energy_character_2 = self.frame_data.character_data[1].energy
        image[0, 0] = hp_character_1
        hp_character_1 -= 255
        if hp_character_1 > 0:
            image[0, 1] = hp_character_1

        image[0, 95] = hp_character_2
        hp_character_2 -= 255
        if hp_character_2 > 0:
            image[0, 94] = hp_character_2

        image[1, 0] = energy_character_1
        energy_character_1 -= 255
        if energy_character_1 > 0:
            image[1, 1] = energy_character_1

        image[1, 95] = energy_character_2
        energy_character_2 -= 255
        if energy_character_2 > 0:
            image[1, 94] = energy_character_2
        return image
