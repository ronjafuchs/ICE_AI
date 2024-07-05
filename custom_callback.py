from stable_baselines3.common.callbacks import BaseCallback
from multiprocessing import Pipe
import numpy as np
from PIL import Image

class PredictionCallback(BaseCallback):
    def __init__(self, pipe: Pipe, verbose=0,):
        super(PredictionCallback, self).__init__(verbose)
        self.pipe = pipe

    def _on_step(self) -> bool:
        # Access the agent's policy and make predictions
        message, image = None, None
        if self.pipe.poll():
            message, image = self.pipe.recv()
        else:
            return True
        if message == "OBSERVATION":
            # save observation as image in folder images with increasing file name
            #Image.fromarray(image.astype(np.uint8), 'L').save("images/{}.png".format(self.num_timesteps))
            pass
        else:
            assert False
        image = image / 255
        image = image.reshape((1, 1, 64, 96))
        action, _ = self.model.predict(image)
        self.training_env
        self.pipe.send(["ACTION", action[0]])
        return True