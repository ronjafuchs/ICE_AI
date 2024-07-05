from worker import Worker
from multiprocessing import Pipe
from keras.models import load_model
import autokeras as ak
from sklearn.preprocessing import LabelBinarizer
import pandas as pd


class AgentPWorker(Worker):
    def __init__(self, pipe: Pipe):
        super().__init__()
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
        self.label_binarizer = LabelBinarizer().fit(self.allowed_values)
        self.desired_columns = pd.read_csv("desired_columns.csv")
        self.pipe = pipe

    def run(self):
        predictor_model = load_model(
            "model_autokeras", custom_objects=ak.CUSTOM_OBJECTS
        )
        while True:
            assert 1 == self.pipe.poll()
            item = self.pipe.recv()
            message, data = item[0], item[1]
            print(message, data)
            assert message == "PREDICT"
            if message == "PREDICT":
                prediction = predictor_model.predict(data)
                prediction = self.label_binarizer.inverse_transform(prediction)
                self.pipe.send(["OUTPUT_PREDICTION", prediction])