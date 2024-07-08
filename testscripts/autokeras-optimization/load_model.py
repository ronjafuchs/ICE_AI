import autokeras as ak
from keras.models import load_model

model = load_model("model_autokeras", custom_objects=ak.CUSTOM_OBJECTS)
model.summary()
print(model.summary())