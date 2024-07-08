from tkinter import font
import visualkeras
import keras
import tensorflow as tf
from keras.models import load_model
import autokeras as ak
from PIL import ImageFont

# load the keras model
model = load_model("../../bigger_mctsai_model_autokeras", custom_objects=ak.CUSTOM_OBJECTS)

# Custom color map
color_map = {
    "InputLayer": "skyblue",
    "Dense": "green",
    "Normalization": "teal",
    "ReLU": "orange",
    "Softmax": "red",
    "MultiCategoryEncoding": "purple",  # add other layers as needed
}

# Custom spacing
spacing = 10  # space between layers

# Generate the visualization with custom settings
visualkeras.layered_view(
    model,
    legend=True,
    color_map=color_map,  # apply the color map
    spacing=spacing,  # apply spacing
    to_file="model_visualization.png",  # save the visualization to a file
).show()  # display using your system viewer
