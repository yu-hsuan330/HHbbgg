import onnxruntime as ort
import tensorflow as tf
import tf2onnx
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization, Masking, Input, Normalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Load or build your Keras model
# model = tf.keras.models.load_model("/home/cosine/HHbbgg/MLForge/results/Pairing_vbf_v3_10jets/DNN/DNN_modelDNN.keras")  # or create one
model = Sequential()
model.add(Input(shape=(18,), name="input"))
# model.add(Masking(mask_value=-999))  # Masking layer to ignore -1 values
model.add(Dense(100, activation='relu'))
model.add(Dense(3, activation="softmax", name="output"))

print(model.summary())
print(model.outputs)
# Convert to ONNX
spec = (tf.TensorSpec((18,), tf.float32, name=[]),)  # adjust input shape
model_proto, _ = tf2onnx.convert.from_keras(
    model,
    # input_signature=spec,
    opset=17,
    output_path="HHbbgg_VBFpair_model.onnx",
    # output_names="output"  # Define manually
)
# model_proto, _ = tf2onnx.convert.from_keras(model, opset=17)

# Save ONNX model
# with open("HHbbgg_VBFpair_model.onnx", "wb") as f:
#     f.write(model_proto.SerializeToString())

sess = ort.InferenceSession("HHbbgg_VBFpair_model.onnx")
input_name = sess.get_inputs()[0].name
print(input_name)
# import onnx

# # Load your model
# model = onnx.load("/home/cosine/GitLab/merge_request/HiggsDNA/higgs_dna/tools/mjj_model_2022.onnx")

# # Print the opset version(s)
# for opset in model.opset_import:
#     print(f"Opset domain: {opset.domain or 'ai.onnx'}, version: {opset.version}")
