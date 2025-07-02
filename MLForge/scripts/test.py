# import tensorflow as tf

# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(19,)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(3, activation='softmax')
# ])
# model.export("test")
# # model = tf.keras.Sequential()
# # model.add(tf.keras.layers.Input(shape=(19,), name="input"))
# # # model.add(Masking(mask_value=-999))  # Masking layer to ignore -1 values
# # model.add(tf.keras.layers.Dense(64, activation='relu'))
# # model.add(tf.keras.layers.Dense(3, activation="softmax", name="output"))
# print(model.summary())
# output_names = [tensor.name for tensor in model.outputs]
# print(output_names)


import onnxruntime as ort

sess = ort.InferenceSession("model.onnx")
input_name = sess.get_inputs()[0].name
print(f"Input name: {input_name}")

outputs = sess.run(None, {input_name: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]})
preds_array = outputs[0]
print(f"Predictions: {preds_array}")