import numpy as np
from tensorflow.keras import optimizers,  models, preprocessing
from keras.utils.np_utils import to_categorical
import utils as my_utils
from constants import data_path, EPOCHS, BATCH_SIZE

questions, answers = my_utils.load_data_v2(data_path)

tokenizer, vocab, VOCAB_SIZE = my_utils.tokenize_data(questions, answers)

# print("answers: ", answers)

# tokenizer, VOCAB_SIZE, vocab, questions, answers = my_utils.load_data(data_path)


# encoder input data
tokenized_questions, encoder_input_data, maxlen_questions = my_utils.encode_data(
    tokenizer, questions)

# decoder input data
tokenized_answers, decoder_input_data, maxlen_answers = my_utils.encode_data(
    tokenizer, answers)

# decoder output data
for i in range(len(tokenized_answers)):
    tokenized_answers[i] = tokenized_answers[i][1:]
padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers, maxlen=maxlen_answers, padding='post')
one_hot_answers = to_categorical(padded_answers, VOCAB_SIZE)
decoder_output_data = np.array(one_hot_answers)


encoder_inputs, encoder_embedding, encoder_outputs, encoder_states, state_h, state_c = my_utils.get_encoder_layers(
    VOCAB_SIZE, maxlen_questions)


decoder_inputs, decoder_embedding, decoder_lstm, decoder_outputs, decoder_dense, output = my_utils.get_decoder_layers(
    VOCAB_SIZE, maxlen_answers, encoder_states)


model = models.Model([encoder_inputs, decoder_inputs], output)

model.compile(optimizer=optimizers.RMSprop(), loss='categorical_crossentropy')

model.summary()

model.fit([encoder_input_data, decoder_input_data],
          decoder_output_data, batch_size=BATCH_SIZE, epochs=EPOCHS)

enc_model, dec_model = my_utils.make_inference_models(
    encoder_inputs=encoder_inputs,
    decoder_lstm=decoder_lstm,
    decoder_embedding=decoder_embedding,
    encoder_states=encoder_states,
    decoder_dense=decoder_dense,
    decoder_inputs=decoder_inputs
)


my_utils.save_models(model, enc_model, dec_model, tokenizer,
                     maxlen_questions, maxlen_answers)
