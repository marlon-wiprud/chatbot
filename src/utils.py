from datetime import datetime
from tensorflow.keras import models, preprocessing, layers, activations
import numpy as np
import re
import yaml
import os
import pickle
import json
from constants import dec_model_path, enc_model_path, model_path, tokenizer_path, max_len_data_path
from pathlib import Path


def encode_data(tokenizer, data):
    tokenized_data = tokenizer.texts_to_sequences(data)
    maxlen = max([len(x) for x in tokenized_data])
    padded_data = preprocessing.sequence.pad_sequences(
        tokenized_data, maxlen=maxlen, padding='post')
    encoder_input_data = np.array(padded_data)
    return tokenized_data, encoder_input_data, maxlen


def tokenize(sentences):
    tokens_list = []
    vocabulary = []
    for sentence in sentences:
        sentence = sentence.lower()
        sentence = re.sub('[^a-zA-Z', ' ', sentence)
        tokens = sentence.split()
        vocabulary += tokens
        tokens_list.append(tokens)
    return tokens_list, vocabulary


def get_encoder_layers(VOCAB_SIZE, maxlen_questions):
    encoder_inputs = layers.Input(shape=(maxlen_questions,))
    encoder_embedding = layers.Embedding(
        VOCAB_SIZE, 200, mask_zero=True)(encoder_inputs)
    encoder_outputs, state_h, state_c = layers.LSTM(
        200, return_state=True)(encoder_embedding)
    encoder_states = [state_h, state_c]
    return encoder_inputs, encoder_embedding, encoder_outputs, encoder_states, state_h, state_c


def get_decoder_layers(VOCAB_SIZE, maxlen_answers, encoder_states):
    decoder_inputs = layers.Input(shape=(maxlen_answers, ))
    decoder_embedding = layers.Embedding(
        VOCAB_SIZE, 200, mask_zero=True)(decoder_inputs)
    decoder_lstm = layers.LSTM(200, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_lstm(
        decoder_embedding, initial_state=encoder_states)
    decoder_dense = layers.Dense(VOCAB_SIZE, activation=activations.softmax)
    output = decoder_dense(decoder_outputs)
    return decoder_inputs, decoder_embedding, decoder_lstm, decoder_outputs, decoder_dense, output


def make_inference_models(encoder_inputs, decoder_lstm, decoder_embedding, encoder_states, decoder_dense, decoder_inputs):
    encoder_model = models.Model(encoder_inputs, encoder_states)

    decoder_state_input_h = layers.Input(shape=(200,))
    decoder_state_input_c = layers.Input(shape=(200,))

    decoder_states_inputs = [decoder_state_input_c, decoder_state_input_h]

    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding, initial_state=decoder_states_inputs)

    decoder_states = [state_h, state_c]

    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = models.Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model


def load_data(dir_path):

    files_list = os.listdir(dir_path + os.sep)

    questions = list()
    answers = list()

    for filepath in files_list:
        stream = open(dir_path + os.sep + filepath, 'rb')
        docs = yaml.safe_load(stream)
        conversations = docs['conversations']
        for con in conversations:
            if len(con) > 2:
                questions.append(con[0])
                replies = con[1:]
                ans = ''
                for rep in replies:
                    ans += ' ' + rep
                answers.append(ans)
            elif len(con) > 1:
                questions.append(con[0])
                answers.append(con[1])

    answers_with_tags = list()
    for i in range(len(answers)):
        if type(answers[i]) == str:
            answers_with_tags.append(answers[i])
        else:
            questions.pop(i)

    answers = list()
    for i in range(len(answers_with_tags)):
        answers.append('<START> ' + answers_with_tags[i] + ' <END>')

    tokenizer = preprocessing.text.Tokenizer()

    tokenizer.fit_on_texts(questions + answers)

    VOCAB_SIZE = len(tokenizer.word_index) + 1

    vocab = []

    for word in tokenizer.word_index:
        vocab.append(word)

    return tokenizer, VOCAB_SIZE, vocab, questions, answers


def str_to_tokens(sentence: str, tokenizer, maxlen_questions):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append(tokenizer.word_index[word])
    return preprocessing.sequence.pad_sequences([tokens_list], maxlen=maxlen_questions, padding='post')


def converse(convo_length, enc_model, dec_model, maxlen_answers, maxlen_questions, tokenizer):
    for _ in range(convo_length):
        question = input('Enter question : ')
        states_values = enc_model.predict(
            str_to_tokens(question, tokenizer, maxlen_questions))
        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = tokenizer.word_index['start']
        stop_condition = False
        decoded_translation = ''
        while not stop_condition:
            x = [empty_target_seq] + states_values
            dec_outputs, h, c = dec_model.predict(x)
            sampled_word_index = np.argmax(dec_outputs[0, -1, :])
            sampled_word = None
            for word, index in tokenizer.word_index.items():
                if sampled_word_index == index:
                    decoded_translation += ' {}'.format(word)
                    sampled_word = word

            if sampled_word == 'end' or len(decoded_translation.split()) > maxlen_answers:
                stop_condition = True

            empty_target_seq = np.zeros((1, 1))
            empty_target_seq[0, 0] = sampled_word_index
            states_values = [h, c]

        print('response: ', decoded_translation)


def save_tokenizer(tokenizer, path):
    with open(path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_tokenizer():
    with open("og_model/" + tokenizer_path, 'rb') as handle:
        return pickle.load(handle)


def save_json(data, path):
    with open(path, 'w') as outfile:
        json.dump(data, outfile)


def load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def newVersionFolder():
    folder_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dir = os.path.join(
        os.getcwd() + "/models",
        datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    os.makedirs(dir)
    return "models/" + folder_name


def save_models(model, enc_model, dec_model, tokenizer):
    folder_name = newVersionFolder()
    save_path = folder_name + "/"
    model.save(save_path + model_path)
    enc_model.save(save_path + enc_model_path)
    dec_model.save(save_path + dec_model_path)
    save_tokenizer(tokenizer, save_path + tokenizer_path)

# converter = tf.lite.TFLiteConverter.from_keras_model(enc_model)
# buffer = converter.convert()
# open('enc_model.tflite', 'wb').write(buffer)

# converter = tf.lite.TFLiteConverter.from_keras_model(dec_model)
# open( 'dec_model.tflite' , 'wb' ).write(buffer)


def load_enc_model(folder_name):
    return models.load_model(folder_name + enc_model_path)


def load_dec_model(folder_name):
    return models.load_model(folder_name + dec_model_path)


def load_max_len_data(folder_name):
    return load_json(folder_name + max_len_data_path)


def get_latest_model_folder():
    paths = sorted(Path("models").iterdir(), key=os.path.getmtime)
    print(paths)
    if paths[len(paths) - 1]:
        folder = paths[len(paths) - 1]
        print(folder.name)
        return "models/" + folder.name + "/"


def load_latest_models():
    folder = get_latest_model_folder()
    enc_model = load_enc_model(folder)
    dec_model = load_dec_model(folder)
    return enc_model, dec_model
