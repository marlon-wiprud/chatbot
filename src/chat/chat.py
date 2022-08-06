import utils as utils
import numpy as np


class Chat:

    def __init__(self):
        self.enc_model, self.dec_model, self.tokenizer, maxlen = utils.load_latest_models()
        self.maxlen_questions = maxlen['maxlen_questions']
        self.maxlen_answers = maxlen['maxlen_answers']

    def converse(self, input):

        states_values = self.enc_model.predict(
            utils.str_to_tokens(input, self.tokenizer, self.maxlen_questions))

        empty_target_seq = np.zeros((1, 1))

        empty_target_seq[0, 0] = self.tokenizer.word_index['start']

        stop_condition = False
        decoded_translation = ''

        while not stop_condition:
            x = [empty_target_seq] + states_values
            dec_outputs, h, c = self.dec_model.predict(x)
            sampled_word_index = np.argmax(dec_outputs[0, -1, :])
            sampled_word = None
            for word, index in self.tokenizer.word_index.items():
                if sampled_word_index == index:
                    decoded_translation += ' {}'.format(word)
                    sampled_word = word

            if sampled_word == 'end' or len(decoded_translation.split()) > self.maxlen_answers:
                stop_condition = True

            empty_target_seq = np.zeros((1, 1))
            empty_target_seq[0, 0] = sampled_word_index
            states_values = [h, c]

        return decoded_translation
