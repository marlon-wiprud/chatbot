from tensorflow.keras import utils
import utils as utils

tokenizer = utils.load_tokenizer()

model_folder = "og_model/"
maxlen_data_json = utils.load_max_len_data(model_folder)


maxlen_questions = maxlen_data_json['maxlen_questions']
maxlen_answers = maxlen_data_json['maxlen_answers']

enc_model, dec_model = utils.load_latest_models()

utils.converse(
    convo_length=10,
    enc_model=enc_model,
    dec_model=dec_model,
    maxlen_answers=maxlen_answers,
    maxlen_questions=maxlen_questions,
    tokenizer=tokenizer
)
