from tensorflow.keras import utils
import utils as utils


# maxlen_data_json = utils.load_max_len_data(model_folder)


enc_model, dec_model, tokenizer, maxlen = utils.load_latest_models()
maxlen_questions = maxlen['maxlen_questions']
maxlen_answers = maxlen['maxlen_answers']

utils.converse(
    enc_model=enc_model,
    dec_model=dec_model,
    maxlen_answers=maxlen_answers,
    maxlen_questions=maxlen_questions,
    tokenizer=tokenizer
)
