#!/usr/bin/env python3

import argparse
import random
import sys

import keras
import numpy as np

import storygenerator.io


def sample(preds, temperature=1.0):
	# helper function to sample an index from a probability array
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)


def generate_language(model):
	# Function invoked at end of each epoch. Prints generated text.
	print()
	# print('----- Generating text after Epoch: %d' % epoch)
#
# start_index = random.randint(0, len(text) - maxlen - 1)
# for diversity in [0.2, 0.5, 1.0, 1.2]:
#     print('----- diversity:', diversity)
#
#     generated = ''
#     sentence = text[start_index: start_index + maxlen]
#     generated += sentence
#     print('----- Generating with seed: "' + sentence + '"')
#     sys.stdout.write(generated)
#
#     for i in range(400):
#         x_pred = np.zeros((1, maxlen, len(chars)))
#         for t, char in enumerate(sentence):
#             x_pred[0, t, char_indices[char]] = 1.
#
#         preds = model.predict(x_pred, verbose=0)[0]
#         next_index = sample(preds, diversity)
#         next_char = indices_char[next_index]
#
#         generated += next_char
#         sentence = sentence[1:] + next_char
#
#         sys.stdout.write(next_char)
#         sys.stdout.flush()

def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Generates text using a given model.")
	result.add_argument("model_file", metavar="MODEL_FILE",
						help="A path to the model file to load.")
	result.add_argument("vocab_file", metavar="VOCAB_FILE",
						help="A path to the vocabulary file to load.")
	result.add_argument("-s", "--random-seed", dest="random_seed", metavar="SEED", type=int, default=7,
						help="The random seed to use.")
	return result


def __main(args):
	random_seed = args.random_seed
	print("Setting random seed to {}.".format(random_seed))
	# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
	# fix random seed for reproducibility
	random.seed(random_seed)
	np.random.seed(random_seed)

	model_file = args.model_file
	print("Will read model from \"{}\".".format(model_file))

	vocab_file = args.vocab_file
	vocab = storygenerator.io.read_vocab(vocab_file)
	print("Read vocabulary of size {}.".format(len(vocab)))

	# https://stackoverflow.com/a/43472000/1391325
	with keras.backend.get_session():
		print('Loading model.')
		model = keras.models.load_model(model_file)
		print(model.summary(), file=sys.stderr)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
