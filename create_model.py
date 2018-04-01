#!/usr/bin/env python3

"""
Creates a language model for generating prose.
"""

__author__ = "Todd Shore <errantlinguist+github@gmail.com>"
__copyright__ = "Copyright (C) 2018 Todd Shore"
__license__ = "Apache License, Version 2.0"

import argparse
import os
import random
import re
import sys
from typing import Callable, Dict, Iterable, Iterator, Sequence, Tuple

import magic
import numpy as np
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.models import Model, Sequential, load_model
from keras.optimizers import RMSprop

import keras.preprocessing.sequence
from keras.layers.wrappers import TimeDistributed
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from nltk.tokenize import sent_tokenize

from storygenerator import Chapter
from storygenerator.io import TextChapterReader

INPUT_FILE_MIMETYPE = "text/plain"
INPUT_FILENAME_PATTERN = re.compile("(\d+)\s+([^\.]+)\..+")


class Book(object):
	def __init__(self, ordinality: int, title: str, chaps: Sequence[Chapter]):
		self.ordinality = ordinality
		self.title = title
		self.chaps = chaps


class EpochEndHook(object):
	def __init__(self, model: Sequential, text: str, chars: Sequence[str], char_indices: Dict[str, int],
				 indices_char: Dict[int, str], maxlen: int):
		self.model = model
		self.text = text
		self.chars = chars
		self.char_indices = char_indices
		self.indices_char = indices_char
		self.maxlen = maxlen

	def __call__(self, epoch, logs):
		# Function invoked at end of each epoch. Prints generated text.
		print()
		print('----- Generating text after Epoch: %d' % epoch)

		start_index = random.randint(0, len(self.text) - self.maxlen - 1)
		for diversity in [0.2, 0.5, 1.0, 1.2]:
			print('----- diversity:', diversity)

			generated = ''
			sentence = self.text[start_index: start_index + self.maxlen]
			generated += sentence
			print('----- Generating with seed: "' + sentence + '"')
			sys.stdout.write(generated)

			for i in range(400):
				x_pred = np.zeros((1, self.maxlen, len(self.chars)))
				for t, char in enumerate(sentence):
					x_pred[0, t, self.char_indices[char]] = 1.

				preds = self.model.predict(x_pred, verbose=0)[0]
				next_index = sample(preds, diversity)
				next_char = self.indices_char[next_index]

				generated += next_char
				sentence = sentence[1:] + next_char

				sys.stdout.write(next_char)
				sys.stdout.flush()
			print()


class MimetypeFileWalker(object):

	def __init__(self, mimetype_matcher: Callable[[str], bool]):
		self.mimetype_matcher = mimetype_matcher
		self.__mime = magic.Magic(mime=True)

	def __call__(self, inpaths: Iterable[str]) -> Iterator[str]:
		for inpath in inpaths:
			if os.path.isdir(inpath):
				for root, _, files in os.walk(inpath, followlinks=True):
					for file in files:
						filepath = os.path.join(root, file)
						mimetype = self.__mime.from_file(filepath)
						if self.mimetype_matcher(mimetype):
							yield filepath
			else:
				mimetype = self.__mime.from_file(inpath)
				if self.mimetype_matcher(mimetype):
					yield inpath


def parse_book_filename(inpath: str) -> Tuple[int, str]:
	filename = os.path.basename(inpath)
	m = INPUT_FILENAME_PATTERN.match(filename)
	if m:
		ordinality = int(m.group(1))
		title = m.group(2)
	else:
		raise ValueError("Could not parse filename for path \"{}\".".format(inpath))
	return ordinality, title


def read_books(infiles: Iterable[str]) -> Iterator[Book]:
	reader = TextChapterReader()

	for infile in infiles:
		print("Reading \"{}\".".format(infile), file=sys.stderr)
		book_ordinality, book_title = parse_book_filename(infile)
		chaps = reader(infile)
		print("Read {} chapter(s) for book {}, titled \"{}\".".format(len(chaps), book_ordinality, book_title),
			  file=sys.stderr)
		yield Book(book_ordinality, book_title, chaps)


def sample(preds, temperature=1.0):
	# helper function to sample an index from a probability array
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)


def __create_argparser() -> argparse.ArgumentParser:
	result = argparse.ArgumentParser(
		description="Creates a language model for generating prose.")
	result.add_argument("inpaths", metavar="FILE", nargs='+',
						help="The text file(s) and/or directory path(s) to process.")
	result.add_argument("-e", "--encoding", metavar="CODEC", default="utf-8",
						help="The input file encoding.")
	result.add_argument("-s", "--random-seed", dest="random_seed", metavar="SEED", type=int, default=7,
						help="The random seed to use.")
	return result


def __main(args):
	random_seed = args.random_seed
	print("Setting random seed to {}.".format(random_seed), file=sys.stderr)
	# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
	# fix random seed for reproducibility
	random.seed(random_seed)
	np.random.seed(random_seed)
	file_walker = MimetypeFileWalker(lambda mimetype: mimetype == INPUT_FILE_MIMETYPE)
	inpaths = args.inpaths
	print("Looking for input text files underneath {}.".format(inpaths), file=sys.stderr)
	infiles = file_walker(inpaths)
	books = tuple(read_books(infiles))
	print("Read {} book(s).".format(len(books)), file=sys.stderr)

	# Concatenate all chapters for testing
	#all_sents = tuple(sent for book in books for chap in book.chaps for par in chap.pars for sent in sent_tokenize(par))
	chap_reprs = []
	for book in books:
		for chap in book.chaps:
			chap_repr = "\n\n".join(chap.pars)
			chap_reprs.append(chap_repr)
	raw_text = "\n\n\n====\n\n\n".join(chap_reprs)
	#print(raw_text)
	n_chars = len(raw_text)
	chars = sorted(list(frozenset(raw_text)))
	#chars = tuple(sorted(frozenset(char for sent in all_sents for char in sent)))
	char_indices = dict((c, i) for i, c in enumerate(chars))
	indices_char = dict((i, c) for i, c in enumerate(chars))
	n_vocab = len(chars)
	print("Total Characters: ", n_chars)
	print("Total Vocab: ", n_vocab)

	# cut the text in semi-redundant sequences of maxlen characters
	maxlen = 40
	step = 3
	sentences = []
	next_chars = []
	for i in range(0, len(raw_text) - maxlen, step):
		sentences.append(raw_text[i: i + maxlen])
		next_chars.append(raw_text[i + maxlen])
	print('nb sequences:', len(sentences))

	print('Vectorization...')
	x = np.zeros((len(sentences), maxlen, n_vocab), dtype=np.bool)
	y = np.zeros((len(sentences), n_vocab), dtype=np.bool)
	for i, sentence in enumerate(sentences):
		for t, char in enumerate(sentence):
			x[i, t, char_indices[char]] = 1
	y[i, char_indices[next_chars[i]]] = 1


	# https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
	# integer encode
	#label_encoder = LabelEncoder()

	# build the model: a single LSTM
	print('Build model...')
	model = Sequential()
	model.add(LSTM(128, input_shape=(maxlen, len(chars))))
	model.add(Dense(len(chars)))
	model.add(Activation('softmax'))

	optimizer = RMSprop(lr=0.01)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer)
	epoch_end_hook = EpochEndHook(model, raw_text, chars, char_indices, indices_char, maxlen)
	#print_callback = LambdaCallback(on_epoch_end=epoch_end_hook)
	# define the checkpoint
	filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]

	model.fit(x, y,
			  batch_size=128,
			  epochs=60,
			  callbacks=callbacks_list)


if __name__ == "__main__":
	__main(__create_argparser().parse_args())
