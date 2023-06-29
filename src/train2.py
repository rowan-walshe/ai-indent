import random
import numpy as np
import tensorflow as tf

# Set a fixed seed for reproducibility, for the random module, numpy, and tensorflow
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

MAX_TOKENS = 256
MAX_INDENTATION = 120

from typing import List, Union
import re

from models.transformers.transformer import Transformer

class Tokenizer:

    _WORD = re.compile(r'^\w*\b')

    _LIBRARY = [' ', '\n', '-- A comment', '.', 'abort', 'else', 'new', 'return', 'elsif', 'not', 'reverse', 'abstract', 'end', 'null', 'accept', 'entry', 'select', 'access', 'exception', 'of', 'separate', 'aliased', 'exit', 'some', 'all', 'others', 'subtype', 'and', 'for', 'out', 'synchronized', 'array', 'function', 'overriding', 'at', 'tagged', 'generic', 'package', 'task', 'begin', 'goto', 'pragma', 'terminate', 'body', 'private', 'then', 'if', 'procedure', 'type', 'case', 'in', 'protected', 'constant', 'interface', 'until', 'is', 'raise', 'use', 'declare', 'range', 'delay', 'limited', 'record', 'when', 'delta', 'loop', 'rem', 'while', 'digits', 'renames', 'with', 'do', 'mod', 'requeue', 'xor', 'abs', 'or', '=>', '(', ')', "'", '>=', '<=', '/=', '>', '<', ':=', '=', '+', '-', '*', '/', '**', '&', ',', ';', ':', '[', ']']
    _LIBRARY_REGEX = {' ': re.compile(r'^ '), '\n': re.compile(r'^(\r)?\n'), '-- A comment': re.compile(r'^--.*'), '.': re.compile(r'^\.'), 'abort': re.compile(r'^\babort\b'), 'else': re.compile(r'^\belse\b'), 'new': re.compile(r'^\bnew\b'), 'return': re.compile(r'^\breturn\b'), 'elsif': re.compile(r'^\belsif\b'), 'not': re.compile(r'^\bnot\b'), 'reverse': re.compile(r'^\breverse\b'), 'abstract': re.compile(r'^\babstract\b'), 'end': re.compile(r'^\bend\b'), 'null': re.compile(r'^\bnull\b'), 'accept': re.compile(r'^\baccept\b'), 'entry': re.compile(r'^\bentry\b'), 'select': re.compile(r'^\bselect\b'), 'access': re.compile(r'^\baccess\b'), 'exception': re.compile(r'^\bexception\b'), 'of': re.compile(r'^\bof\b'), 'separate': re.compile(r'^\bseparate\b'), 'aliased': re.compile(r'^\baliased\b'), 'exit': re.compile(r'^\bexit\b'), 'some': re.compile(r'^\bsome\b'), 'all': re.compile(r'^\ball\b'), 'others': re.compile(r'^\bothers\b'), 'subtype': re.compile(r'^\bsubtype\b'), 'and': re.compile(r'^\band\b'), 'for': re.compile(r'^\bfor\b'), 'out': re.compile(r'^\bout\b'), 'synchronized': re.compile(r'^\bsynchronized\b'), 'array': re.compile(r'^\barray\b'), 'function': re.compile(r'^\bfunction\b'), 'overriding': re.compile(r'^\boverriding\b'), 'at': re.compile(r'^\bat\b'), 'tagged': re.compile(r'^\btagged\b'), 'generic': re.compile(r'^\bgeneric\b'), 'package': re.compile(r'^\bpackage\b'), 'task': re.compile(r'^\btask\b'), 'begin': re.compile(r'^\bbegin\b'), 'goto': re.compile(r'^\bgoto\b'), 'pragma': re.compile(r'^\bpragma\b'), 'terminate': re.compile(r'^\bterminate\b'), 'body': re.compile(r'^\bbody\b'), 'private': re.compile(r'^\bprivate\b'), 'then': re.compile(r'^\bthen\b'), 'if': re.compile(r'^\bif\b'), 'procedure': re.compile(r'^\bprocedure\b'), 'type': re.compile(r'^\btype\b'), 'case': re.compile(r'^\bcase\b'), 'in': re.compile(r'^\bin\b'), 'protected': re.compile(r'^\bprotected\b'), 'constant': re.compile(r'^\bconstant\b'), 'interface': re.compile(r'^\binterface\b'), 'until': re.compile(r'^\buntil\b'), 'is': re.compile(r'^\bis\b'), 'raise': re.compile(r'^\braise\b'), 'use': re.compile(r'^\buse\b'), 'declare': re.compile(r'^\bdeclare\b'), 'range': re.compile(r'^\brange\b'), 'delay': re.compile(r'^\bdelay\b'), 'limited': re.compile(r'^\blimited\b'), 'record': re.compile(r'^\brecord\b'), 'when': re.compile(r'^\bwhen\b'), 'delta': re.compile(r'^\bdelta\b'), 'loop': re.compile(r'^\bloop\b'), 'rem': re.compile(r'^\brem\b'), 'while': re.compile(r'^\bwhile\b'), 'digits': re.compile(r'^\bdigits\b'), 'renames': re.compile(r'^\brenames\b'), 'with': re.compile(r'^\bwith\b'), 'do': re.compile(r'^\bdo\b'), 'mod': re.compile(r'^\bmod\b'), 'requeue': re.compile(r'^\brequeue\b'), 'xor': re.compile(r'^\bxor\b'), 'abs': re.compile(r'^\babs\b'), 'or': re.compile(r'^\bor\b'), '=>': re.compile(r'^=>'), '(': re.compile(r'^\('), ')': re.compile(r'^\)'), "'": re.compile(r"^'"), '>=': re.compile(r'^>='), '<=': re.compile(r'^<='), '/=': re.compile(r'^/='), '>': re.compile(r'^>'), '<': re.compile(r'^<'), ':=': re.compile(r'^:='), '=': re.compile(r'^='), '+': re.compile(r'^\+'), '-': re.compile(r'^-'), '*': re.compile(r'^\*'), '/': re.compile(r'^/'), '**': re.compile(r'^\*\*'), '&': re.compile(r'^&'), ',': re.compile(r'^,'), ';': re.compile(r'^;'), ':': re.compile(r'^:'), '[': re.compile(r'^\['), ']': re.compile(r'^\]'),}

    _STRING_LIT = 'STRING_LIT'

    _LIBRARY = [_STRING_LIT] + _LIBRARY
    _LIBRARY_REGEX[_STRING_LIT] = re.compile(r'^"(""|[^"\n])*"')

    _TOKEN_TO_ID = {k: v + 1 for v, k in enumerate(_LIBRARY)}
    _ID_TO_TOKEN = {v: k for k, v in _TOKEN_TO_ID.items()}

    _PAD = 0
    _UKN1 = len(_LIBRARY) + 1
    _ID_TO_TOKEN[_PAD] = ''
    _ID_TO_TOKEN[_UKN1] = '#'


    _NEWLINE_TOKEN = _TOKEN_TO_ID['\n']
    @classmethod
    def newline_token(cls) -> int:
        return cls._NEWLINE_TOKEN

    _SPACE_TOKEN = _TOKEN_TO_ID[' ']
    @classmethod
    def space_token(cls) -> int:
        return cls._SPACE_TOKEN

    @classmethod
    def _gen_uknown(cls, unknown_count: int) -> List[int]:
        return [cls._UKN1] * unknown_count

    @classmethod
    def encode(cls, text: str) -> List[int]:
        token_ids = []
        unknown_count = 0
        while text:
            for token in cls._LIBRARY:
                if match := cls._LIBRARY_REGEX[token].match(text):
                    token_ids.extend(cls._gen_uknown(unknown_count))
                    unknown_count = 0
                    if token == cls._STRING_LIT:
                        match_length = len(match.group())
                        token_ids.extend([cls._TOKEN_TO_ID[token]] * match_length)
                        text = text[match_length:]
                    else:
                        text = cls._LIBRARY_REGEX[token].sub('', text, count=1)
                        token_ids.append(cls._TOKEN_TO_ID[token])
                    break
            else:
                word_length = 1
                if match := cls._WORD.match(text):
                    word_length = len(match.group())
                unknown_count += word_length
                text = text[word_length:]
        token_ids.extend(cls._gen_uknown(unknown_count))
        return token_ids

    @classmethod
    def _decode_string_literals(cls, token_ids: List[int]) -> List[Union[int, str]]:
        result = []
        str_char_count = 0
        for token_id in token_ids:
            if token_id == cls._TOKEN_TO_ID[cls._STRING_LIT]:
                str_char_count += 1
            else:
                if str_char_count > 0:
                    result.append('"' + cls._ID_TO_TOKEN[cls._UKN1] * (str_char_count - 2) + '"')
                    str_char_count = 0
                result.append(token_id)
        else:
            if str_char_count > 0:
                    result.append('"' + cls._ID_TO_TOKEN[cls._UKN1] * (str_char_count - 2) + '"')
                    str_char_count = 0
        return result

    @classmethod
    def decode(cls, token_ids: List[int]) -> str:
        partial_decode = cls._decode_string_literals(token_ids)
        text_parts = [cls._ID_TO_TOKEN[x] if isinstance(x, int) else x for x in partial_decode]
        return ''.join(text_parts)

    @classmethod
    def resize(cls, token_ids: List[int], max_length: int) -> List[int]:
        # If the token_ids are longer than max_length, truncate the start
        # If the token_ids are shorter than max_length, pad the start with _PAD
        if len(token_ids) > max_length:
            return token_ids[-max_length:]
        else:
            return [cls._PAD] * (max_length - len(token_ids)) + token_ids

    @classmethod
    def n_vocab(cls) -> int:
        return len(cls._ID_TO_TOKEN) + 1

import os
import subprocess

DATA_DIR = "data_2"

git_repos = [
    "https://github.com/AdaCore/Ada_Drivers_Library.git",
    "https://github.com/AdaCore/gnatstudio.git",
    "https://github.com/AdaCore/spark2014.git",
    "https://github.com/AdaCore/ada_language_server.git",
    "https://github.com/AdaCore/gnat-llvm.git",
    "https://github.com/AdaCore/libadalang.git",
    "https://github.com/AdaCore/aws.git",
    "https://github.com/AdaCore/RecordFlux.git",
    "https://github.com/AdaCore/learn.git",
    "https://github.com/AdaCore/gtkada.git",
    "https://github.com/AdaCore/gprbuild.git",
    "https://github.com/AdaCore/bb-runtimes.git",
    "https://github.com/AdaCore/svd2ada.git",
    "https://github.com/AdaCore/VSS.git",
    "https://github.com/AdaCore/gnatcoll-core.git",
    "https://github.com/AdaCore/Certyflie.git",
    "https://github.com/AdaCore/gnatcoverage.git",
]

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

for repo in git_repos:
    subprocess.run(["git", "clone", "--depth", "1", repo], cwd=DATA_DIR)

import hashlib
import math

from typing import List


def file_hash(file_path: str):
    # Calculate the hash of a file
    with open(file_path, "rb") as f:
        file_hash = hashlib.sha256()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()

def is_file_mostly_space_indented(file_path: str):
    # Returns True if the file is mostly space indented
    # Returns False if the file is mostly tab indented
    # Defaults to False if the file is empty
    space_indent_count = 0
    tab_indent_count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        file_contents = f.readlines()
        for line in file_contents:
            whitespace_count = len(line) - len(line.lstrip())
            whitespaces = line[:whitespace_count]
            space_indent_count += whitespaces.count(" ")
            tab_indent_count += whitespaces.count("\t")

    # In ada, the convention is to use 3 spaces for indentation
    space_indent_count = math.ceil(space_indent_count / 3)
    return space_indent_count > tab_indent_count or tab_indent_count == 0


def get_files_to_process(data_dir: str, skip_non_utf8_files: bool = True):
    # returns a list of unique ada files in the data ada_code_bases directory
    file_types_to_keep = {".ads", ".adb", ".gpr"}
    hashes = set()
    files_to_process = []

    for root, _, files in os.walk(data_dir):
        for file in files:
            file_type = os.path.splitext(file)[1]
            if file_type in file_types_to_keep:
                file_path = os.path.join(root, file)
                hash = file_hash(file_path)
                if hash not in hashes:
                    hashes.add(hash)
                    # If the file is not UTF-8, skip it
                    if skip_non_utf8_files:
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                f.read()
                            # We only want to process files that are mostly space indented
                            if not is_file_mostly_space_indented(file_path):
                                continue
                            files_to_process.append(file_path)
                        except UnicodeDecodeError:
                            continue
                    else:
                        files_to_process.append(file_path)
    return files_to_process

UNINDENT_LINES = re.compile(r'^\s*(begin\s*$|end(;|\s+\S+))')

def read_and_split_file(file_path: str) -> List[str]:
    # Read a file and return the contents as a string
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Split the List of strings into a List of List of Strings
    # Each inner List is delimited by a line that matches the regex UNINDENT_LINES
    # This is used to split the file into predictable blocks of code
    all_blocks = []
    block = []
    for line in lines:
        if len(line.strip()) == 0:
            continue
        if UNINDENT_LINES.match(line):
            if len(block) > 0:
                all_blocks.append(block)
                block = []
        else:
            block.append(line)
    if len(block) > 0:
        print("Warning: file did not end with an end statement")
        print(block)
        all_blocks.append(block)

    # Turn each block into a single string
    all_blocks = [''.join(x) for x in all_blocks]
    return all_blocks

files_to_process = get_files_to_process(DATA_DIR)
print(f"Number of files to process: {len(files_to_process)}")

code_blocks = []
for file_path in files_to_process:
    code_blocks.extend(read_and_split_file(file_path))

print(f"Number of code blocks: {len(code_blocks)}")

for i in range(len(code_blocks)):
    code_blocks[i] = Tokenizer.encode(code_blocks[i])

all_blocks_split_points = []
new_blocks = []
for block in code_blocks:
    block_split_points = [i+1 for i, x in enumerate(block) if x == Tokenizer.newline_token()]
    if len(block_split_points) > 1:
        block_split_points.pop()  # We don't actually want to split on the last newline
        all_blocks_split_points.append(block_split_points)
        new_blocks.append(block)

code_blocks = new_blocks

def count_leading_spaces(block):
    count = 0
    for token in block:
        if token == Tokenizer.space_token():
            count += 1
        else:
            break
    return count

class DataPoint:
    def __init__(self, input: List[int], label: int, next_line: List[int]) -> None:
        # self._input = input
        self._numpy_input = np.array(input)
        # self._label = label
        self._one_hot_label = tf.keras.utils.to_categorical(label, num_classes=MAX_INDENTATION)
        # self._next_line = next_line

    @property
    def training_input(self) -> np.array:
        return self._numpy_input

    @property
    def one_hot_label(self) -> np.array:
        return self._one_hot_label

    # @property
    # def input(self) -> List[int]:
    #     return self._input

    # @property
    # def label(self) -> int:
    #     return self._label

    # @property
    # def next_line(self) -> List[int]:
    #     return self._next_line

    # def __str__(self) -> str:
    #     return f"Input: {Tokenizer.decode(self._input)}\nLabel: {self._label}\nNext Line: {Tokenizer.decode(self._next_line)}"

data_points = []
for i, block in enumerate(code_blocks):
    split_points = all_blocks_split_points[i]
    for j in split_points:
        sub_block = Tokenizer.resize(block[:j], MAX_TOKENS)
        next_newline = block[j:].index(Tokenizer.newline_token()) + 1
        next_line = block[j:j+next_newline]
        label = count_leading_spaces(next_line)
        data_points.append(DataPoint(sub_block, label, next_line))

X = np.array([x.training_input for x in data_points])
y = np.array([x.one_hot_label for x in data_points])

del data_points
del code_blocks
del all_blocks_split_points
del new_blocks
del files_to_process


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


num_layers = 6
d_model = 64
dff = 256
num_heads = 8
dropout_rate = 0.1


model = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    vocab_size=Tokenizer.n_vocab(),
    dropout_rate=dropout_rate,
    max_indentation=MAX_INDENTATION)

learning_rate = CustomSchedule(d_model)
# adam = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
adam = tf.keras.optimizers.Adam(learning_rate=0.0003)

model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

model(X[0:1])

model.summary()

checkpoint_path = "checkpoints/indentation_prediction_v5.ckpt"
callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=0)

# if os.path.exists(checkpoint_path + ".index"):
#     print("Loading weights from checkpoint")
#     model.load_weights(checkpoint_path)

model.fit(X, y, epochs=100, batch_size=32, verbose=1, validation_split=0.1, callbacks=[callback])

test = model(X[0:1], training=False, mask=None)
