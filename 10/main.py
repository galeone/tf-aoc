#!/usr/bin/env python3

"""
https://adventofcode.com/2021/day/10
"""

import sys

import tensorflow as tf


class Tokenizer(tf.Module):
    def __init__(self):
        super().__init__()
        self._score_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant([")", "]", "}", ">"]), tf.constant([3, 57, 1197, 25137])
            ),
            default_value=-1,
        )

        self._open_close = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant(["(", "[", "{", "<"]),
                tf.constant([")", "]", "}", ">"]),
            ),
            default_value="",
        )
        self._opening_tokens = self._open_close.export()[0]

        self._pos = tf.Variable(0)
        self._corrupted_score = tf.Variable(0)

    @tf.function
    def corrupted(self, dataset):
        # All the first illegal chars found

        for line in dataset:
            stack = tf.TensorArray(tf.string, size=0, dynamic_size=True)
            self._pos.assign(0)
            for position in tf.range(tf.size(line)):
                current_token = line[position]
                if tf.reduce_any(tf.equal(current_token, self._opening_tokens)):
                    stack = stack.write(self._pos, current_token)
                    self._pos.assign_add(1)
                else:
                    expected_token = self._open_close.lookup(stack.read(self._pos - 1))
                    self._pos.assign_sub(1)
                    if tf.not_equal(current_token, expected_token):
                        tf.print(
                            position,
                            ": expected: ",
                            expected_token,
                            " but found ",
                            current_token,
                            " instead",
                        )
                        self._corrupted_score.assign_add(
                            self._score_table.lookup(current_token)
                        )
                        break
        return self._corrupted_score


def main():
    """Entrypoint. Suppose the "input" file is in the cwd."""

    dataset = tf.data.TextLineDataset("input").map(tf.strings.bytes_split)

    tf.print(next(iter(dataset)), summarize=-1)

    tokenier = Tokenizer()

    tf.print("Part one: ", tokenier.corrupted(dataset))


if __name__ == "__main__":
    sys.exit(main())
