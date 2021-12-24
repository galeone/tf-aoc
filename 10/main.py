#!/usr/bin/env python3

"""
https://adventofcode.com/2021/day/10
"""

import sys

import tensorflow as tf


class Tokenizer(tf.Module):
    def __init__(self):
        super().__init__()

        self._opening_tokens = tf.constant(["(", "[", "{", "<"])
        self._closing_tokens = tf.constant([")", "]", "}", ">"])

        self._syntax_score_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                self._closing_tokens,
                tf.constant([3, 57, 1197, 25137], tf.int64),
            ),
            default_value=tf.constant(-1, tf.int64),
        )

        self._autocomplete_score_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                self._closing_tokens,
                tf.constant([1, 2, 3, 4], tf.int64),
            ),
            default_value=tf.constant(-1, tf.int64),
        )

        self._open_close = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                self._opening_tokens,
                self._closing_tokens,
            ),
            default_value="",
        )

        self._close_open = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                self._closing_tokens,
                self._opening_tokens,
            ),
            default_value="",
        )

        self._pos = tf.Variable(0, dtype=tf.int64)
        self._corrupted_score = tf.Variable(0, dtype=tf.int64)

    @tf.function
    def corrupted(self, dataset):
        for line in dataset:
            stack = tf.TensorArray(tf.string, size=0, dynamic_size=True)
            self._pos.assign(0)
            for position in tf.range(tf.size(line)):
                current_token = line[position]
                if tf.reduce_any(tf.equal(current_token, self._opening_tokens)):
                    stack = stack.write(tf.cast(self._pos, tf.int32), current_token)
                    self._pos.assign_add(1)
                else:
                    expected_token = self._open_close.lookup(
                        stack.read(tf.cast(self._pos - 1, tf.int32))
                    )
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
                            self._syntax_score_table.lookup(current_token)
                        )
                        break
        return self._corrupted_score

    @tf.function
    def incomplete(self, dataset):
        scores = tf.TensorArray(tf.int64, size=0, dynamic_size=True)
        for line in dataset:
            stack = tf.TensorArray(tf.string, size=0, dynamic_size=True)
            self._pos.assign(0)

            for position in tf.range(tf.size(line)):
                current_token = line[position]
                if tf.reduce_any(tf.equal(current_token, self._opening_tokens)):
                    stack = stack.write(tf.cast(self._pos, tf.int32), current_token)
                    self._pos.assign_add(1)
                else:
                    expected_token = self._open_close.lookup(
                        stack.read(tf.cast(self._pos - 1, tf.int32))
                    )
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
                        self._pos.assign(0)
                        break

            if tf.not_equal(self._pos, 0):  # stack not completely unrolled
                unstacked = tf.squeeze(
                    tf.reverse(
                        tf.expand_dims(stack.stack()[: self._pos], axis=0), axis=[1]
                    )
                )
                closing = self._open_close.lookup(unstacked)
                tf.print("Unstacked missing part: ", closing, summarize=-1)

                # Use pos variable as line score
                self._pos.assign(0)
                for idx in tf.range(tf.shape(closing)[0]):
                    char = closing[idx]
                    self._pos.assign(self._pos * 5)
                    self._pos.assign_add(self._autocomplete_score_table.lookup(char))

                scores = scores.write(scores.size(), self._pos)

        # sort the scores
        scores_tensors = tf.sort(scores.stack())
        # tf.print(scores_tensors)
        return scores_tensors[(tf.shape(scores_tensors)[0] - 1) // 2]


def main():
    """Entrypoint. Suppose the "input" file is in the cwd."""

    dataset = tf.data.TextLineDataset("input").map(tf.strings.bytes_split)
    tokenier = Tokenizer()

    tf.print("Part one: ", tokenier.corrupted(dataset))
    tf.print("Part two: ", tokenier.incomplete(dataset))


if __name__ == "__main__":
    sys.exit(main())
