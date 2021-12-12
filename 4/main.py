#!/usr/bin/env python3

"""
https://adventofcode.com/2021/day/4
"""

import sys
from typing import Tuple

import tensorflow as tf


class Bingo(tf.Module):
    def __init__(self):
        # Assign every board in a TensorArray so we can read/write every board
        self._ta = tf.TensorArray(dtype=tf.int64, size=1, dynamic_size=True)

        self._stop = tf.Variable(False, trainable=False)

        self._winner_board = tf.Variable(
            tf.zeros((5, 5), dtype=tf.int64), trainable=False
        )
        self._last_number = tf.Variable(0, trainable=False, dtype=tf.int64)

    @staticmethod
    def is_winner(board: tf.Tensor) -> tf.Tensor:
        rows = tf.reduce_sum(board, axis=0)
        cols = tf.reduce_sum(board, axis=1)

        return tf.logical_or(
            tf.reduce_any(tf.equal(rows, -5)), tf.reduce_any(tf.equal(cols, -5))
        )

    # @tf.function
    def __call__(
        self,
        extractions: tf.data.Dataset,
        boards: tf.data.Dataset,
        first_winner: tf.Tensor = tf.constant(True),
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # Convert the datasaet to a tensor  and assign it to the ta
        # use the tensor to get its shape and know the numnber of boards
        tensor_boards = tf.convert_to_tensor(list(boards))  # pun intended
        tot_boards = tf.shape(tensor_boards)[0]
        self._ta = self._ta.unstack(tensor_boards)

        # Remove the number from the board when extracted
        # The removal is just the set of the number to -1
        # When a row or a column becomes a line of -1s then bingo!
        for number in extractions:
            if self._stop:
                break
            for idx in tf.range(tot_boards):
                board = self._ta.read(idx)
                board = tf.where(tf.equal(number, board), -1, board)
                if self.is_winner(board):
                    self._winner_board.assign(board)
                    self._last_number.assign(number)
                    if first_winner:
                        self._stop.assign(tf.constant(True))
                        break
                    # When searching for the last winner
                    # we just invalidate every winning board
                    # by setting all the values to zero
                    board = tf.zeros_like(board)
                self._ta = self._ta.write(idx, board)
        return self._winner_board, self._last_number


def main():
    """Entrypoint. Suppose the "input" file is in the cwd."""

    dataset = tf.data.TextLineDataset("input")

    # The first row is a csv line containing the number extracted in sequence
    extractions = (
        dataset.take(1)
        .map(lambda line: tf.strings.split(line, ","))
        .map(lambda string: tf.strings.to_number(string, out_type=tf.int64))
        .unbatch()
    )

    # All the other rows are the boards, every 5 lines containing an input
    # is a board. We can organize the boards as elements of the dataset, a dataset of boards
    boards = (
        dataset.skip(1)
        .filter(lambda line: tf.greater(tf.strings.length(line), 0))
        .map(tf.strings.split)
        .map(
            lambda string: tf.strings.to_number(string, out_type=tf.int64)
        )  # row with 5 numbers
        .batch(5)  # board 5 rows, 5 columns
    )

    bingo = Bingo()
    winner_board, last_number = bingo(extractions, boards)

    def _score(board, number):
        tf.print("Winner board: ", board)
        tf.print("Last number: ", number)

        # Sum all unmarked numbers
        unmarked_sum = tf.reduce_sum(
            tf.gather_nd(board, tf.where(tf.not_equal(board, -1)))
        )
        tf.print("Unmarked sum: ", unmarked_sum)

        final_score = unmarked_sum * number
        tf.print("Final score: ", final_score)

    _score(winner_board, last_number)

    ## --- Part Two ---
    # Figure out the last board that will win
    bingo = Bingo()
    last_winner_board, last_number = bingo(extractions, boards, tf.constant(False))
    _score(last_winner_board, last_number)


if __name__ == "__main__":
    sys.exit(main())
