#!/usr/bin/env python3

"""
https://adventofcode.com/2021/day/2
"""

import sys
from enum import IntEnum, auto
from typing import Tuple

import tensorflow as tf


class Action(IntEnum):
    """Action enum, to map the direction read to an action to perform."""

    INCREASE_HORIZONTAL = auto()
    INCREASE_DEPTH = auto()
    DECREASE_DEPTH = auto()


class PositionCounter:
    """Stateful counter. Get the final horizontal position and depth."""

    def __init__(self):
        self._horizontal_position = tf.Variable(0, trainable=False, dtype=tf.int64)
        self._depth = tf.Variable(0, trainable=False, dtype=tf.int64)

    def reset(self):
        """Reset the counter state."""
        self._horizontal_position.assign(0)
        self._depth.assign(0)

    @tf.function
    def __call__(self, dataset: tf.data.Dataset) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args:
            dataset: dataset yielding tuples (action, value), where action is
                     a valida Action enum.
        Returns:
            (horizontal_position, depth)
        """
        for action, amount in dataset:
            if tf.equal(action, Action.INCREASE_DEPTH):
                self._depth.assign_add(amount)
            elif tf.equal(action, Action.DECREASE_DEPTH):
                self._depth.assign_sub(amount)
            elif tf.equal(action, Action.INCREASE_HORIZONTAL):
                self._horizontal_position.assign_add(amount)
        return self._horizontal_position, self._depth


def main():
    """Entrypoint. Suppose the "input" file is in the cwd."""

    def _processor(line):
        splits = tf.strings.split(line)
        direction = splits[0]
        amount = splits[1]

        if tf.equal(direction, "forward"):
            action = Action.INCREASE_HORIZONTAL
        elif tf.equal(direction, "down"):
            action = Action.INCREASE_DEPTH
        elif tf.equal(direction, "up"):
            action = Action.DECREASE_DEPTH
        else:
            action = -1
        #    tf.debugging.Assert(False, f"Unhandled direction: {direction}")

        amount = tf.strings.to_number(amount, out_type=tf.int64)
        return action, amount

    dataset = tf.data.TextLineDataset("input").map(_processor)

    counter = PositionCounter()
    horizontal_position, depth = counter(dataset)
    result = horizontal_position * depth
    tf.print("[part one] result: ", result)


if __name__ == "__main__":
    sys.exit(main())
