#!/usr/bin/env python3

# pylint: disable=line-too-long
"""
--- Day 2: Dive! ---
Now, you need to figure out how to pilot this thing.

It seems like the submarine can take a series of commands like forward 1, down 2, or up 3:

forward X increases the horizontal position by X units.
down X increases the depth by X units.
up X decreases the depth by X units.
Note that since you're on a submarine, down and up affect your depth, and so they have the opposite result of what you might expect.

The submarine seems to already have a planned course (your puzzle input). You should probably figure out where it's going. For example:

forward 5
down 5
forward 8
up 3
down 8
forward 2
Your horizontal position and depth both start at 0. The steps above would then modify them as follows:

forward 5 adds 5 to your horizontal position, a total of 5.
down 5 adds 5 to your depth, resulting in a value of 5.
forward 8 adds 8 to your horizontal position, a total of 13.
up 3 decreases your depth by 3, resulting in a value of 2.
down 8 adds 8 to your depth, resulting in a value of 10.
forward 2 adds 2 to your horizontal position, a total of 15.
After following these instructions, you would have a horizontal position of 15 and a depth of 10. (Multiplying these together produces 150.)

Calculate the horizontal position and depth you would have after following the planned course. What do you get if you multiply your final horizontal position by your final depth?
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
