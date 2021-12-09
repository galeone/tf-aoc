#!/usr/bin/env python3

# pylint: disable=line-too-long
"""
Based on your calculations, the planned course doesn't seem to make any sense. You find the submarine manual and discover that the process is actually slightly more complicated.

In addition to horizontal position and depth, you'll also need to track a third value, aim, which also starts at 0. The commands also mean something entirely different than you first thought:

down X increases your aim by X units.
up X decreases your aim by X units.
forward X does two things:
It increases your horizontal position by X units.
It increases your depth by your aim multiplied by X.
Again note that since you're on a submarine, down and up do the opposite of what you might expect: "down" means aiming in the positive direction.

Now, the above example does something different:

forward 5 adds 5 to your horizontal position, a total of 5. Because your aim is 0, your depth does not change.
down 5 adds 5 to your aim, resulting in a value of 5.
forward 8 adds 8 to your horizontal position, a total of 13. Because your aim is 5, your depth increases by 8*5=40.
up 3 decreases your aim by 3, resulting in a value of 2.
down 8 adds 8 to your aim, resulting in a value of 10.
forward 2 adds 2 to your horizontal position, a total of 15. Because your aim is 10, your depth increases by 2*10=20 to a total of 60.
After following these new instructions, you would have a horizontal position of 15 and a depth of 60. (Multiplying these produces 900.)

Using this new interpretation of the commands, calculate the horizontal position and depth you would have after following the planned course. What do you get if you multiply your final horizontal position by your final depth?
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
    INCREASE_AIM = auto()
    DECREASE_AIM = auto()
    INCREASE_HORIZONTAL_MUTIPLY_BY_AIM = auto()


class PositionCounter:
    """Stateful counter. Get the final horizontal position and depth, keeping track of the aim."""

    def __init__(self):
        self._horizontal_position = tf.Variable(0, trainable=False, dtype=tf.int64)
        self._depth = tf.Variable(0, trainable=False, dtype=tf.int64)
        self._aim = tf.Variable(0, trainable=False, dtype=tf.int64)

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
            elif tf.equal(action, Action.INCREASE_HORIZONTAL_MUTIPLY_BY_AIM):
                self._horizontal_position.assign_add(amount)
                self._depth.assign_add(self._aim * amount)
            elif tf.equal(action, Action.DECREASE_AIM):
                self._aim.assign_sub(amount)
            elif tf.equal(action, Action.INCREASE_AIM):
                self._aim.assign_add(amount)
        return self._horizontal_position, self._depth


def main():
    """Entrypoint. Suppose the "input" file is in the cwd."""

    def _processor(line):
        splits = tf.strings.split(line)
        direction = splits[0]
        amount = splits[1]

        if tf.equal(direction, "forward"):
            action = Action.INCREASE_HORIZONTAL_MUTIPLY_BY_AIM
        elif tf.equal(direction, "down"):
            action = Action.INCREASE_AIM
        elif tf.equal(direction, "up"):
            action = Action.DECREASE_AIM
        else:
            action = -1
        #    tf.debugging.Assert(False, f"Unhandled direction: {direction}")

        amount = tf.strings.to_number(amount, out_type=tf.int64)
        return action, amount

    dataset = tf.data.TextLineDataset("input").map(_processor)

    counter = PositionCounter()
    horizontal_position, depth = counter(dataset)
    result = horizontal_position * depth
    tf.print("[part two] result: ", result)


if __name__ == "__main__":
    sys.exit(main())
