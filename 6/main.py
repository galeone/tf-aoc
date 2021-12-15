#!/usr/bin/env python3

"""
https://adventofcode.com/2021/day/6
"""

import sys

import tensorflow as tf


@tf.function
def count(initial_state: tf.Tensor, days):
    ta = tf.TensorArray(tf.int32, size=tf.size(initial_state), dynamic_size=True)
    ta = ta.unstack(initial_state)

    for day in tf.range(1, days + 1):
        yesterday_state = ta.stack()
        index_map = tf.equal(yesterday_state, 0)
        if tf.reduce_any(index_map):
            indices = tf.where(index_map)
            transition_state = tf.tensor_scatter_nd_update(
                yesterday_state - 1,
                indices,
                tf.cast(tf.ones(tf.shape(indices)[0]) * 6, tf.int32),
            )
            ta = ta.unstack(transition_state)
            new_born = tf.reduce_sum(tf.cast(index_map, tf.int32))
            for n in tf.range(new_born):
                ta = ta.write(tf.size(transition_state, tf.int32) + n, 8)
        else:
            transition_state = yesterday_state - 1
            ta = ta.unstack(transition_state)
        today_state = ta.stack()
        # tf.print("after ", day, "days: ", today_state, summarize=-1)
    return today_state


def main():
    """Entrypoint. Suppose the "input" file is in the cwd."""

    initial_state = next(
        iter(
            tf.data.TextLineDataset("fake")
            .map(lambda string: tf.strings.split(string, ","))
            .map(lambda numbers: tf.strings.to_number(numbers, out_type=tf.int32))
            .take(1)
        )
    )

    days = tf.constant(256, tf.int32)
    last_state = count(initial_state, days)
    tf.print("# fish after ", days, " days: ", tf.size(last_state))


if __name__ == "__main__":
    sys.exit(main())
