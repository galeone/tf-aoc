#!/usr/bin/env python3

"""
https://adventofcode.com/2021/day/6
"""

import sys

import tensorflow as tf


@tf.function
def evolve(initial_state: tf.Tensor, days: tf.Tensor):
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


# Returning the state is useless, we are interested only in the # of elements
# the last state contains.
# Hence we can try to reason about this number instead of the state.
# Given an initial state of 1 2 3 4 1 -> tot = 5
# After a time step, we are in the state 0 1 2 3 0 -> tot 5
# when we have zeroes, it means will span eights on the next time step.
# After a time step we are in the state 6 0 1 2 6 8 8
# If we only keep track of the number of fish in a certain state, we can compress all the information
# Something like
# 1 -> 10 #there are 10 fish at status 1. Nothing happens to the lenght
# 2 -> 1 # there's one fish in status 2. Nothing happens to the lenght
# k -> v # there's v fish in status k. Depends on on k.
# If k = 0, it means there are v fish in status 0, hence the lenght (number of fish) will increment by v
# on the next time step, while the v becomes 0 for k=0 and the number of fishes in status 8 increases by k.a
@tf.function
def count(initial_state: tf.Tensor, days: tf.Tensor):
    hashmap = tf.lookup.experimental.MutableHashTable(
        tf.int32, tf.int32, tf.constant(0, tf.int32)
    )
    keys, _, count = tf.unique_with_counts(initial_state, tf.int32)
    hashmap.insert(keys, count)

    def _count_elements():
        return tf.reduce_sum(hashmap.lookup(tf.range(9)))

    for day in tf.range(1, days + 1):
        yesterday_state = hashmap.lookup(tf.range(9))
        if tf.greater(yesterday_state[0], 0):
            tf.print("shift from: ", yesterday_state, summarize=-1)
            # handled values in keys [0, 5], [7, 8]
            today_state = tf.tensor_scatter_nd_update(
                yesterday_state,
                tf.concat([tf.reshape(tf.range(6), (6, 1)), [[6], [7], [8]]], axis=0),
                tf.concat(
                    [
                        hashmap.lookup(tf.range(1, 7)),
                        [yesterday_state[7]],
                        [yesterday_state[8]],
                        [yesterday_state[0]],
                    ],
                    axis=0,
                ),
            )
            tf.print("to: ", today_state, summarize=-1)
            today_state = tf.tensor_scatter_nd_add(
                today_state, [[6]], [yesterday_state[0]]
            )
            tf.print(
                "updated position 6 and 8 summing ",
                yesterday_state[0],
                "obtaining ",
                summarize=-1,
            )
        else:
            # shift the the left all the map
            # put a 0 in 8 position
            updates = tf.concat(
                [
                    tf.unstack(tf.gather(yesterday_state, tf.range(1, 9))),
                    [tf.constant(0)],
                ],
                axis=0,
            )
            indices = tf.reshape(tf.range(9), (9, 1))
            today_state = tf.tensor_scatter_nd_update(yesterday_state, indices, updates)
            tf.print("complete shift, from ", yesterday_state)

        tf.print("day ", day, " : ", today_state, summarize=-1)
        hashmap.insert(tf.range(9), today_state)
        tf.print(_count_elements())
    return _count_elements()


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

    # days = tf.constant(80, tf.int32)
    # last_state = evolve(initial_state, days)
    # tf.print("# fish after ", days, " days: ", tf.size(last_state))

    days = tf.constant(80, tf.int32)
    count(initial_state, days)


if __name__ == "__main__":
    sys.exit(main())
