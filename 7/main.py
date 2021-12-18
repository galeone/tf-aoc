#!/usr/bin/env python3

"""
https://adventofcode.com/2021/day/7
"""

import sys

import tensorflow as tf


def main():
    """Entrypoint. Suppose the "input" file is in the cwd."""

    dataset = (
        tf.data.TextLineDataset("input")
        .map(lambda string: tf.strings.split(string, ","))
        .map(lambda string: tf.strings.to_number(string, out_type=tf.int64))
        .unbatch()
    )

    dataset_tensor = tf.convert_to_tensor(list(dataset))
    y, idx, count = tf.unique_with_counts(dataset_tensor, tf.int64)

    max_elements = tf.reduce_max(count)
    most_frequent_position = y[idx[tf.argmax(count)]]

    tf.print(max_elements, " in position ", most_frequent_position)

    neighborhood_size = tf.constant(
        tf.shape(dataset_tensor, tf.int64)[0] // tf.constant(2, tf.int64), tf.int64
    )
    # for every x in the nehighborhhod of the max_element
    # find the sum {p_i - x} < sum {p_i - y} for all x != y

    min_neigh_val = tf.clip_by_value(
        most_frequent_position - neighborhood_size,
        tf.constant(0, tf.int64),
        most_frequent_position,
    )

    max_val = tf.reduce_max(dataset_tensor) + 1
    max_neigh_val = tf.clip_by_value(
        most_frequent_position + neighborhood_size,
        most_frequent_position,
        max_val,
    )

    min_cost, found_position = tf.cast(-1, tf.uint64), -1
    for x in tf.range(min_neigh_val, max_neigh_val):
        cost = tf.cast(tf.reduce_sum(tf.abs(dataset_tensor - x)), tf.uint64)
        if tf.less(cost, min_cost):
            min_cost = cost
            found_position = x
    tf.print("min_cost: ", min_cost, " in position: ", found_position)


if __name__ == "__main__":
    sys.exit(main())
