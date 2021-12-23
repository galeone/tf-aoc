#!/usr/bin/env python3

"""
https://adventofcode.com/2021/day/9
"""

import sys

import tensorflow as tf


def main():
    """Entrypoint. Suppose the "input" file is in the cwd."""

    dataset = (
        tf.data.TextLineDataset("input")
        .map(tf.strings.bytes_split)
        .map(lambda string: tf.strings.to_number(string, out_type=tf.int32))
    )

    image = tf.convert_to_tensor(list(dataset))
    shape = tf.shape(image)
    max_value = tf.reduce_max(image)

    if tf.not_equal(tf.math.mod(shape[0], 3), 0):
        pad_h = shape[0] - (shape[0] // 3 + 3)
    else:
        pad_h = 0
    if tf.not_equal(tf.math.mod(shape[1], 3), 0):
        pad_w = shape[1] - (shape[1] // 3 + 3)
    else:
        pad_w = 0

    padded_image = tf.pad(
        image, [[0, pad_w], [1, pad_h]], mode="CONSTANT", constant_values=max_value
    )

    shape = tf.shape(padded_image)

    neigh_mask = tf.constant([(-1, 0), (0, -1), (1, 0), (0, 1), (0, 0)])

    count = tf.Variable(0)
    for y in tf.range(shape[0] - 1):
        for x in tf.range(shape[1] - 1):
            if tf.logical_and(tf.less(y, 1), tf.less(x, 1)):
                mask = neigh_mask[2:]
            elif tf.less(y, 1):
                mask = neigh_mask[1:]
            elif tf.less(x, 1):
                mask = tf.concat([[neigh_mask[0]], neigh_mask[2:]], axis=0)
            else:
                mask = neigh_mask

            coords = tf.convert_to_tensor([y, x]) + mask
            neighborhood = tf.gather_nd(padded_image, coords)

            minval = tf.reduce_min(neighborhood)
            if tf.logical_and(
                tf.reduce_any(tf.not_equal(neighborhood, minval)),
                tf.equal(minval, padded_image[y, x]),
            ):
                count.assign_add(1 + padded_image[y, x])
    tf.print(count)


if __name__ == "__main__":
    sys.exit(main())
