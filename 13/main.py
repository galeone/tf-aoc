#!/usr/bin/env python3

"""
https://adventofcode.com/2021/day/13
"""

import sys

import tensorflow as tf


def main():
    """Entrypoint. Suppose the "input" file is in the cwd."""

    dataset = tf.data.TextLineDataset("input")
    # Split in two: coordinates, and fold instructions
    # Empty line is the separator

    for idx, line in enumerate(dataset):
        if tf.equal(tf.strings.length(line), 0):
            coordinates = dataset.take(idx)
            instructions = dataset.skip(idx + 1)
            break

    coordinates = coordinates.map(lambda line: tf.strings.split(line, ",")).map(
        lambda pair: tf.strings.to_number(pair, tf.int32)
    )

    instructions = instructions.map(
        lambda line: tf.strings.regex_replace(
            line, r"fold along ([x,y])=(\d+)", r"\1=\2"
        )
    ).map(lambda line: tf.strings.split(line, "="))

    coordinates = tf.convert_to_tensor(list(coordinates))
    shape = (
        tf.convert_to_tensor(
            [tf.reduce_max(coordinates[:, 0]), tf.reduce_max(coordinates[:, 1])]
        )
        + 1
    )

    sheet = tf.Variable(tf.zeros(shape, tf.int32))
    sheet.scatter_nd_update(coordinates, tf.repeat(1, tf.shape(coordinates)[0]))

    for idx, fold in enumerate(instructions):
        axis = fold[0]
        coord = tf.strings.to_number(fold[1], tf.int32)

        if tf.equal(axis, "y"):
            sub = sheet[:, coord + 1 :]
            indices_y, indices_x = tf.meshgrid(
                tf.range(tf.shape(sheet)[0]), tf.range(coord + 1, tf.shape(sheet)[1])
            )
            indices = tf.stack([indices_y, indices_x], axis=-1)
        else:
            sub = sheet[coord + 1 :, :]
            indices_y, indices_x = tf.meshgrid(
                tf.range(coord + 1, tf.shape(sheet)[0]), tf.range(tf.shape(sheet)[1])
            )
            indices = tf.stack([indices_y, indices_x], axis=-1)

        indices = tf.reshape(indices, (-1, 2))
        updates = tf.repeat(0, tf.shape(indices)[0])
        # Set source positions to zero
        sheet.scatter_nd_update(indices, updates)

        # If axis == y, fold over left (tensorflow orientation)
        if tf.equal(axis, "y"):
            # tf.linalg.LinearOperatorPermutation uses the same
            # idea of tf.transpose but instead of swapping dimensions it swappes
            # ROWS.

            # Hence, for this folding we need to transpose first
            sub = tf.transpose(sub)

            perm = tf.range(tf.shape(sub)[0] - 1, -1, -1)
            operator = tf.linalg.LinearOperatorPermutation(perm)
            sub = tf.cast(operator.matmul(tf.cast(sub, tf.float32)), tf.int32)

            # back to the original position
            sub = tf.transpose(sub)

            # add on the left side the submatrix
            indices_y, indices_x = tf.meshgrid(
                tf.range(tf.shape(sheet)[0]), tf.range(coord)
            )
            indices = tf.stack([indices_y, indices_x], axis=-1)
            updates = tf.transpose(sub)
            sheet.scatter_nd_add(indices, updates)
        # If axis == x, fold up (tensorflow orientation)
        if tf.equal(axis, "x"):
            perm = tf.range(tf.shape(sub)[0] - 1, -1, -1)
            operator = tf.linalg.LinearOperatorPermutation(perm)
            sub = tf.cast(operator.matmul(tf.cast(sub, tf.float32)), tf.int32)
            indices_y, indices_x = tf.meshgrid(
                tf.range(coord), tf.range(tf.shape(sheet)[1])
            )
            indices = tf.stack([indices_y, indices_x], axis=-1)
            updates = tf.transpose(sub)
            sheet.scatter_nd_add(indices, updates)

        if tf.equal(idx, 0):
            tf.print(
                "Part one: ",
                tf.reduce_sum(tf.cast(tf.greater_equal(sheet, 1), tf.int64)),
            )

    display = tf.transpose(sheet)
    tf.print(display, summarize=-1)


if __name__ == "__main__":
    sys.exit(main())
