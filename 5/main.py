#!/usr/bin/env python3

"""
https://adventofcode.com/2021/day/5
"""

import sys
from typing import Tuple

import tensorflow as tf


def interpolate(p1: tf.Tensor, p2: tf.Tensor):
    # +1 handles the case of p1 - p2 == 1
    # and linarg does not goes outside the last value
    # hence the +1 is not harmful
    norm = tf.norm(tf.cast(p1 - p2, tf.float32), ord=2) + 1
    return tf.cast(tf.math.ceil(tf.linspace(p1, p2, tf.cast(norm, tf.int64))), tf.int64)


def main():
    """Entrypoint. Suppose the "input" file is in the cwd."""

    def _get_segment(
        line: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        points = tf.strings.split(line, " -> ")
        p1 = tf.strings.split(points[0], ",")
        p2 = tf.strings.split(points[1], ",")

        x1 = tf.strings.to_number(p1[0], tf.int64)
        y1 = tf.strings.to_number(p1[1], tf.int64)
        x2 = tf.strings.to_number(p2[0], tf.int64)
        y2 = tf.strings.to_number(p2[1], tf.int64)
        return tf.convert_to_tensor((x1, y1)), tf.convert_to_tensor((x2, y2))

    dataset = tf.data.TextLineDataset("input").map(_get_segment)
    bbox_w = tf.reduce_max(list(dataset.map(lambda p1, p2: (p1[0], p2[0])))) + 1
    bbox_h = tf.reduce_max(list(dataset.map(lambda p1, p2: (p1[1], p2[1])))) + 1
    grid = tf.Variable(tf.zeros((bbox_w, bbox_h), dtype=tf.int64), trainable=False)
    for start, end in dataset:
        # Discrete interpolation between start and end
        # part 1 requires to consider only straight lines
        # (x1 = x2 or y1 = y2)
        # but I guess (hope) doing the generic discrete interpolation
        # will simplify part 2 (no idea, just a guess)
        if tf.reduce_any([tf.equal(start[0], end[0]), tf.equal(start[1], end[1])]):
            pixels = interpolate(start, end)
            grid.assign(
                tf.tensor_scatter_nd_add(
                    grid, pixels, tf.ones(tf.shape(pixels)[0], dtype=tf.int64)
                )
            )

    # tf.print(tf.transpose(grid, perm=(1, 0)), summarize=-1)
    n_overlaps = tf.reduce_sum(tf.cast(tf.greater(grid, 1), tf.int64))
    tf.print("# overlaps > 1: ", n_overlaps)


if __name__ == "__main__":
    sys.exit(main())
