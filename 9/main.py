#!/usr/bin/env python3

"""
https://adventofcode.com/2021/day/9
"""

import sys

import tensorflow as tf


class Finder(tf.Module):
    def __init__(self, dataset: tf.data.Dataset):

        super().__init__()

        self._count = tf.Variable(0)
        self._dataset = dataset
        self._image = tf.convert_to_tensor(list(self._dataset))
        self._shape = tf.shape(self._image)
        self._max_value = tf.reduce_max(self._image)

        if tf.not_equal(tf.math.mod(self._shape[0], 3), 0):
            pad_h = self._shape[0] - (self._shape[0] // 3 + 3)
        else:
            pad_h = 0
        if tf.not_equal(tf.math.mod(self._shape[1], 3), 0):
            pad_w = self._shape[1] - (self._shape[1] // 3 + 3)
        else:
            pad_w = 0

        self._padded_image = tf.pad(
            self._image,
            [[0, pad_w], [1, pad_h]],
            mode="CONSTANT",
            constant_values=self._max_value,
        )

        self._padded_shape = tf.shape(self._padded_image)

    @tf.function
    def risk_levels(self) -> tf.Tensor:
        neigh_mask = tf.constant([(-1, 0), (0, -1), (1, 0), (0, 1), (0, 0)])

        for y in tf.range(self._padded_shape[0] - 1):
            for x in tf.range(self._padded_shape[1] - 1):
                if tf.logical_and(tf.less(y, 1), tf.less(x, 1)):
                    mask = neigh_mask[2:]
                elif tf.less(y, 1):
                    mask = neigh_mask[1:]
                elif tf.less(x, 1):
                    mask = tf.concat([[neigh_mask[0]], neigh_mask[2:]], axis=0)
                else:
                    mask = neigh_mask

                coords = tf.convert_to_tensor([y, x]) + mask
                neighborhood = tf.gather_nd(self._padded_image, coords)

                minval = tf.reduce_min(neighborhood)
                if tf.logical_and(
                    tf.reduce_any(tf.not_equal(neighborhood, minval)),
                    tf.equal(minval, self._padded_image[y, x]),
                ):
                    self._count.assign_add(1 + self._padded_image[y, x])
        return self._count

    @tf.function
    def basins(self) -> tf.Tensor:
        pass


def main():
    """Entrypoint. Suppose the "input" file is in the cwd."""

    dataset = (
        tf.data.TextLineDataset("input")
        .map(tf.strings.bytes_split)
        .map(lambda string: tf.strings.to_number(string, out_type=tf.int32))
    )

    finder = Finder(dataset)
    tf.print("Part one: ", finder.risk_levels())


if __name__ == "__main__":
    sys.exit(main())
