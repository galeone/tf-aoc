#!/usr/bin/env python3

"""
https://adventofcode.com/2021/day/9
"""

import sys
from typing import Tuple

import tensorflow as tf


class Finder(tf.Module):
    def __init__(self, dataset: tf.data.Dataset):

        super().__init__()

        self._count = tf.Variable(0)
        self._dataset = dataset
        self._image = tf.convert_to_tensor(list(self._dataset))
        self._shape = tf.shape(self._image)
        self._max_value = tf.reduce_max(self._image)
        self._prev = tf.Variable(0)

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
        self._norm = tf.Variable(tf.zeros(self._padded_shape, dtype=tf.int32) - 1)
        self._stop = tf.Variable(False)
        self._neigh_mask = tf.constant([(-1, 0), (0, -1), (1, 0), (0, 1)])
        self._queue = tf.queue.FIFOQueue(-1, [tf.int32])

    @tf.function
    def _four_neigh(
        self, grid: tf.Tensor, center: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        y, x = center[0], center[1]

        if tf.logical_and(tf.less(y, 1), tf.less(x, 1)):
            mask = self._neigh_mask[2:]
        elif tf.less(y, 1):
            mask = self._neigh_mask[1:]
        elif tf.less(x, 1):
            mask = tf.concat([[self._neigh_mask[0]], self._neigh_mask[2:]], axis=0)
        else:
            mask = self._neigh_mask

        coords = center + mask

        neighborhood = tf.gather_nd(grid, coords)
        return neighborhood, coords

    @tf.function
    def low_points(self) -> Tuple[tf.Tensor, tf.Tensor]:
        self._count.assign(0)
        ta = tf.TensorArray(tf.int32, size=0, dynamic_size=True)

        for y in tf.range(self._padded_shape[0] - 1):
            for x in tf.range(self._padded_shape[1] - 1):
                center = tf.convert_to_tensor([y, x])
                neighborhood, _ = self._four_neigh(self._padded_image, center)
                extended_neighborhood = tf.concat(
                    [tf.expand_dims(self._padded_image[y, x], axis=0), neighborhood],
                    axis=0,
                )

                minval = tf.reduce_min(extended_neighborhood)
                if tf.logical_and(
                    tf.reduce_any(tf.not_equal(extended_neighborhood, minval)),
                    tf.equal(minval, self._padded_image[y, x]),
                ):
                    self._count.assign_add(1 + self._padded_image[y, x])

                    ta = ta.write(ta.size(), center)

        return ta.stack(), self._count

    @tf.function
    def basins(self) -> tf.Tensor:
        batch = tf.reshape(
            self._padded_image, (1, self._padded_shape[0], self._padded_shape[1], 1)
        )
        gradients = tf.squeeze(tf.image.image_gradients(batch), axis=1)

        y_grad, x_grad = gradients[0], gradients[1]

        # Gradienti magnitude is constant where there are no changes
        # Increases or stray constants from the low point (seed)
        norm = tf.cast(tf.norm(tf.cast(y_grad + x_grad, tf.float32), axis=-1), tf.int32)
        # Set the basin thresholds to -1 (where the 9s are)
        norm = tf.where(tf.equal(self._padded_image, 9), -1, norm)
        self._norm.assign(norm)

        # For every se_posd, "propagate" in a flood fill-fashion.
        # The -1s are the thresholds
        seeds = self.low_points()[0]
        ta = tf.TensorArray(tf.int32, size=3)
        ta.unstack([0, 0, 0])
        for idx in tf.range(2, tf.shape(seeds)[0] + 2):
            # Fill with idx (watershed like: different colors)
            seed = seeds[idx - 2]
            y = seed[0]
            x = seed[1]

            # Set the seed position to the label
            self._norm.scatter_nd_update([[y, x]], [-idx])

            # Find the 4 neighborhood, and get the values != -1
            neighborhood, neigh_coords = self._four_neigh(self._norm, seed)
            update_coords = tf.gather_nd(
                neigh_coords, tf.where(tf.not_equal(neighborhood, -1))
            )
            if tf.greater(tf.size(update_coords), 0):
                self._queue.enqueue_many(update_coords)
                while tf.greater(self._queue.size(), 0):
                    pixel = self._queue.dequeue()
                    # Update this pixel to the label value
                    py, px = pixel[0], pixel[1]
                    self._norm.scatter_nd_update([[py, px]], [-idx])
                    px_neigh_vals, px_neigh_coords = self._four_neigh(self._norm, pixel)
                    px_update_coords = tf.gather_nd(
                        px_neigh_coords,
                        tf.where(
                            tf.logical_and(
                                tf.not_equal(px_neigh_vals, -1),
                                tf.not_equal(px_neigh_vals, -idx),
                            )
                        ),
                    )
                    if tf.greater(tf.size(px_update_coords), 0):
                        self._queue.enqueue_many(px_update_coords)

            basin_size = tf.reduce_sum(tf.cast(tf.equal(self._norm, -idx), 3))
            if tf.greater(basin_size, ta.read(0)):
                first = basin_size
                second = ta.read(0)
                third = ta.read(1)
                ta = ta.unstack([first, second, third])
            elif tf.greater(basin_size, ta.read(1)):
                first = ta.read(0)
                second = basin_size
                third = ta.read(1)
                ta = ta.unstack([first, second, third])
            elif tf.greater(basin_size, ta.read(2)):
                ta = ta.write(2, basin_size)

        # tf.print(self._norm, summarize=-1)
        return tf.reduce_prod(ta.stack())


def main():
    """Entrypoint. Suppose the "input" file is in the cwd."""

    dataset = (
        tf.data.TextLineDataset("input")
        .map(tf.strings.bytes_split)
        .map(lambda string: tf.strings.to_number(string, out_type=tf.int32))
    )

    finder = Finder(dataset)
    tf.print("Part one: ", finder.low_points()[1])
    tf.print("Part two: ", finder.basins())


if __name__ == "__main__":
    sys.exit(main())
