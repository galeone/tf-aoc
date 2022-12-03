#!/usr/bin/env python3

"""
https://adventofcode.com/2021/day/1
"""

import sys

import tensorflow as tf


class IncreasesCounter:
    """Stateful counter. Counts the number of "increases"."""

    def __init__(self):
        self._count = tf.Variable(0, trainable=False, dtype=tf.int64)
        self._prev = tf.Variable(0, trainable=False, dtype=tf.int64)

    def reset(self):
        """Reset the counter state."""
        self._count.assign(0)
        self._prev.assign(0)

    @tf.function
    def __call__(self, dataset: tf.data.Dataset) -> tf.Tensor:
        """
        Args:
            dataset: the dataset containing the ordered sequence of numbers
                     to process.
        Returns:
            The number of increases. tf.Tensor, dtype=tf.int64
        """
        self._prev.assign(next(iter(dataset.take(1))))
        for number in dataset.skip(1):
            if tf.greater(number, self._prev):
                self._count.assign_add(1)
            self._prev.assign(number)
        return self._count


def main():
    """Entrypoint. Suppose the "input" file is in the cwd."""

    dataset = tf.data.TextLineDataset("input").map(
        lambda string: tf.strings.to_number(string, out_type=tf.int64)
    )

    counter = IncreasesCounter()
    increases = counter(dataset)
    tf.print("[part one] increases: ", increases)

    # --- Part Two ---

    # Create 3 datasets by shifting by 1 element every time
    # Craete batches of 3 elements, sum them. Create a new dataset
    # interleaving the values and call the counter over this dataset
    datasets = [dataset, dataset.skip(1), dataset.skip(2)]
    for idx, dataset in enumerate(datasets):
        datasets[idx] = dataset.batch(3, drop_remainder=True).map(tf.reduce_sum)

    interleaved_dataset = tf.data.Dataset.choose_from_datasets(
        datasets, tf.data.Dataset.range(3).repeat()
    )

    # New counter, because the `IncreaseCounter` object has a state
    counter = IncreasesCounter()
    increases = counter(interleaved_dataset)
    tf.print("[part two] increases: ", increases)


if __name__ == "__main__":
    sys.exit(main())
