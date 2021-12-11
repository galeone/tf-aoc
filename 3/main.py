#!/usr/bin/env python3

"""
https://adventofcode.com/2021/day/3
"""

import sys
from typing import Tuple

import tensorflow as tf


@tf.function
def most_frequent_bits(tensor: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    count = tf.reduce_sum(tensor, axis=0)
    tot = tf.cast(tf.shape(tensor)[0], tf.int64)
    half = tot // 2
    ret = tf.cast(tf.greater(count, half), tf.int64)
    return tf.squeeze(ret), tf.squeeze(
        tf.logical_and(tf.equal(count, half), tf.equal(tf.math.mod(tot, 2), 0))
    )  # True where #1 == #0


@tf.function
def bin2dec(bin_tensor: tf.Tensor):
    two = tf.cast(2, tf.int64)
    return tf.reduce_sum(
        tf.reverse(bin_tensor, axis=[0])
        * two ** tf.range(tf.size(bin_tensor), dtype=tf.int64)
    )


class RateFinder(tf.Module):
    def __init__(self, bits):
        super().__init__()
        # Constants
        self._zero = tf.constant(0, tf.int64)
        self._one = tf.constant(1, tf.int64)
        self._two = tf.constant(2, tf.int64)
        self._bits = tf.constant(tf.cast(bits, tf.int64))
        # Variables
        self._rating = tf.Variable(tf.zeros([bits], dtype=tf.int64), trainable=False)
        self._frequencies = tf.Variable(
            tf.zeros([bits], dtype=tf.int64), trainable=False
        )
        self._ta = tf.TensorArray(
            size=1, dtype=tf.int64, dynamic_size=True, clear_after_read=True
        )

    @tf.function(experimental_relax_shapes=True)
    def filter_by_bit_criteria(
        self,
        dataset_tensor: tf.Tensor,
        current_bit_position: tf.Tensor,
        oxigen: tf.Tensor,
    ):
        if oxigen:
            flag = self._one
            frequencies, mask = most_frequent_bits(dataset_tensor)
        else:
            flag = self._zero
            frequencies, mask = most_frequent_bits(dataset_tensor)
            frequencies = tf.cast(
                tf.logical_not(tf.cast(frequencies, tf.bool)),
                tf.int64,
            )
        # #0 == #1 pick the elements with the correct bitflag
        if mask[current_bit_position]:
            indices = tf.where(
                tf.equal(
                    dataset_tensor[:, current_bit_position],
                    flag,
                )
            )
        else:
            indices = tf.where(
                tf.equal(
                    dataset_tensor[:, current_bit_position],
                    frequencies[current_bit_position],
                )
            )

        # All elements with the bit "position" equal to frequencies[position]
        gathered = tf.gather_nd(dataset_tensor, indices)
        return gathered

    # @tf.function
    def find(self, dataset_tensor: tf.Tensor, oxigen: tf.Tensor):
        num_bits = tf.shape(dataset_tensor)[-1]
        self._ta.unstack(dataset_tensor)
        for current_bit_position in tf.range(num_bits):
            ta = self._ta.stack()
            gathered = tf.squeeze(
                self.filter_by_bit_criteria(ta, current_bit_position, oxigen)
            )
            if tf.equal(tf.size(gathered), num_bits):
                self._rating.assign(gathered)
                break
            self._ta.unstack(gathered)

        return self._rating


def main():
    """Entrypoint. Suppose the "input" file is in the cwd."""

    dataset = (
        tf.data.TextLineDataset("input")  # "0101"
        .map(tf.strings.bytes_split)  # '0', '1', '0', '1'
        .map(lambda digit: tf.strings.to_number(digit, out_type=tf.int64))  # 0 1 0 1
    )
    # We can do this in a raw way, treating the whole dataset as a tensor
    # so we can know its shape and extract the most frequent elements easily
    tensor_dataset = tf.convert_to_tensor(list(dataset))
    gamma_rate, _ = most_frequent_bits(tensor_dataset)
    tf.print("gamma rate (bin): ", gamma_rate)
    gamma_rate_dec = bin2dec(gamma_rate)
    tf.print("gamma rate (dec): ", gamma_rate_dec)

    # epsilon rate is the complement
    epsilon_rate = tf.cast(tf.logical_not(tf.cast(gamma_rate, tf.bool)), tf.int64)
    tf.print("epsilon rate (bin): ", epsilon_rate)
    epsilon_rate_dec = bin2dec(epsilon_rate)
    tf.print("epislon rate (dec): ", epsilon_rate_dec)

    power_consuption = gamma_rate_dec * epsilon_rate_dec
    tf.print("power consumption: ", power_consuption)

    # -- Part Two ---

    # gamma_rate contains the most frequent bit in each position 0 1 0 1 0 ...
    # starting from that, we can gather all the numbers that have the more common bit
    # in the "position".
    finder = RateFinder(bits=tf.size(epsilon_rate))

    oxigen_generator_rating = finder.find(tensor_dataset, True)
    tf.print("Oxigen generator rating (bin): ", oxigen_generator_rating)
    oxigen_generator_rating_dec = bin2dec(oxigen_generator_rating)
    tf.print("Oxigen generator rating (dec): ", oxigen_generator_rating_dec)

    co2_generator_rating = finder.find(tensor_dataset, False)
    tf.print("C02 scrubber rating (bin): ", co2_generator_rating)
    co2_generator_rating_dec = bin2dec(co2_generator_rating)
    tf.print("C02 scrubber rating (dec): ", co2_generator_rating_dec)

    tf.print(
        "life support rating = ", oxigen_generator_rating_dec * co2_generator_rating_dec
    )


if __name__ == "__main__":
    sys.exit(main())
