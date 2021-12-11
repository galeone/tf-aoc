#!/usr/bin/env python3

# pylint: disable=line-too-long
"""
--- Day 3: Binary Diagnostic ---
The submarine has been making some odd creaking noises, so you ask it to produce a diagnostic report just in case.

The diagnostic report (your puzzle input) consists of a list of binary numbers which, when decoded properly, can tell you many useful things about the conditions of the submarine. The first parameter to check is the power consumption.

You need to use the binary numbers in the diagnostic report to generate two new binary numbers (called the gamma rate and the epsilon rate). The power consumption can then be found by multiplying the gamma rate by the epsilon rate.

Each bit in the gamma rate can be determined by finding the most common bit in the corresponding position of all numbers in the diagnostic report. For example, given the following diagnostic report:

00100
11110
10110
10111
10101
01111
00111
11100
10000
11001
00010
01010
Considering only the first bit of each number, there are five 0 bits and seven 1 bits. Since the most common bit is 1, the first bit of the gamma rate is 1.

The most common second bit of the numbers in the diagnostic report is 0, so the second bit of the gamma rate is 0.

The most common value of the third, fourth, and fifth bits are 1, 1, and 0, respectively, and so the final three bits of the gamma rate are 110.

So, the gamma rate is the binary number 10110, or 22 in decimal.

The epsilon rate is calculated in a similar way; rather than use the most common bit, the least common bit from each position is used. So, the epsilon rate is 01001, or 9 in decimal. Multiplying the gamma rate (22) by the epsilon rate (9) produces the power consumption, 198.

Use the binary numbers in your diagnostic report to calculate the gamma rate and epsilon rate, then multiply them together. What is the power consumption of the submarine? (Be sure to represent your answer in decimal, not binary.)

--- Part Two ---
Next, you should verify the life support rating, which can be determined by multiplying the oxygen generator rating by the CO2 scrubber rating.

Both the oxygen generator rating and the CO2 scrubber rating are values that can be found in your diagnostic report - finding them is the tricky part. Both values are located using a similar process that involves filtering out values until only one remains. Before searching for either rating value, start with the full list of binary numbers from your diagnostic report and consider just the first bit of those numbers. Then:

Keep only numbers selected by the bit criteria for the type of rating value for which you are searching. Discard numbers which do not match the bit criteria.
If you only have one number left, stop; this is the rating value for which you are searching.
Otherwise, repeat the process, considering the next bit to the right.
The bit criteria depends on which type of rating value you want to find:

To find oxygen generator rating, determine the most common value (0 or 1) in the current bit position, and keep only numbers with that bit in that position. If 0 and 1 are equally common, keep values with a 1 in the position being considered.
To find CO2 scrubber rating, determine the least common value (0 or 1) in the current bit position, and keep only numbers with that bit in that position. If 0 and 1 are equally common, keep values with a 0 in the position being considered.
For example, to determine the oxygen generator rating value using the same example diagnostic report from above:

Start with all 12 numbers and consider only the first bit of each number. There are more 1 bits (7) than 0 bits (5), so keep only the 7 numbers with a 1 in the first position: 11110, 10110, 10111, 10101, 11100, 10000, and 11001.
Then, consider the second bit of the 7 remaining numbers: there are more 0 bits (4) than 1 bits (3), so keep only the 4 numbers with a 0 in the second position: 10110, 10111, 10101, and 10000.
In the third position, three of the four numbers have a 1, so keep those three: 10110, 10111, and 10101.
In the fourth position, two of the three numbers have a 1, so keep those two: 10110 and 10111.
In the fifth position, there are an equal number of 0 bits and 1 bits (one each). So, to find the oxygen generator rating, keep the number with a 1 in that position: 10111.
As there is only one number left, stop; the oxygen generator rating is 10111, or 23 in decimal.
Then, to determine the CO2 scrubber rating value from the same example above:

Start again with all 12 numbers and consider only the first bit of each number. There are fewer 0 bits (5) than 1 bits (7), so keep only the 5 numbers with a 0 in the first position: 00100, 01111, 00111, 00010, and 01010.
Then, consider the second bit of the 5 remaining numbers: there are fewer 1 bits (2) than 0 bits (3), so keep only the 2 numbers with a 1 in the second position: 01111 and 01010.
In the third position, there are an equal number of 0 bits and 1 bits (one each). So, to find the CO2 scrubber rating, keep the number with a 0 in that position: 01010.
As there is only one number left, stop; the CO2 scrubber rating is 01010, or 10 in decimal.
Finally, to find the life support rating, multiply the oxygen generator rating (23) by the CO2 scrubber rating (10) to get 230.

Use the binary numbers in your diagnostic report to calculate the oxygen generator rating and CO2 scrubber rating, then multiply them together. What is the life support rating of the submarine?
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
