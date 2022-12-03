#!/usr/bin/env python3

"""
https://adventofcode.com/2021/day/8
"""

import sys

import tensorflow as tf


def main():
    """Entrypoint. Suppose the "input" file is in the cwd."""

    dataset = (
        tf.data.TextLineDataset("input")
        .map(lambda line: tf.strings.split(line, " | "))
        .map(lambda lines: tf.strings.split(lines, " "))
    )
    count = tf.Variable(0, trainable=False, dtype=tf.int64)
    for _, output_digits in dataset:
        lengths = tf.strings.length(output_digits)
        one = tf.gather_nd(lengths, tf.where(tf.equal(lengths, 2)))
        four = tf.gather_nd(lengths, tf.where(tf.equal(lengths, 4)))
        seven = tf.gather_nd(lengths, tf.where(tf.equal(lengths, 3)))
        eight = tf.gather_nd(lengths, tf.where(tf.equal(lengths, 7)))
        count.assign_add(
            tf.cast(
                tf.reduce_sum(
                    [tf.size(one), tf.size(four), tf.size(seven), tf.size(eight)]
                ),
                tf.int64,
            )
        )
    tf.print("Part one: ", count)

    # -- Part 2 --
    count.assign(0)  # use count for sum

    def search_by_segments(digits_set, num_segments):
        lengths = tf.strings.length(digits_set)
        where = tf.where(tf.equal(lengths, num_segments))
        number = tf.gather_nd(lengths, where)
        num_found = tf.size(number)
        if tf.greater(num_found, 0):
            segments = tf.gather_nd(digits_set, where)[0]
        else:
            segments = tf.constant("")
        # segments (a,b,c), #num found, positions
        return tf.strings.bytes_split(segments), num_found, where

    for signal_patterns, output_digits in dataset:
        # reverse because we compute from units to decimals, ...
        output_digits = tf.reverse(output_digits, axis=[0])
        all_digits = tf.concat([output_digits, signal_patterns], axis=0)
        lengths = tf.strings.length(all_digits)

        # Num | segments | Dec  | # same
        # 0   | 6        | no   | 1 (9)
        # 1   | 2        | yes
        # 2   | 5        | no   | 2 (3, 5)
        # 3   | 5        | no   | 2 (2, 5)
        # 4   | 4        | yes
        # 5   | 5        | no   | 2 (2, 3)
        # 6   | 6        | no   | 1 (9)
        # 7   | 3        | yes
        # 8   | 7        | yes
        # 9   | 6        | no   | 1 (6)

        eight_chars, _, _ = search_by_segments(all_digits, 7)
        four_chars, _, _ = search_by_segments(all_digits, 4)
        seven_chars, _, _ = search_by_segments(all_digits, 3)
        one_chars, _, _ = search_by_segments(all_digits, 2)
        zero_chars = [""]
        two_chars = [""]
        three_chars = [""]
        five_chars = [""]
        six_chars = [""]
        nine_chars = [""]

        # All the 5 segments: 2, 3, 5
        five_segments = tf.strings.bytes_split(
            tf.gather_nd(all_digits, tf.where(tf.equal(lengths, 5)))
        )
        if tf.greater(tf.size(five_segments), 0):
            for candidate in five_segments:
                candidate_inter_seven = tf.sets.intersection(
                    tf.expand_dims(candidate, axis=0),
                    tf.expand_dims(seven_chars, axis=0),
                )
                candidate_inter_four = tf.sets.intersection(
                    tf.expand_dims(candidate, axis=0),
                    tf.expand_dims(four_chars, axis=0),
                )
                candidate_inter_one = tf.sets.intersection(
                    tf.expand_dims(candidate, axis=0),
                    tf.expand_dims(one_chars, axis=0),
                )
                # Use 7 as a reference:

                # A 2 has 2 in common with 7. I cannot identify it only with this
                # because also 2 has 2 in common with 7.

                # A 3 has 3 in common with 7. I can identify the 3 since 2 and 5 have only 2 in common.

                # 5 has 2 in common with 7. Cannot identify because of the 2.

                # Hence for identify a 2/5 I need a 7 and something else.
                # If I have a four:
                # A 2 has 2 in common with 7 and 2 in common with 4. Found!
                # A 5 has 2 in common with 7 and 3 in common with 4. Found!

                # If I have a one
                # A 2 has 2 in common with 7 and 1 in common with 1. Cannot identify.
                # A 5 has 2 in common with 7 and 1 in common with 1. Cannot identify.
                if tf.greater(tf.size(seven_chars), 0):
                    if tf.equal(tf.size(candidate_inter_seven), 3):
                        three_chars = candidate
                    elif tf.logical_and(
                        tf.greater(tf.size(four_chars), 0),
                        tf.equal(tf.size(candidate_inter_seven), 2),
                    ):
                        if tf.equal(tf.size(candidate_inter_four), 2):
                            two_chars = candidate
                        elif tf.equal(tf.size(candidate_inter_four), 3):
                            five_chars = candidate

                # use 4 as a reference

                # A 2 has 2 in common with 4. Found!
                # A 5 has 3 in common with 4.
                # A 3 has 3 in common with 4.

                # To find a 5,3 i need something else. Useless to check for seven, already done.
                # A 5 has 3 in common with 4 and 1 in common with 1. Found!
                # A 3 has 3 in common with 2 and 2 in common with 1. Found!

                if tf.greater(tf.size(four_chars), 0):
                    if tf.equal(tf.size(candidate_inter_four), 2):
                        two_chars = candidate
                    if tf.logical_and(
                        tf.equal(tf.size(candidate_inter_four), 3),
                        tf.greater(tf.size(one_chars), 0),
                    ):
                        if tf.equal(tf.size(candidate_inter_one), 1):
                            five_chars = candidate
                        else:
                            three_chars = candidate

        # All the 6 segments: 6, 9, 0
        six_segments = tf.strings.bytes_split(
            tf.gather_nd(all_digits, tf.where(tf.equal(lengths, 6)))
        )
        if tf.greater(tf.size(six_segments), 0):
            for candidate in six_segments:
                candidate_inter_seven = tf.sets.intersection(
                    tf.expand_dims(candidate, axis=0),
                    tf.expand_dims(seven_chars, axis=0),
                )
                candidate_inter_four = tf.sets.intersection(
                    tf.expand_dims(candidate, axis=0),
                    tf.expand_dims(four_chars, axis=0),
                )
                candidate_inter_one = tf.sets.intersection(
                    tf.expand_dims(candidate, axis=0),
                    tf.expand_dims(one_chars, axis=0),
                )
                # Use 7 as a reference:

                # A 9 has 3 in common with 7.
                # A 6 has 2 in common with 7. Found!
                # A 0 has 3 in common with 7.

                # A 9 has 3 in common with 7 and 4 in common with 4.
                # A 0 has 3 in common with 7 and 3 in common with 4.
                if tf.greater(tf.size(seven_chars), 0):
                    if tf.equal(tf.size(candidate_inter_seven), 2):
                        six_chars = candidate
                    elif tf.logical_and(
                        tf.equal(tf.size(candidate_inter_seven), 3),
                        tf.greater(tf.size(four_chars), 0),
                    ):
                        if tf.equal(tf.size(candidate_inter_four), 4):
                            nine_chars = candidate
                        else:
                            zero_chars = candidate
                # Use 4 as a reference:

                # A 9 has 4 in common with 4. Found!
                # A 6 has 3 in common with 4.
                # A 0 has 3 in common with 4.

                # A 6 has 3 in common with 4 and 1 in common with 1.
                # A 0 has 3 in common with 4 and 2 in common with 1.
                if tf.greater(tf.size(four_chars), 0):
                    if tf.equal(tf.size(candidate_inter_four), 4):
                        nine_chars = candidate
                    elif tf.logical_and(
                        tf.equal(tf.size(candidate_inter_one), 3),
                        tf.greater(tf.size(one_chars), 0),
                    ):
                        if tf.equal(tf.size(candidate_inter_four), 1):
                            six_chars = candidate
                        else:
                            zero_chars = candidate

                # Use 1 as a refrence

                # A 9 has 2 in common with 1.
                # A 6 has 1 in common with 1. Found!
                # A 0 has 2 in common with 1.
                if tf.greater(tf.size(one_chars), 0):
                    if tf.equal(tf.size(candidate_inter_one), 1):
                        six_chars = candidate

                # I can also use the other discovered digits

                # Use 2 as a reference
                # A 9 has 4 in common with 2.
                # A 6 has 4 in commom with 2.
                # A 0 has 4 in common with 2.

                # Use 3 as a reference
                # A 9 has 5 in common with 3. Found!
                # A 6 has 4 in common with 3.
                # A 0 has 4 in common with 3.
                if tf.greater(tf.size(three_chars), 0):
                    candidate_inter_three = tf.sets.intersection(
                        tf.expand_dims(candidate, axis=0),
                        tf.expand_dims(three_chars, axis=0),
                    )
                    if tf.equal(tf.size(candidate_inter_three), 5):
                        nine_chars = candidate

                # Use 5 as a reference
                # A 9 has 5 in common with 5.
                # A 6 has 5 in common with 5.
                # A 0 has 4 in common with 5. Found!
                if tf.greater(tf.size(five_chars), 0):
                    candidate_inter_five = tf.sets.intersection(
                        tf.expand_dims(candidate, axis=0),
                        tf.expand_dims(five_chars, axis=0),
                    )
                    if tf.equal(tf.size(candidate_inter_five), 4):
                        zero_chars = candidate

        for position, digit in enumerate(output_digits):
            digit = tf.strings.bytes_split(digit)
            for num, k in enumerate(
                [
                    zero_chars,
                    one_chars,
                    two_chars,
                    three_chars,
                    four_chars,
                    five_chars,
                    six_chars,
                    seven_chars,
                    eight_chars,
                    nine_chars,
                ]
            ):
                difference_1 = tf.sets.difference(
                    tf.expand_dims(digit, axis=0), tf.expand_dims(k, axis=0)
                )
                difference_2 = tf.sets.difference(
                    tf.expand_dims(k, axis=0), tf.expand_dims(digit, axis=0)
                )
                if tf.logical_and(
                    tf.equal(tf.size(difference_1), 0),
                    tf.equal(tf.size(difference_2), 0),
                ):
                    count.assign_add(num * 10 ** position)

    tf.print("Part two: ", count)


if __name__ == "__main__":
    sys.exit(main())
