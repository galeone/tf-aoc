"""
Solution in pure TensorFlow to the puzzle

https://adventofcode.com/2022/day/1

of the Advent of Code 2022.
"""

import sys
from pathlib import Path

import tensorflow as tf


def main(input_path: Path) -> int:
    """entrypoint"""

    dataset = tf.data.TextLineDataset(input_path.as_posix())
    dataset = dataset.concatenate(tf.data.Dataset.from_tensors([""]))
    initial_state = tf.constant(0, dtype=tf.int64)

    @tf.function
    def scan_func(state, line):
        if tf.strings.length(line) > 0:
            new_state = state + tf.strings.to_number(line, tf.int64)
            output_element = tf.constant(-1, tf.int64)
        else:
            new_state = tf.constant(0, tf.int64)
            output_element = state
        return new_state, output_element

    dataset = dataset.scan(initial_state, scan_func)
    dataset = dataset.filter(lambda x: x > 0)
    tensor = tf.convert_to_tensor(list(dataset.as_numpy_iterator()))

    max_calories = tf.reduce_max(tensor)
    elf_id = tf.argmax(tensor) + 1
    tf.print("## top elf ##")
    tf.print("max calories: ", max_calories)
    tf.print("elf id: ", elf_id)

    tf.print("## top 3 elves ##")
    top_calories, top_indices = tf.math.top_k(tensor, k=3)
    tf.print("calories: ", top_calories)
    tf.print("indices: ", top_indices + 1)
    tf.print("sum top calories: ", tf.reduce_sum(top_calories))
    return 0


if __name__ == "__main__":
    INPUT: Path = Path(sys.argv[1] if len(sys.argv) > 1 else "fake")
    sys.exit(main(INPUT))
