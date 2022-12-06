"""
Solution in pure TensorFlow to the puzzle

https://adventofcode.com/2022/day/5

of the Advent of Code 2022.
"""

import sys
from pathlib import Path

import tensorflow as tf


def main(input_path: Path) -> int:
    """entrypoint"""

    dataset = tf.data.TextLineDataset(input_path.as_posix())

    @tf.function
    def to_array(line):
        length = tf.strings.length(line) + 1
        stacks = length // 4
        ta = tf.TensorArray(tf.string, size=0, dynamic_size=True)
        for i in tf.range(stacks):
            substr = tf.strings.strip(tf.strings.substr(line, i * 4, 4))
            stripped = tf.strings.regex_replace(substr, "\[|\]", "")
            ta = ta.write(i, stripped)

        return ta.stack()

    stacks_dataset = dataset.filter(
        lambda line: tf.strings.regex_full_match(line, r".*\[.*")
    ).map(to_array)

    stacks_tensor = tf.convert_to_tensor(list(stacks_dataset))

    tf.print(stacks_tensor)

    num_stacks = tf.shape(stacks_tensor, tf.int64)[0]

    stacks = tf.Variable(
        stacks_tensor, validate_shape=False, dtype=tf.string, shape=tf.TensorShape(None)
    )
    tf.print(stacks)

    shape = tf.shape(stacks)
    # stack = stacks.assign(tf.transpose(stacks, (1, 0)))
    stack = stacks.assign(
        tf.reshape(
            stacks,
            [
                shape[0],
                shape[1],
                1,
            ],
        )
    )

    moves_dataset = dataset.skip(tf.shape(stacks_tensor, tf.int64)[0] + 2)

    tops = tf.lookup.experimental.MutableHashTable(tf.int64, tf.int64, default_value=-1)

    def update_tops():
        tops.insert(
            tf.range(num_stacks),
            tf.squeeze(
                tf.reduce_sum(tf.cast(tf.not_equal(stacks, ""), tf.int64), axis=[0])
            ),
        )

    update_tops()
    tf.print("tops: ", tops.export())

    # move 1 from 2 to 1
    def move(line):
        amount = tf.strings.to_number(
            tf.strings.regex_replace(
                tf.strings.regex_replace(line, "move ", ""), " from \d* to \d*$", ""
            ),
            tf.int64,
        )

        source_dest = tf.strings.regex_replace(line, "move \d* from ", "")
        source = (
            tf.strings.to_number(
                tf.strings.regex_replace(source_dest, " to \d*$", ""), tf.int64
            )
            - 1
        )

        dest = (
            tf.strings.to_number(
                tf.strings.regex_replace(source_dest, "\d* to ", ""), tf.int64
            )
            - 1
        )

        tf.print(line)
        tf.print("s: ", source, " d: ", dest)
        read = stacks[:amount, source]
        tf.print("read: ", read)

        # remove from source
        # TODO

        top = tops[dest]
        tf.print("top in dest: ", top)
        # insert in dest
        # TODO

        # stacks.assign(
        #    tf.tensor_scatter_nd_update(
        #        stacks, [tf.range(top, top + amount)], tf.reshape(read, -1)
        #    )
        # )
        tf.print(stacks)

        # stacks[dest].write(stacks[dest].size(), stacks[source].

        return line

    moves_dataset = moves_dataset.map(move)
    tf.print(next(moves_dataset.take(1).as_numpy_iterator()))

    return 0


if __name__ == "__main__":
    INPUT: Path = Path(sys.argv[1] if len(sys.argv) > 1 else "fake")
    sys.exit(main(INPUT))
