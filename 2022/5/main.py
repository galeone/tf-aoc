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
    num_stacks = tf.shape(stacks_tensor, tf.int64)[1] + 1

    moves_dataset = dataset.skip(tf.shape(stacks_tensor, tf.int64)[0] + 2)

    # stacks = tf.Variable(
    #    stacks_tensor, validate_shape=False, dtype=tf.string, shape=tf.TensorShape(None)
    # )

    max_stack_size = 200
    stacks = tf.Variable(tf.zeros((max_stack_size, num_stacks - 1, 1), dtype=tf.string))

    def initialize_stacks():
        tf.print(stacks_tensor, summarize=-1)
        tf.print(tf.shape(stacks_tensor))

        # shape = tf.shape(stacks)
        # stacks.assign(
        #    tf.reshape(
        #        stacks,
        #        [
        #            shape[0],
        #            shape[1],
        #            1,
        #        ],
        #    )
        # )

        indices_x, indices_y = tf.meshgrid(
            tf.range(max_stack_size - tf.shape(stacks_tensor)[0], max_stack_size),
            tf.range(tf.shape(stacks_tensor)[1]),
        )

        indices = tf.stack([indices_x, indices_y], axis=-1)

        updates = tf.expand_dims(tf.transpose(stacks_tensor), axis=2)
        stacks.assign(tf.tensor_scatter_nd_update(stacks, indices, updates))

    initialize_stacks()

    num_elements = tf.lookup.experimental.MutableHashTable(
        tf.int64, tf.int64, default_value=-1
    )

    def update_num_elements():
        num_elements.insert(
            tf.range(num_stacks - 1),
            tf.squeeze(
                tf.reduce_sum(tf.cast(tf.not_equal(stacks, ""), tf.int64), axis=[0])
            ),
        )

    update_num_elements()

    one_at_a_time = tf.Variable(True)

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
        num_element_source = num_elements.lookup([source])[0]
        top = max_stack_size - num_element_source
        # tf.print("amount: ", amount, " source: ", source, " top: ", top)

        read = stacks[top : top + amount, source]

        # remove from source
        # tf.print("removing from source stacks[:amount, source]: ", read)
        indices_x, indices_y = tf.meshgrid(tf.range(top, top + amount), [source])
        indices = tf.reshape(tf.stack([indices_x, indices_y], axis=-1), (-1, 2))
        updates = tf.reshape(tf.repeat("", amount), (-1, 1))

        stacks.assign(
            tf.tensor_scatter_nd_update(stacks, indices, updates), use_locking=True
        )

        num_element_dest = num_elements.lookup([dest])[0]
        # tf.print("num_element in dest: ", num_element_dest)
        # tf.print(stacks[num_element_dest, :])

        top = max_stack_size - num_element_dest - 1
        # tf.print("top: ", top)

        # one a at a time -> reverse
        if one_at_a_time:
            insert = tf.reverse(read, axis=[0])
            insert = tf.reshape(insert, (-1, 1))
        else:
            insert = tf.reshape(read, (-1, 1))

        tf.print("inserting in dest: ", insert)
        tf.print("inserting at pos: ", top - amount + 1, " - ", top + 1)

        indices_x, indices_y = tf.meshgrid(tf.range(top - amount + 1, top + 1), [dest])
        indices = tf.reshape(tf.stack([indices_x, indices_y], axis=-1), (-1, 2))

        stacks.assign(
            tf.tensor_scatter_nd_update(stacks, indices, insert), use_locking=True
        )

        update_num_elements()
        # tf.print(tf.squeeze(stacks), summarize=-1)
        return stacks

    """
    tf.print("part 1")
    play = moves_dataset.map(move)

    list(play)

    indices_x = tf.range(num_stacks - 1)
    indices_y = max_stack_size - tf.reverse(num_elements.export()[1], axis=[0])

    indices = tf.reshape(tf.stack([indices_y, indices_x], axis=-1), (-1, 2))

    tf.print(tf.strings.join(tf.squeeze(tf.gather_nd(stacks, indices)), ""))
    """

    tf.print("part 2")
    initialize_stacks()
    update_num_elements()
    one_at_a_time.assign(False)
    play = moves_dataset.map(move)
    list(play)

    indices_x = tf.range(num_stacks - 1)
    indices_y = max_stack_size - tf.reverse(num_elements.export()[1], axis=[0])

    indices = tf.reshape(tf.stack([indices_y, indices_x], axis=-1), (-1, 2))

    tf.print(tf.strings.join(tf.squeeze(tf.gather_nd(stacks, indices)), ""))

    return 0


if __name__ == "__main__":
    INPUT: Path = Path(sys.argv[1] if len(sys.argv) > 1 else "fake")
    sys.exit(main(INPUT))
