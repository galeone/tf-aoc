"""
Solution in pure TensorFlow to the puzzle

https://adventofcode.com/2022/day/11

of the Advent of Code 2022.
"""

import sys
from pathlib import Path

import tensorflow as tf


def main(input_path: Path) -> int:
    """entrypoint"""

    dataset = tf.data.TextLineDataset(input_path.as_posix())
    dataset = dataset.concatenate(tf.data.Dataset.from_tensors([""]))

    monkey = tf.Variable(["", "", "", "", "", ""], dtype=tf.string)
    monkey_id = tf.Variable(-1)
    pos = tf.Variable(0)

    initial_state = tf.constant(["", "", "", "", "", ""])

    def init(old_state, line):

        if tf.equal(line, ""):
            monkey.assign(old_state, use_locking=True)
            pos.assign(0)
            return initial_state, True

        if tf.strings.regex_full_match(line, r"^Monkey \d*:$"):
            items = tf.strings.split(tf.strings.split([line], " ")[0][1], ":")[0]
            updates = [items]
        elif tf.equal(pos, 1):
            items = tf.strings.strip(tf.strings.split([line], ":")[0][1])
            updates = [items]
        elif tf.equal(pos, 2):
            op = tf.strings.strip(tf.strings.split([line], "="))[0][1]
            updates = [op]
        elif tf.equal(pos, 3):
            divisible_by = tf.strings.strip(tf.strings.split([line], " "))[0][-1]
            updates = [divisible_by]
        else:  # if tf.reduce_any([tf.equal(pos, 4), tf.equal(pos, 5)]):
            monkey_dest = tf.strings.strip(tf.strings.split([line], " "))[0][-1]
            updates = [monkey_dest]

        indices = tf.reshape(pos, (1, 1))
        new_state = tf.tensor_scatter_nd_update(old_state, indices, updates)
        pos.assign_add(1)

        return new_state, False

    dataset = dataset.scan(initial_state, init)

    monkey_count = tf.Variable(0)
    for monkey_ready in dataset:
        if monkey_ready:
            tf.print(monkey)
            monkey.assign(tf.zeros_like(monkey))
            monkey_count.assign_add(1)

    inspected_count = tf.Variable(tf.zeros((monkey_count), tf.int32))

    @tf.function
    def apply_operation(worry_level, op):
        op = tf.strings.split([op], " ")[0]  # lhs, op, rhs
        ret = 0
        # lhs always = "old"
        if tf.strings.regex_full_match(op[2], r"^\d*$"):
            val = tf.strings.to_number(op[2], tf.int32)
        else:
            val = worry_level
        if tf.equal(op[1], "+"):
            ret = worry_level + val
        if tf.equal(op[1], "*"):
            ret = worry_level * val

        return ret

    @tf.function
    def monkey_play(rounds):
        items = tf.TensorArray(tf.int32, size=1, dynamic_size=True)
        operation = tf.TensorArray(tf.string, size=1, dynamic_size=True)
        divisible_test = tf.TensorArray(tf.int32, size=1, dynamic_size=True)
        throw_if_true = tf.TensorArray(tf.int32, size=1, dynamic_size=True)
        throw_if_false = tf.TensorArray(tf.int32, size=1, dynamic_size=True)

        for monkey_ready in dataset:
            if monkey_ready:
                idx = tf.strings.to_number(monkey[0], tf.int32)
                items = items.write(
                    idx,
                    tf.strings.to_number(tf.strings.split(monkey[1], ","), tf.int32),
                )
                operation = operation.write(idx, monkey[2])
                divisible_test = divisible_test.write(
                    idx, tf.strings.to_number(monkey[3], tf.int32)
                )
                throw_if_true = throw_if_true.write(
                    idx, tf.strings.to_number(monkey[4], tf.int32)
                )
                throw_if_false = throw_if_false.write(
                    idx, tf.strings.to_number(monkey[5], tf.int32)
                )

        for r in tf.range(rounds):
            # Now items contains all the starting items for every monkey
            # Let's play
            for m in tf.range(monkey_count):
                m_items = items.read(m)
                op = operation.read(m)
                test = divisible_test.read(m)

                tf.print("Monkey ", m, ":")
                for i in tf.range(tf.shape(m_items)[0]):
                    tf.print(
                        " Monkey inspects an item with a worry level of ", m_items[i]
                    )
                    worry_level = apply_operation(m_items[i], op)
                    tf.print(
                        "  Worry level is processed accoring to: ",
                        op,
                        " becoming: ",
                        worry_level,
                    )
                    worry_level //= 3
                    tf.print(
                        "  Monkey gets bored with item. Worry level is divided by 3 to ",
                        worry_level,
                    )

                    if tf.equal(tf.math.mod(worry_level, test), 0):
                        dest = throw_if_true.read(m)
                    else:
                        dest = throw_if_false.read(m)

                    tf.print("dest items before: ", items.read(dest))

                    items = items.write(
                        dest,
                        tf.concat(
                            [items.read(dest), tf.expand_dims(worry_level, axis=0)],
                            axis=0,
                        ),
                    )
                    tf.print("dest items: ", items.read(dest))

                    update = tf.tensor_scatter_nd_add(inspected_count, [[m]], [1])
                    inspected_count.assign(update)

                items = items.write(m, [])

        tf.print("after: ", items.concat(), summarize=-1)

    monkey_play(20)

    top_values, _ = tf.math.top_k(inspected_count, k=2)

    monkey_business = tf.reduce_prod(top_values)

    tf.print("Part 1: ", monkey_business)

    return 0


if __name__ == "__main__":
    INPUT: Path = Path(sys.argv[1] if len(sys.argv) > 1 else "fake")
    sys.exit(main(INPUT))
