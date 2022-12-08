"""
Solution in pure TensorFlow to the puzzle

https://adventofcode.com/2022/day/7

of the Advent of Code 2022.
"""

import sys
from pathlib import Path

import tensorflow as tf


def main(input_path: Path) -> int:
    """entrypoint"""

    dataset = tf.data.TextLineDataset(input_path.as_posix())
    dataset = dataset.concatenate(tf.data.Dataset.from_tensors(tf.constant("$ cd /")))

    def func(old_state, line):
        # tf.print("line: ", line)
        is_command = tf.strings.regex_full_match(line, r"^\$.*")
        new_state = old_state
        if is_command:
            if tf.strings.regex_full_match(line, r"\$ cd .*"):
                dest = tf.strings.split([line], " ")[0][-1]
                if tf.equal(dest, "/"):
                    new_state = tf.constant("/ 0")
                else:
                    old_path = tf.strings.split([old_state], " ")[0][0]
                    new_state = tf.strings.join(
                        [tf.strings.join([old_path, dest], "/"), "0"], " "
                    )
        else:
            split = tf.strings.split([line], " ")[0]
            if tf.not_equal(split[0], "dir"):
                size = tf.strings.to_number(split[0], tf.int64)
                state_size = tf.strings.split([old_state], " ")[0]
                if tf.equal(tf.shape(state_size, tf.int64)[0], 1):
                    old_size = tf.constant(0, tf.int64)
                else:
                    old_size = tf.strings.to_number(state_size[1], tf.int64)

                partial_size = size + old_size
                new_state = tf.strings.join(
                    [
                        tf.strings.split(old_state, " ")[0],
                        tf.strings.as_string(partial_size),
                    ],
                    " ",
                )

        if tf.not_equal(new_state, old_state):
            # output_value = tf.strings.to_number(
            #    tf.strings.split([new_state], " ")[0][1], tf.int64
            # )
            output_value = new_state
        else:
            # output_value = tf.constant(-1, tf.int64)
            output_value = tf.constant("")

        return new_state, output_value

    initial_state = tf.constant("/ 0")
    # ['', '/ 14848514', '/ 23352670', ...., '//a/e/../../d 17719346', '//a/e/../../d 24933642', '/ 0']
    intermediate_dataset = dataset.scan(initial_state, func)

    filtered_dataset = intermediate_dataset.filter(
        lambda line: tf.strings.regex_full_match(line, "^.* \d*$")
    ).map(lambda line: tf.strings.regex_replace(line, r"\/\/", "/"))

    def gen(ds):
        def resolve():
            for pair in ds:
                path, count = tf.strings.split([pair], " ")[0]
                path = Path(path.numpy().decode("utf-8")).resolve().as_posix()
                yield path, count.numpy().decode("utf-8")

        return resolve

    filtered_dataset = tf.data.Dataset.from_generator(
        gen(filtered_dataset), tf.string, output_shapes=[2]
    )

    def mapper(old_state, pair):
        old_path = old_state[0]
        new_path = pair[0]
        output_value = tf.constant(["", ""])
        if tf.logical_or(
            tf.equal(old_path, "fake_path"), tf.equal(new_path, "fake_path")
        ):
            output_value = tf.constant(["", ""])
        elif tf.not_equal(old_path, new_path):
            output_value = old_state

        return pair, output_value

    initial_state = tf.constant(["fake_path", "-1"])
    filtered_dataset = (
        filtered_dataset.concatenate(tf.data.Dataset.from_tensors(initial_state))
        .scan(initial_state, mapper)
        .filter(
            lambda pair: tf.logical_and(
                tf.greater(tf.strings.length(pair[0]), 0), tf.not_equal(pair[1], "0")
            )
        )
    )
    x = list(filtered_dataset.as_numpy_iterator())
    print(x)

    lut = tf.lookup.experimental.MutableHashTable(tf.string, tf.int64, default_value=0)
    for pair in filtered_dataset:
        path, value = pair[0], tf.strings.to_number(pair[1], tf.int64)
        parts = tf.strings.split(path, "/")
        tf.print(parts)
        if tf.logical_and(tf.equal(parts[0], parts[1]), tf.equal(parts[0], "")):
            keys = ["/"]
            old = lut.lookup(keys)[0]
            new = old + value
            lut.insert(keys, [new])
        else:
            for idx, part in enumerate(parts):
                if tf.equal(part, ""):
                    keys = ["/"]
                else:
                    tf.print("parts: ", parts[1 : idx + 1])
                    l = [tf.constant("")] + parts[1 : idx + 1]
                    tf.print(l)
                    j = tf.strings.join(l, "/")
                    tf.print("j:", j)
                    keys = [j]
                # tf.print("k: ", keys)
                old = lut.lookup(keys)[0]
                # tf.print("old: ", old)
                new = old + value
                # tf.print("new: ", new)
                lut.insert(keys, [new])

    paths, sizes = lut.export()
    print(paths, sizes)
    tf.print(
        "part 1: ",
        tf.reduce_sum(tf.gather(sizes, tf.where(tf.math.less_equal(sizes, 100000)))),
    )

    update_size = 30000000
    free_space = 70000000 - lut.lookup("/")
    required_space = update_size - free_space
    tf.print(required_space)

    big_enough = tf.gather(
        sizes, tf.where(tf.math.greater_equal(sizes - required_space, 0))
    )
    tf.print("part 2: ", tf.gather(big_enough, tf.math.argmin(big_enough, axis=0)))
    return 0


if __name__ == "__main__":
    INPUT: Path = Path(sys.argv[1] if len(sys.argv) > 1 else "fake")
    sys.exit(main(INPUT))
