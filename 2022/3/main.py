import string
import sys
from pathlib import Path

import tensorflow as tf


def main(input_path: Path) -> int:
    dataset = tf.data.TextLineDataset(input_path.as_posix())

    keys_tensor = tf.concat(
        [
            tf.strings.bytes_split(string.ascii_lowercase),
            tf.strings.bytes_split(string.ascii_uppercase),
        ],
        axis=0,
    )
    vals_tensor = tf.concat([tf.range(1, 27), tf.range(27, 53)], axis=0)

    item_priority_lut = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), default_value=-1
    )

    @tf.function
    def split(line):
        length = tf.strings.length(line) // 2
        position = length

        return tf.strings.substr(line, pos=0, len=length), tf.strings.substr(
            line, pos=position, len=length
        )

    splitted_dataset = dataset.map(split)

    @tf.function
    def to_priority(first, second):
        first = tf.strings.bytes_split(first)
        second = tf.strings.bytes_split(second)
        return item_priority_lut.lookup(first), item_priority_lut.lookup(second)

    splitted_priority_dataset = splitted_dataset.map(to_priority)

    @tf.function
    def to_common(first, second):
        first = tf.expand_dims(first, axis=0)
        second = tf.expand_dims(second, axis=0)
        intersection = tf.sets.intersection(first, second)
        return tf.squeeze(tf.sparse.to_dense(intersection))

    common_elements = splitted_priority_dataset.map(to_common)
    tensor = tf.convert_to_tensor(list(common_elements.as_numpy_iterator()))
    tf.print("sum of priorities of common elements: ", tf.reduce_sum(tensor))

    grouped_dataset = dataset.batch(3)
    grouped_priority_dataset = grouped_dataset.map(
        lambda line: item_priority_lut.lookup(tf.strings.bytes_split(line))
    )

    @tf.function
    def to_common_in_batch(batch):
        a, b, c = batch[0], batch[1], batch[2]
        aIb = tf.sets.intersection(tf.expand_dims(a, axis=0), tf.expand_dims(b, axis=0))
        intersection = tf.sets.intersection(aIb, tf.expand_dims(c, axis=0))
        return tf.squeeze(tf.sparse.to_dense(intersection))

    grouped_common_elements = grouped_priority_dataset.map(to_common_in_batch)
    tensor = tf.convert_to_tensor(list(grouped_common_elements.as_numpy_iterator()))
    tf.print("sum of priorities of grouped by 3 elements: ", tf.reduce_sum(tensor))

    return 0


if __name__ == "__main__":
    INPUT: Path = Path(sys.argv[1] if len(sys.argv) > 1 else "fake")
    sys.exit(main(INPUT))
