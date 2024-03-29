"""
Solution in pure TensorFlow to the puzzle

https://adventofcode.com/2022/day/2

of the Advent of Code 2022.
"""

import sys
from pathlib import Path

import tensorflow as tf


def main(input_path: Path) -> int:
    """entrypoint"""

    dataset = tf.data.TextLineDataset(input_path.as_posix())

    keys_tensor = tf.constant(["X", "Y", "Z"])
    vals_tensor = tf.constant([1, 2, 3])

    action_to_score = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
        default_value=-1,
    )

    opponent_action = dataset.map(lambda line: tf.strings.split(line, " "))

    def play(opponent_action):
        opponent = opponent_action[0]
        action = opponent_action[1]
        outcome = 3
        my_action_score = action_to_score.lookup(action)
        if tf.equal(opponent, "A"):
            if tf.equal(action, "Y"):
                outcome = 6
            if tf.equal(action, "Z"):
                outcome = 0
        if tf.equal(opponent, "B"):
            if tf.equal(action, "X"):
                outcome = 0
            if tf.equal(action, "Z"):
                outcome = 6
        if tf.equal(opponent, "C"):
            if tf.equal(action, "X"):
                outcome = 6
            if tf.equal(action, "Y"):
                outcome = 0
        return outcome + my_action_score

    opponent_action_played = opponent_action.map(play)

    tf.print(
        "sum of scores according to strategy: ",
        tf.reduce_sum(
            tf.convert_to_tensor(list(opponent_action_played.as_numpy_iterator()))
        ),
    )

    outcome_to_score = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(["X", "Y", "Z"]), tf.constant([0, 3, 6])
        ),
        default_value=-1,
    )

    @tf.function
    def play_knowing_outcome(opponent_outcome):
        opponent = opponent_outcome[0]
        outcome = opponent_outcome[1]

        # draw
        my_action = tf.constant("Z")
        if tf.equal(outcome, "Y"):
            if tf.equal(opponent, "A"):
                my_action = tf.constant("X")
            if tf.equal(opponent, "B"):
                my_action = tf.constant("Y")
        # lose
        if tf.equal(outcome, "X"):
            if tf.equal(opponent, "A"):
                my_action = tf.constant("Z")
            if tf.equal(opponent, "B"):
                my_action = tf.constant("X")
            if tf.equal(opponent, "C"):
                my_action = tf.constant("Y")

        # win
        if tf.equal(outcome, "Z"):
            if tf.equal(opponent, "A"):
                my_action = tf.constant("Y")
            if tf.equal(opponent, "B"):
                my_action = tf.constant("Z")
            if tf.equal(opponent, "C"):
                my_action = tf.constant("X")

        return action_to_score.lookup(my_action) + outcome_to_score.lookup(outcome)

    opponent_outcome = opponent_action
    opponent_outcome_played = opponent_outcome.map(play_knowing_outcome)

    tf.print(
        "sum of scores according to new strategy: ",
        tf.reduce_sum(
            tf.convert_to_tensor(list(opponent_outcome_played.as_numpy_iterator()))
        ),
    )

    return 0


if __name__ == "__main__":
    INPUT: Path = Path(sys.argv[1] if len(sys.argv) > 1 else "fake")
    sys.exit(main(INPUT))
