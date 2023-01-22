import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import Conv2D


def main(input_path: Path):
    # N is the size of the board for simulation and L is number of knots in the rope
    N, L = 512, 10
    # Common setting of the layers.
    common_kw = dict(padding="same", kernel_initializer=tf.keras.initializers.Zeros())
    # Define the architecture of the model. Will fill the weight later.
    model = tf.keras.models.Sequential()
    # The input has 1+L channels of size NxN. The first channel is used for storing visited/unvisited information.
    # The rest L channels are used for representing the position of each knot.
    # The first convolution layer simulates head movement
    model.add(
        Conv2D(
            filters=1 + L,
            kernel_size=3,
            input_shape=(N, N, 1 + L),
            use_bias=False,
            **common_kw
        )
    )
    # For each non-head knot, use the following two layers to simulate the movement of the knot.
    for i in range(1, L):
        # the additional 9 channels corresponds 9 different patterns for non-head knot movement.
        model.add(
            Conv2D(
                filters=1 + L + 9,
                kernel_size=3,
                activation="relu",
                bias_initializer=tf.keras.initializers.Constant(-1),
                **common_kw
            )
        )
        # This convolution layer collects the 9 different situations and update the knot into the new position.
        model.add(
            Conv2D(
                filters=1 + L,
                kernel_size=1,
                activation="relu",
                use_bias=False,
                **common_kw
            )
        )

    # %% Fill weights in the layers
    # For the head knot, the construction is straightforward. For example,
    # a 3x3 convolution kernel `[[0 0 0] [1 0 0] [0 0 0]]` can move the knot one step right.
    # We will rotate the kernel on the fly when we want to move the head to another direction.
    # This looks a bit like cheating. We choose this approach so that we can focus on explaining the main idea.
    # It is not difficult to construct a conditional network to avoid switching the kernel for different directions.
    head_move = model.layers[0]
    (head_W,) = head_move.get_weights()
    for i in range(1 + L):
        head_W[1, 1, i, i] = 1  # copy all
    head_W[:, :, 1, 1] = 0
    head_W[1, 0, 1, 1] = 1
    head_WT = head_W.transpose((3, 0, 1, 2))  # prepare for rotation

    # The following layers are for non-head knots movement
    for i in range(1, L):
        # there are a 'move layer' and a 'collect layer' for each knot.
        move, collect = model.layers[i * 2 - 1 : i * 2 + 1]
        W, b = move.get_weights()
        # First 1+L channels are unmodified.
        for t in range(1 + L):
            W[1, 1, t, t] = 2  # copy all, note that b=-1, so 1=2*1-1 unchanged.
        # The new position of knot j=i+1 depends on the current position of knot i, and knot j.
        j = i + 1  # knot j follows knot i
        # If knot i is adjacent to knot j(taxicab distance<=1), then knot j stays the same position.
        W[:, :, i, 1 + L] = W[1, 1, j, 1 + L] = 1
        # the following kernels will match patterns like
        # X X X
        # _ _ _
        # _ i _
        # where knot j is at one of the X position and knot j is expected to moved to the center position.
        for n, k in enumerate([0, 2]):
            W[:, k, j, 1 + L + 1 + n] = W[1, 2 - k, i, 1 + L + 1 + n] = 1
            W[k, :, j, 1 + L + 3 + n] = W[2 - k, 1, i, 1 + L + 3 + n] = 1
        # the following kernels match the patterns like
        # j _ _
        # _ _ _
        # _ _ i
        # knot j is expected to moved to the center position.
        for n, (y, x) in enumerate(zip([0, 0, 2, 2], [0, 2, 0, 2])):
            W[y, x, j, 1 + L + 5 + n] = W[2 - y, 2 - x, i, 1 + L + 5 + n] = 1
        move.set_weights([W, b])
        # The collect layer collect the results matched by above patterns
        (W,) = collect.get_weights()
        # Copy the first 1+L channels, except channel j for knot j.
        for t in range(1 + L):
            W[..., t, t] = 1  # copy
        W[..., j, j] = 0
        # For channel j, sum up the last 9 channels. There will be exactly one position has value 1, and rest of the position are all 0.
        W[..., 1 + L :, j] = 1  # collect moves
        collect.set_weights([W])
    # For the last layer, also collect the position of the tail. 0 represents and 1 represent unvisited.
    # Because the non-linear function is relu, it will clip the negative values into 0.
    W[..., 1 + L :, 0] = -1  # collect unvisited
    collect.set_weights([W])

    # %% run the simulation
    state = tf.zeros((1, N, N, 1 + L), dtype=tf.float32).numpy()
    # Starts with  every knot at the center position
    state[0, N // 2, N // 2, :] = 1
    # Every position is marked as unvisited.
    state[..., 0] = 1 - state[..., 0]

    for n, line in enumerate(open(input_path.as_posix()).read().splitlines()):
        tf.print(n, line)
        direction, num = line.split(" ")
        # Rotate the kernel of the first layer according to the direction.
        angle = {"R": 0, "U": 1, "L": 2, "D": 3}[direction]
        head_move.set_weights(
            [tf.transpose(tf.image.rot90(head_WT, angle), (1, 2, 3, 0))]
        )
        # Simulate the movement num times
        for i in range(int(num)):
            state = model(state)
    # Count visited positions.
    print(state[..., 0])
    print("Ans:", int(tf.reduce_sum(1 - state[..., 0])))
    return 0


if __name__ == "__main__":
    INPUT: Path = Path(sys.argv[1] if len(sys.argv) > 1 else "fake")
    sys.exit(main(INPUT))
