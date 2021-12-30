#!/usr/bin/env python3

"""
https://adventofcode.com/2021/day/12
"""

import sys

import tensorflow as tf


def main():
    """Entrypoint. Suppose the "input" file is in the cwd."""

    connections = tf.data.TextLineDataset("input").map(
        lambda string: tf.strings.split(string, "-")
    )

    # Create a map between human readable node names and
    # numeric indices
    human_to_id = tf.lookup.experimental.MutableHashTable(tf.string, tf.int64, -1)
    id_to_human = tf.lookup.experimental.MutableHashTable(tf.int64, tf.string, "")

    idx = tf.Variable(0, dtype=tf.int64)
    indices = tf.TensorArray(tf.int64, size=0, dynamic_size=True)
    for edge in connections:
        node_i = human_to_id.lookup(edge[0])
        node_j = human_to_id.lookup(edge[1])

        if tf.equal(node_i, -1):
            human_to_id.insert([edge[0]], [idx])
            id_to_human.insert([idx], [edge[0]])
            node_i = tf.identity(idx)
            idx.assign_add(1)
        if tf.equal(node_j, -1):
            human_to_id.insert([edge[1]], [idx])
            id_to_human.insert([idx], [edge[1]])
            node_j = tf.identity(idx)
            idx.assign_add(1)

        ij = tf.convert_to_tensor([node_i, node_j])
        indices = indices.write(indices.size(), ij)

    indices = indices.stack()
    indices = tf.reshape(indices, (-1, 2))
    tf.print(indices, summarize=-1)
    A = tf.tensor_scatter_nd_update(
        tf.zeros((idx, idx), dtype=tf.int64),
        indices,
        tf.repeat(tf.cast(1, tf.int64), tf.shape(indices)[0]),
    )
    A = A + tf.transpose(A)
    tf.print(A)

    # Visit only once (per path search)
    keys = human_to_id.export()[0]
    visit_only_once_human = tf.gather(
        keys,
        tf.where(
            tf.equal(
                tf.range(tf.shape(keys)[0]),
                tf.cast(tf.strings.regex_full_match(keys, "[a-z]+?"), tf.int32)
                * tf.range(tf.shape(keys)[0]),
            )
        ),
    )
    visit_only_once_human = tf.squeeze(visit_only_once_human)
    visit_only_once_id = human_to_id.lookup(visit_only_once_human)

    # Visit multiple times = {keys} - {only once}
    visit_multiple_times_human = tf.sparse.to_dense(
        tf.sets.difference(
            tf.reshape(keys, (1, -1)), tf.reshape(visit_only_once_human, (1, -1))
        )
    )
    visit_multiple_times_human = tf.squeeze(visit_multiple_times_human)
    visit_multiple_times_id = human_to_id.lookup(visit_multiple_times_human)

    tf.print(visit_only_once_id, visit_only_once_human)
    tf.print(visit_multiple_times_id, visit_multiple_times_human)

    # Goal: go from start to end
    # Finding all the possible paths
    # I can use the adjiacency matrix for finding the neighbors of every node
    # extracting the neighbord coordinates to get their IDs
    # Check if I've been already in that neighbor the right amount of times

    start_id, end_id = human_to_id.lookup(["start", "end"])

    # Every neighbor is a possible new, distinct path, hence
    # every time we visit a new neighbor we should have a different state.

    # Note: the problem asks us to COUNT the number of possible paths
    count = tf.Variable(0, dtype=tf.int64)

    @tf.function
    def _neigh_ids(A, node_id):
        return tf.squeeze(tf.where(tf.equal(A[node_id, :], 1)))

    stack = []
    # @tf.function
    def _visit(A: tf.Tensor, node_id: tf.Tensor, path: tf.Tensor):

        current_path = tf.concat([path, [node_id]], axis=0)
        tf.print("current path: ", current_path)
        if tf.equal(node_id, end_id):
            return current_path

        neighs = _neigh_ids(A, node_id)
        neigh_shape = tf.shape(neighs)
        if tf.equal(tf.size(neighs), 0):
            tf.print("(tf.size(neighs) == 0)")
            return current_path

        if tf.equal(tf.size(neigh_shape), 0):
            neighs = tf.expand_dims(neighs, 0)
            neigh_shape = tf.shape(neighs)

        # TODO: return a list of all possible paths from here
        for idx in tf.range(neigh_shape[0]):
            neigh_id = neighs[idx]
            if tf.logical_and(
                tf.reduce_any(tf.equal(neigh_id, visit_only_once_id)),
                tf.reduce_any(tf.equal(neigh_id, current_path)),
            ):
                tf.print("Skipping ", neigh_id, " visit again")
                continue
            # Remove node_id from neigh neighborhood in A if it's
            # a lowercase node
            """
            if tf.reduce_any(tf.equal(node_id, visit_only_once_id)):
                n = tf.shape(A, tf.int64)[0]
                # rows
                indices = tf.stack(
                    [
                        tf.repeat(node_id, n),
                        tf.range(n),
                    ],
                    axis=1,
                )
                updates = tf.repeat(tf.cast(0, tf.int64), n)

                A = tf.tensor_scatter_nd_update(A, indices, updates)

                # cols
                indices = tf.stack(
                    [
                        tf.range(n),
                        tf.repeat(node_id, n),
                    ],
                    axis=1,
                )

                A = tf.tensor_scatter_nd_update(A, indices, updates)
                tf.print("cant visit ", node_id, " anymore")
            else:
                tf.print("CAN visit ", node_id, " again")
                # B = tf.identity(A)
            """

            path = _visit(A, neigh_id, current_path)

            if tf.equal(path[-1], end_id):
                tf.print("FOUND!", path)
                stack.append(path)
        return current_path

    # All the paths starts from start
    neighs = _neigh_ids(A, start_id)
    for idx in tf.range(tf.shape(neighs)[0]):
        neigh_id = neighs[idx]
        _visit(A, neigh_id, [start_id])

    print(stack, len(stack))
    sys.exit()


if __name__ == "__main__":
    sys.exit(main())
