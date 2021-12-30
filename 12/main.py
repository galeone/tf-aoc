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
    A = tf.tensor_scatter_nd_update(
        tf.zeros((idx, idx), dtype=tf.int64),
        indices,
        tf.repeat(tf.cast(1, tf.int64), tf.shape(indices)[0]),
    )
    A = A + tf.transpose(A)

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

    def _visit(A: tf.Tensor, node_id: tf.Tensor, path: tf.Tensor):
        current_path = tf.concat([path, [node_id]], axis=0)
        if tf.equal(node_id, end_id):
            count.assign_add(1)
            return current_path

        neighs = _neigh_ids(A, node_id)
        neigh_shape = tf.shape(neighs)
        if tf.equal(tf.size(neighs), 0):
            return current_path

        if tf.equal(tf.size(neigh_shape), 0):
            neighs = tf.expand_dims(neighs, 0)
            neigh_shape = tf.shape(neighs)

        for idx in tf.range(neigh_shape[0]):
            neigh_id = neighs[idx]
            if tf.logical_and(
                tf.reduce_any(tf.equal(neigh_id, visit_only_once_id)),
                tf.reduce_any(tf.equal(neigh_id, current_path)),
            ):
                continue
            _visit(A, neigh_id, current_path)
        return current_path

    # All the paths starts from start
    neighs = _neigh_ids(A, start_id)
    for idx in tf.range(tf.shape(neighs)[0]):
        neigh_id = neighs[idx]
        _visit(A, neigh_id, [start_id])

    tf.print("Part one: ", count)


if __name__ == "__main__":
    sys.exit(main())
