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
    paths = []

    @tf.function
    def _neigh_ids(A, node_id):
        return tf.squeeze(tf.where(tf.equal(A[node_id, :], 1)))

    def _visit(A: tf.Tensor, node_id: tf.Tensor, path: tf.Tensor):
        current_path = tf.concat([path, [node_id]], axis=0)
        if tf.equal(node_id, end_id):
            paths.append(current_path)
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

    # for path in paths:
    #    tf.print(id_to_human.lookup(path), summarize=-1)

    count.assign(0)
    inner_count = tf.Variable(0)

    def _visit2(A: tf.Tensor, node_id: tf.Tensor, path: tf.Tensor):
        current_path = tf.concat([path, [node_id]], axis=0)

        # Skip start
        if tf.equal(node_id, start_id):
            return current_path

        # Success on end node
        if tf.equal(node_id, end_id):
            # paths.append(current_path)
            count.assign_add(1)
            return current_path

        # More than 2 lowercase visited twice
        visited, visited_idx, visited_count = tf.unique_with_counts(current_path)
        visited = tf.gather_nd(visited, tf.where(tf.greater(visited_count, 1)))
        inner_count.assign(0)
        for idx in tf.range(tf.shape(visited)[0]):
            if tf.reduce_any(tf.equal(visited[idx], visit_only_once_id)):
                inner_count.assign_add(1)

            if tf.greater_equal(inner_count, 2):
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

            # already visited twice and is lowcase
            if tf.logical_and(
                tf.reduce_any(tf.equal(neigh_id, visit_only_once_id)),
                tf.greater(
                    tf.reduce_sum(tf.cast(tf.equal(neigh_id, current_path), tf.int32)),
                    1,
                ),
            ):
                continue

            _visit2(A, neigh_id, current_path)

        return current_path

    neighs = _neigh_ids(A, start_id)
    for idx in tf.range(tf.shape(neighs)[0]):
        neigh_id = neighs[idx]
        _visit2(A, neigh_id, [start_id])

    # for path in paths:
    #    tf.print(id_to_human.lookup(path), summarize=-1)
    tf.print("Part two: ", count)


if __name__ == "__main__":
    sys.exit(main())
