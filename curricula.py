"""Graph-size curricula shared by script_rl and build_val_set."""


def parse_graph_size(arg):
    """Parses comma-separated strings into tuples, or returns an int."""
    if ',' in arg:
        return tuple(map(int, arg.split(',')))
    return int(arg)


def stage_ranges(sizes):
    """Turns -n style sizes (ints or (start, stop) tuples) into ranges."""
    return [range(n, n + 1) if isinstance(n, int) else range(*n)
            for n in sizes]


curricula = {
    's2v': [
        (10, 11), (15, 21), (40, 51), (51, 101), (101, 201), (200, 301), (300, 401), (400, 501)
            ],
}
