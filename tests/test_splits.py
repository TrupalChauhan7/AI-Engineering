"""Dev/test split: by consultation, seeded, deterministic, disjoint."""

from s2n.data.splits import load_or_create_split, make_split


def test_split_is_deterministic_and_disjoint():
    ids = [f"c{i:02d}" for i in range(57)]
    a = make_split(ids, dev_size=10, seed=42)
    b = make_split(ids, dev_size=10, seed=42)
    assert a.dev == b.dev  # same seed -> same split
    assert len(a.dev) == 10 and len(a.test) == 47
    assert set(a.dev).isdisjoint(a.test)
    assert set(a.dev) | set(a.test) == set(ids)


def test_different_seed_changes_split():
    ids = [f"c{i:02d}" for i in range(57)]
    assert make_split(ids, 10, 42).dev != make_split(ids, 10, 7).dev


def test_frozen_split_loads_same_each_time():
    s1 = load_or_create_split()
    s2 = load_or_create_split()
    assert s1.dev == s2.dev and s1.test == s2.test
    assert len(s1.dev) == 10 and len(s1.test) == 47
