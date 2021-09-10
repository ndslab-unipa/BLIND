import pytest
import numpy.testing as npt
import sys
from pathlib import Path
from configobj import ConfigObj
from typing import List
import numpy as np

# Add blind directory to sys.path
sys.path.append((Path().cwd()).as_posix())
from blind.unencrypted_blind import UnUser, UnServiceProvider

# Initialize random number generator
test_conf = ConfigObj((Path() / 'blind' / 'test' / 'test_config.ini').as_posix(), unrepr=True)
rng = np.random.default_rng(test_conf['random_seed'])


# Fixtures #
@pytest.fixture(scope="module")
def conf() -> ConfigObj:
    return ConfigObj('base_config.ini', unrepr=True)

@pytest.fixture(scope="module")
def sp(conf) -> ConfigObj:
    return UnServiceProvider([], conf)

@pytest.fixture(scope="module")
def users_example(conf) -> List[UnUser]:
    # example data from https://www.datasciencecentral.com/profiles/blogs/
    # /steps-to-calculate-centroids-in-cluster-using-k-means-clustering
    all_values = [[2, 0], [1, 3], [3, 5], [2, 2], [4, 6]]
    return [UnUser(conf, values=values) for values in all_values]

@pytest.fixture(scope="module")
def users_example_outlier(conf) -> List[UnUser]:
    all_values = [[2, 0], [1, 3], [3, 5], [2, 2], [4, 6], [12, 15]]
    return [UnUser(conf, values=values) for values in all_values]

@pytest.fixture(scope="function")
def sp_example(users_example, conf) -> UnServiceProvider:
    sp = UnServiceProvider(users_example, conf)

    # starting centroids
    sp.centroids = [[1, 3], [2, 2]]
    return sp

@pytest.fixture(scope="function")
def sp_example_outlier(users_example_outlier, conf) -> UnServiceProvider:
    sp = UnServiceProvider(users_example_outlier, conf)

    # starting centroids
    sp.centroids = [[1, 3], [2, 2]]
    return sp

# UnServiceProvider tests #
def test_initial_centroids(sp_example):
    # given
    sp = sp_example

    # when
    centroids = sp.centroids

    # then
    expected = [[1, 3], [2, 2]]
    npt.assert_allclose(centroids, expected)

def test_calc_distance_squared(sp: UnServiceProvider):
    # example from http://rosalind.info/glossary/euclidean-distance/

    # given
    x = [-1, 2, 3]
    y = [4, 0, -3]

    # when
    result = sp.calc_distance_squared(x, y)

    # then
    expected = 65
    npt.assert_almost_equal(result, expected)

def test_calc_distance_squared_different_len_expect_fail(sp: UnServiceProvider):
    # given
    x = rng.uniform(-1000, 1000, size=10)
    y = rng.uniform(-1000, 1000, size=8)

    # when
    # then
    with pytest.raises(ValueError):
        result = sp.calc_distance_squared(x, y)

def test_calc_distance_squared_random(sp: UnServiceProvider):
    # given
    num_tasks = sp.conf['num_tasks']
    x = rng.uniform(-1000, 1000, size=num_tasks)
    y = rng.uniform(-1000, 1000, size=num_tasks)

    # when
    result = sp.calc_distance_squared(x, y)

    # then
    # use numpy to calculate Euclidean distance
    expected = np.linalg.norm(x - y) ** 2
    npt.assert_almost_equal(result, expected)

# Update centroids tests
def test_update_centroids(sp_example: UnServiceProvider):
    # given
    # example data

    # when
    sp_example.update_centroids()

    # then
    expected = [[8/3, 14/3], [2.0, 1.0]]
    npt.assert_allclose(sp_example.centroids, expected)

def test_update_centroids_2_steps(sp_example: UnServiceProvider):
    # given
    # example data

    # when
    sp_example.update_centroids()
    sp_example.update_centroids()

    # then
    expected = [[3.5, 5.5], [5/3, 5/3]]
    npt.assert_allclose(sp_example.centroids, expected)

# Test for cluster with zero users
def test_update_centroids_with_zero_users(sp_example: UnServiceProvider):
    # given
    # example data
    sp_example.centroids = [[1, 3], [2, 2], [99, 99]]

    # when
    sp_example.update_centroids()

    # then
    expected = [[8/3, 14/3], [2.0, 1.0], [99, 99]]
    npt.assert_allclose(sp_example.centroids, expected)

def test_run_kmeans_2_steps(sp_example: UnServiceProvider):
    # given
    # example data
    num_steps = 2

    # when
    sp_example.run_kmeans(num_steps)

    # then
    expected = [[3.5, 5.5], [5/3, 5/3]]
    npt.assert_allclose(sp_example.centroids, expected)

# k-means with outlier removal tests
def test_run_kmeans_outlier(sp_example_outlier: UnServiceProvider):
    # given
    # all_values = [[2, 0], [1, 3], [3, 5], [2, 2], [4, 6], [12, 15]]
    sp = sp_example_outlier
    num_steps = 2
    threshold = 2

    sp.centroids = [[2, 2]]

    # when
    sp.run_kmeans(num_steps)
    result = sp.calc_centroids_without_outliers(threshold=threshold)

    # then
    expected = [[3.5, 5.5]]

    # Results as expected when removing outliers
    npt.assert_allclose(result, expected)

    # Wrong results when NOT removing outliers
    try:
        npt.assert_allclose(sp.centroids, expected)
        raise RuntimeError
    except AssertionError:
        pass
