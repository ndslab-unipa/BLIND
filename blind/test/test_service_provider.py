import pytest
import numpy.testing as npt
import sys
from pathlib import Path
from configobj import ConfigObj
from typing import List
import numpy as np

# Tests for the ServiceProvider

# Add blind directory to sys.path
sys.path.append((Path().cwd()).as_posix())
from blind.paillier_blind import User, KeyProvider, ServiceProvider
from blind.time_experiments import setup_experiment

# Initialize random number generator
test_conf = ConfigObj((Path() / 'blind' / 'test' / 'test_config.ini').as_posix(), unrepr=True)
rng = np.random.default_rng(test_conf['random_seed'])

# Fixtures #
@pytest.fixture(scope="module")
def conf() -> ConfigObj:
    return ConfigObj('base_config.ini', unrepr=True)

@pytest.fixture(scope="module")
def kp(conf) -> KeyProvider:
    return KeyProvider(conf)

@pytest.fixture(scope="module")
def users(kp, conf) -> List[User]:
    return [User(kp.pu, conf) for i in range(conf['num_users'])]

@pytest.fixture(scope="module")
def users_example(kp, conf) -> List[User]:
    # example data from https://www.datasciencecentral.com/profiles/blogs/
    # /steps-to-calculate-centroids-in-cluster-using-k-means-clustering
    all_values = [[2, 0], [1, 3], [3, 5], [2, 2], [4, 6]]
    return [User(kp.pu, conf, values=values) for values in all_values]

@pytest.fixture(scope="module")
def users_example_outlier(kp, conf) -> List[User]:
    all_values = [[2, 0], [1, 3], [3, 5], [2, 2], [4, 6], [12, 15]]
    return [User(kp.pu, conf, values=values) for values in all_values]

@pytest.fixture(scope="module")
def sp(kp, users, conf) -> ServiceProvider:
    return ServiceProvider(kp.pu, kp, users, conf)

@pytest.fixture(scope="function")
def sp_example(kp, users_example, conf) -> ServiceProvider:
    sp = ServiceProvider(kp.pu, kp, users_example, conf)

    # starting centroids
    sp.centroids = [[1, 3], [2, 2]]
    return sp

@pytest.fixture(scope="function")
def sp_example_outlier(kp, users_example_outlier, conf) -> ServiceProvider:
    sp = ServiceProvider(kp.pu, kp, users_example_outlier, conf)

    # starting centroids
    sp.centroids = [[1, 3], [2, 2]]
    return sp

# ServiceProvider tests #
def test_encrypt_decrypt_int(sp: ServiceProvider):
    # given
    value = 8

    # when
    encrypted = sp.encrypt_value(value)
    decrypted = sp.kp.decrypt_value(encrypted)

    # then
    npt.assert_almost_equal(decrypted, value)

def test_encrypt_decrypt_float(sp: ServiceProvider):
    # given
    value = 42.4276598

    # when
    encrypted = sp.encrypt_value(value)
    decrypted = sp.kp.decrypt_value(encrypted)

    # then
    npt.assert_almost_equal(decrypted, value)

def test_encrypt_decrypt_vector(sp: ServiceProvider):
    # given
    vector = rng.uniform(-1000, 1000, 10)

    # when
    encrypted = sp.encrypt_vector(vector)
    decrypted = sp.kp.decrypt_vector(encrypted)

    # then
    npt.assert_allclose(decrypted, vector)

def test_encrypted_sum(sp: ServiceProvider):
    # given
    x = 8
    y = 12

    # when
    x_enc = sp.encrypt_value(x)
    y_enc = sp.encrypt_value(y)
    result = x_enc + y_enc
    decrypted = sp.kp.decrypt_value(result)

    # then
    expected = x + y
    npt.assert_almost_equal(decrypted, expected)

def test_sum_multiple(sp: ServiceProvider):
    for i in range(10):
        # given
        x = rng.uniform(-1000, 1000)
        y = rng.uniform(-1000, 1000)
        
        # when
        x_enc = sp.encrypt_value(x)
        y_enc = sp.encrypt_value(y)
        result = x_enc + y_enc
        decrypted = sp.kp.decrypt_value(result)
        
        # then
        expected = x + y
        npt.assert_almost_equal(decrypted, expected)

def test_mult_cipher_plain(sp: ServiceProvider):
    # given
    x = 5
    y = 8.3

    # when
    x_enc = sp.encrypt_value(x)
    result = x_enc * y
    decrypted = sp.kp.decrypt_value(result)

    # then
    expected = x * y
    npt.assert_almost_equal(decrypted, expected)

def test_mult_cipher_cipher(sp: ServiceProvider):
    # given
    x = 5.126
    y = 8.3794

    # when
    x_enc = sp.encrypt_value(x)
    y_enc = sp.encrypt_value(y)
    result = sp.multiply(x_enc, y_enc)
    decrypted = sp.kp.decrypt_value(result)

    # then
    expected = x * y
    npt.assert_almost_equal(decrypted, expected)

def test_mult_cipher_0(sp: ServiceProvider):
    # given
    x = rng.uniform(-1000, 1000)
    y = 0

    # when
    x_enc = sp.encrypt_value(x)
    y_enc = sp.encrypt_value(y)
    result = sp.multiply(x_enc, y_enc)
    decrypted = sp.kp.decrypt_value(result)

    # then
    expected = 0
    npt.assert_almost_equal(decrypted, expected)

def test_calc_squared(sp: ServiceProvider):
    # given
    x = rng.uniform(-1000, 1000)

    # when
    x_enc = sp.encrypt_value(x)
    result = sp.calc_squared(x_enc)
    decrypted = sp.kp.decrypt_value(result)

    # then
    expected = x ** 2
    npt.assert_almost_equal(decrypted, expected)

def test_calc_0_squared(sp: ServiceProvider):
    # given
    x = 0

    # when
    x_enc = sp.encrypt_value(x)
    result = sp.calc_squared(x_enc)
    decrypted = sp.kp.decrypt_value(result)

    # then
    expected = 0
    npt.assert_almost_equal(decrypted, expected)

def test_calc_sqrt(sp: ServiceProvider):
    # given
    x = rng.uniform(0, 1000)

    # when
    x_enc = sp.encrypt_value(x)
    result = sp.calc_sqrt(x_enc)
    decrypted = sp.kp.decrypt_value(result)

    # then
    expected = x ** 0.5
    npt.assert_almost_equal(decrypted, expected)

def test_calc_sqrt_small_number(sp: ServiceProvider):
    # given
    x = rng.uniform(1e-10, 1e-6)

    # when
    x_enc = sp.encrypt_value(x)
    result = sp.calc_sqrt(x_enc)
    decrypted = sp.kp.decrypt_value(result)

    # then
    expected = x ** 0.5
    npt.assert_almost_equal(decrypted, expected)

def test_calc_sqrt_negative_expect_fail(sp: ServiceProvider):
    # given
    x = rng.uniform(-1000, -0.1)

    # when
    x_enc = sp.encrypt_value(x)

    # then
    with pytest.raises(TypeError):
        result = sp.calc_sqrt(x_enc)
        decrypted = sp.kp.decrypt_value(result)

def test_inverse(sp: ServiceProvider):
    # given
    x = rng.uniform(-1000, 1000)

    # when
    x_enc = sp.encrypt_value(x)
    result = sp.inverse(x_enc)
    decrypted = sp.kp.decrypt_value(result)

    # then
    expected = 1 / x
    npt.assert_almost_equal(decrypted, expected)

def test_inverse_0_expect_fail(sp: ServiceProvider):
    # given
    x = 0

    # when
    x_enc = sp.encrypt_value(x)

    # then
    with pytest.raises(ZeroDivisionError):
        result = sp.inverse(x_enc)


def test_divide(sp: ServiceProvider):
    # given
    x = rng.uniform(-1000, 1000)
    y = rng.uniform(-1000, 1000)

    # when
    x_enc = sp.encrypt_value(x)
    y_enc = sp.encrypt_value(y)
    result = sp.divide(x_enc, y_enc)
    decrypted = sp.kp.decrypt_value(result)

    # then
    expected = x / y
    npt.assert_almost_equal(decrypted, expected)

def test_calc_distance_squared(sp: ServiceProvider):
    # example from http://rosalind.info/glossary/euclidean-distance/

    # given
    x = [-1, 2, 3]
    y = [4, 0, -3]

    # when
    x_enc = sp.encrypt_vector(x)
    y_enc = sp.encrypt_vector(y)
    result = sp.calc_distance_squared(x_enc, y_enc)
    decrypted = sp.kp.decrypt_value(result)

    # then
    expected = 65
    npt.assert_almost_equal(decrypted, expected)

def test_calc_user_centroid_distance_squared(sp: ServiceProvider):
    # example from http://rosalind.info/glossary/euclidean-distance/

    # given
    x = [-1, 2, 3]
    y = [4, 0, -3]

    # when
    x_enc = sp.encrypt_vector(x)
    sum_x_squared_enc = sum(val ** 2 for val in x)

    # y_enc = sp.encrypt_vector(y)
    result = sp.calc_user_centroid_distance_squared(x_enc, sum_x_squared_enc, y)
    decrypted = sp.kp.decrypt_value(result)

    # then
    expected = 65
    npt.assert_almost_equal(decrypted, expected)

def test_calc_distance_different_len_expect_fail(sp: ServiceProvider):
    # given
    x = rng.uniform(-1000, 1000, size=10)
    y = rng.uniform(-1000, 1000, size=8)

    # when
    x_enc = sp.encrypt_vector(x)
    y_enc = sp.encrypt_vector(y)

    # then
    with pytest.raises(ValueError):
        result = sp.calc_distance(x_enc, y_enc)

def test_calc_distance(sp: ServiceProvider):
    # example from http://rosalind.info/glossary/euclidean-distance/

    # given
    x = [-1, 2, 3]
    y = [4, 0, -3]

    # when
    x_enc = sp.encrypt_vector(x)
    y_enc = sp.encrypt_vector(y)
    result = sp.calc_distance(x_enc, y_enc)
    decrypted = sp.kp.decrypt_value(result)

    # then
    expected = 65 ** 0.5
    npt.assert_almost_equal(decrypted, expected)

def test_calc_distance_random(sp: ServiceProvider):
    # given
    num_tasks = sp.conf['num_tasks']
    x = rng.uniform(-1000, 1000, size=num_tasks)
    y = rng.uniform(-1000, 1000, size=num_tasks)

    # when
    x_enc = sp.encrypt_vector(x)
    y_enc = sp.encrypt_vector(y)
    result = sp.calc_distance(x_enc, y_enc)
    decrypted = sp.kp.decrypt_value(result)

    # then
    # use numpy to calculate Euclidean distance
    expected = np.linalg.norm(x - y)
    npt.assert_almost_equal(decrypted, expected)

def test_calc_user_centroids_distance_squared_random(sp: ServiceProvider):
    # given
    num_tasks = sp.conf['num_tasks']
    x = rng.uniform(-1000, 1000, size=num_tasks)
    y = rng.uniform(-1000, 1000, size=num_tasks)

    # when
    x_enc = sp.encrypt_vector(x)
    sum_x_squared_enc = sum(val ** 2 for val in x)

    result = sp.calc_user_centroid_distance_squared(x_enc, sum_x_squared_enc, y)
    decrypted = sp.kp.decrypt_value(result)

    # then
    # use numpy to calculate Euclidean distance squared
    expected = np.linalg.norm(x - y) ** 2
    npt.assert_almost_equal(decrypted, expected)

def test_blind_and_decrypt_value(sp: ServiceProvider):
    # given
    x = rng.uniform(-1000, 1000)

    # when
    x_enc = sp.encrypt_value(x)
    decrypted = sp.blind_and_decrypt_value(x_enc)

    # then
    expected = x
    npt.assert_almost_equal(decrypted, expected)

def test_blind_and_decrypt_vector(sp: ServiceProvider):
    # given
    num_tasks = sp.conf['num_tasks']
    x = rng.uniform(-1000, 1000, size=num_tasks)

    # when
    x_enc = sp.encrypt_vector(x)
    decrypted = sp.blind_and_decrypt_vector(x_enc)

    # then
    expected = x
    npt.assert_allclose(decrypted, expected)

# Update centroids tests
def test_update_centroids(sp_example: ServiceProvider):
    # given
    # example data

    # when
    sp_example.update_centroids()

    # then
    expected = [[8/3, 14/3], [2.0, 1.0]]
    npt.assert_allclose(sp_example.centroids, expected)

def test_update_centroids_2_steps(sp_example: ServiceProvider):
    # given
    # example data

    # when
    sp_example.update_centroids()
    sp_example.update_centroids()

    # then
    expected = [[3.5, 5.5], [5/3, 5/3]]
    npt.assert_allclose(sp_example.centroids, expected)

# Test for cluster with zero users
def test_update_centroids_with_zero_users(sp_example: ServiceProvider):
    # given
    # example data
    sp_example.centroids = [[1, 3], [2, 2], [99, 99]]

    # when
    sp_example.update_centroids()

    # then
    expected = [[8/3, 14/3], [2.0, 1.0], [99, 99]]
    npt.assert_allclose(sp_example.centroids, expected)

def test_run_kmeans_2_steps(sp_example: ServiceProvider):
    # given
    # example data
    num_steps = 2

    # when
    sp_example.run_kmeans(num_steps)

    # then
    expected = [[3.5, 5.5], [5/3, 5/3]]
    npt.assert_allclose(sp_example.centroids, expected)

# Value comparison tests
def test_less_than_treshold_true(sp: ServiceProvider):
    # given
    x = 7.2156
    y = 8.402

    # when
    x_enc = sp.encrypt_value(x)
    result = sp.less_than_threshold(x_enc, y)

    # then
    expected = True
    assert result == expected

def test_less_than_treshold_false(sp: ServiceProvider):
    # given
    x = 5.174
    y = 4.9871045

    # when
    x_enc = sp.encrypt_value(x)
    result = sp.less_than_threshold(x_enc, y)

    # then
    expected = False
    assert result == expected

def test_less_than_treshold_mutual(sp: ServiceProvider):
    # given
    x = 5.1
    y = 5

    # when
    x_enc = sp.encrypt_value(x)
    y_enc = sp.encrypt_value(y)

    # x less than y
    result1 = sp.less_than_threshold(x_enc, y)

    # y less than x
    result2 = sp.less_than_threshold(y_enc, x)

    # then
    expected = [False, True]
    assert [result1, result2] == expected

def test_less_than_treshold_mutual_precision_2(sp: ServiceProvider):
    # given
    x = 0.08
    y = 0.07

    # when
    x_enc = sp.encrypt_value(x)
    y_enc = sp.encrypt_value(y)

    # x less than y
    result1 = sp.less_than_threshold(x_enc, y, precision=2)

    # y less than x
    result2 = sp.less_than_threshold(y_enc, x, precision=2)

    # then
    expected = [False, True]
    assert [result1, result2] == expected

def test_less_than_treshold_negative_threshold(sp: ServiceProvider):
    # given
    x = rng.uniform(0.1, 10)
    y = rng.uniform(-10, -0.1)

    # when
    x_enc = sp.encrypt_value(x)
    result = sp.less_than_threshold(x_enc, y)

    # then
    expected = False
    assert result == expected

def test_less_than_treshold_negative_value(sp: ServiceProvider):
    # given
    x = rng.uniform(-10, -0.1)
    y = rng.uniform(0.1, 10)

    # when
    x_enc = sp.encrypt_value(x)
    result = sp.less_than_threshold(x_enc, y)

    # then
    expected = True
    assert result == expected

def test_less_than_treshold_big_threshold(sp: ServiceProvider):
    # given
    x = rng.uniform(0.1, 10)
    y = rng.uniform(50, 100)

    # when
    x_enc = sp.encrypt_value(x)
    result = sp.less_than_threshold(x_enc, y)

    # then
    expected = True
    assert result == expected

def test_less_than_treshold_big_value(sp: ServiceProvider):
    # given
    x = rng.uniform(50, 100)
    y = rng.uniform(0.1, 10)

    # when
    x_enc = sp.encrypt_value(x)
    result = sp.less_than_threshold(x_enc, y)

    # then
    expected = False
    assert result == expected

def test_less_than_treshold_encrypted_mutual(sp: ServiceProvider):
    # given
    x = 5.1
    y = 5

    # when
    x_enc = sp.encrypt_value(x)
    y_enc = sp.encrypt_value(y)

    # x less than y
    result1_enc = sp.less_than_threshold_encrypted(x_enc, y)
    result1 = sp.kp.decrypt_value(result1_enc)

    # y less than x
    result2_enc = sp.less_than_threshold_encrypted(y_enc, x)
    result2 = sp.kp.decrypt_value(result2_enc)


    # then
    result = [result1, result2]
    expected = [0, 1]
    npt.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

def test_less_than_treshold_encrypted_mutual_precision_2(sp: ServiceProvider):
    # given
    x = 0.08
    y = 0.07

    # when
    x_enc = sp.encrypt_value(x)
    y_enc = sp.encrypt_value(y)

    # x less than y
    result1_enc = sp.less_than_threshold_encrypted(x_enc, y, precision=2)
    result1 = sp.kp.decrypt_value(result1_enc)

    # y less than x
    result2_enc = sp.less_than_threshold_encrypted(y_enc, x, precision=2)
    result2 = sp.kp.decrypt_value(result2_enc)

    # then
    result = [result1, result2]
    expected = [0, 1]
    npt.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

def test_less_than_value_true(sp: ServiceProvider):
    # given
    x = 7.2156
    y = 8.402

    # when
    x_enc = sp.encrypt_value(x)
    y_enc = sp.encrypt_value(y)
    result = sp.less_than_value(x_enc, y_enc)

    # then
    expected = True
    assert result == expected

def test_less_than_value_false(sp: ServiceProvider):
    # given
    x = 5.174
    y = 4.9871045

    # when
    x_enc = sp.encrypt_value(x)
    y_enc = sp.encrypt_value(y)
    result = sp.less_than_value(x_enc, y_enc)

    # then
    expected = False
    assert result == expected

def test_less_than_value_mutual_precision_2(sp: ServiceProvider):
    # given
    x = 14.03
    y = 14.02

    # when
    x_enc = sp.encrypt_value(x)
    y_enc = sp.encrypt_value(y)

    # x less than y
    result1 = sp.less_than_value(x_enc, y_enc, precision=2)

    # y less than x
    result2 = sp.less_than_value(y_enc, x_enc, precision=2)

    # then
    expected = [False, True]
    assert [result1, result2] == expected

# TODO: Remove - Test private methods
def test_calc_min_unsafe(sp: ServiceProvider):
    # given
    x = [2.5, 3.7, 1.4, 8.2, 1.3, 2.6]

    # when
    x_enc = sp.encrypt_vector(x)

    result = sp._ServiceProvider__calc_min_blinded(x_enc)
    decrypted = sp.kp.decrypt_value(result)

    # then
    expected = min(x)
    npt.assert_almost_equal(decrypted, expected)

def test_calc_min_unsafe_precision_2(sp: ServiceProvider):
    # given
    x = [2.5, 1.28, 3.7, 1.27, 8.2, 1.3, 2.6]
    precision = 2

    # when
    x_enc = sp.encrypt_vector(x)

    result = sp._ServiceProvider__calc_min_blinded(x_enc, precision=precision)
    decrypted = sp.kp.decrypt_value(result)

    # then
    expected = min(x)
    npt.assert_almost_equal(decrypted, expected)


def test_calc_argmin_unsafe(sp: ServiceProvider):
    # given
    x = [2.5, 3.7, 1.4, 8.2, 1.3, 2.6]

    # when
    x_enc = sp.encrypt_vector(x)

    result = sp._ServiceProvider__calc_argmin_blinded(x_enc)

    # then
    expected = np.argmin(x)
    npt.assert_almost_equal(result, expected)


# min_indicator_tests
def test_calc_min_indicator(sp: ServiceProvider):
    # given
    x = [2.5, 3.7, 7.5, 8.2, 1.3, 2.6] 
    
    # when
    x_enc = sp.encrypt_vector(x)
    indicator_enc = sp.calc_min_indicator(x_enc)
    decrypted = sp.kp.decrypt_vector(indicator_enc)

    # then
    expected = [0, 0, 0, 0, 1, 0]
    npt.assert_allclose(decrypted, expected)

def test_calc_min_indicator_multiple(sp: ServiceProvider):
    num_el = 10
    precision = 1

    for i in range(10):
        # given
        x = rng.uniform(0.1, 10, size=num_el)
        
        # when
        x_enc = sp.encrypt_vector(x)
        indicator_enc = sp.calc_min_indicator(x_enc, precision=precision)
        decrypted = sp.kp.decrypt_vector(indicator_enc)

        res_idx = np.argmax(decrypted)
        result = x[res_idx]

        # then
        expected = min(x)
        assert abs(result - expected) < (10 ** (-precision))

def test_calc_min_indicator_slow(sp: ServiceProvider):
    # given
    x = [2.5, 3.7, 7.5, 8.2, 1.3, 2.6]
    
    # when
    x_enc = sp.encrypt_vector(x)
    indicator_enc = sp.calc_min_indicator_slow(x_enc)
    decrypted = sp.kp.decrypt_vector(indicator_enc)

    # then
    expected = [0, 0, 0, 0, 1, 0]
    npt.assert_allclose(decrypted, expected, atol=1e-8)

def test_calc_min_indicator_slow_multiple(sp: ServiceProvider):
    num_el = 10
    precision = 1

    for i in range(2):
        # given
        x = rng.uniform(0.1, 10, size=num_el)
        
        # when
        x_enc = sp.encrypt_vector(x)
        indicator_enc = sp.calc_min_indicator_slow(x_enc, precision=precision)
        decrypted = sp.kp.decrypt_vector(indicator_enc)

        res_idx = np.argmax(decrypted)
        result = x[res_idx]

        # then
        expected = min(x)
        assert abs(result - expected) < (10 ** (-precision))

# calc_min tests
def test_calc_min(sp: ServiceProvider):
    # given
    x = [2.5, 3.7, 7.5, 8.2, 1.3, 2.6] 
    
    # when
    x_enc = sp.encrypt_vector(x)
    indicator_enc = sp.calc_min_indicator(x_enc)
    min_enc = sp.calc_min(x_enc, indicator_enc)
    decrypted = sp.kp.decrypt_value(min_enc)

    # then
    expected = min(x)
    npt.assert_almost_equal(decrypted, expected)

def test_calc_min_multiple(sp: ServiceProvider):
    num_el = 10
    precision = 1

    for i in range(10):
        # given
        x = rng.uniform(0.1, 10, size=num_el)
        
        # when
        x_enc = sp.encrypt_vector(x)
        indicator_enc = sp.calc_min_indicator(x_enc, precision=precision)
        min_enc = sp.calc_min(x_enc, indicator_enc)
        decrypted = sp.kp.decrypt_value(min_enc)

        # then
        expected = min(x)
        assert abs(decrypted - expected) < (10 ** (-precision))

# k-means with outlier removal tests
def test_run_kmeans_outlier_with_fast_removal(sp_example_outlier: ServiceProvider):
    # given
    # all_values = [[2, 0], [1, 3], [3, 5], [2, 2], [4, 6], [12, 15]]
    sp = sp_example_outlier
    num_steps = 2
    threshold = 2

    sp.conf['fast_outlier_removal'] = True

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

def test_run_kmeans_outlier_without_fast_removal(sp_example_outlier: ServiceProvider):
    # given
    # all_values = [[2, 0], [1, 3], [3, 5], [2, 2], [4, 6], [12, 15]]
    sp = sp_example_outlier
    num_steps = 2
    threshold = 2

    sp.conf['fast_outlier_removal'] = False

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

def test_kmeans_256_bit_key_no_fastargmin(conf):
    # givenci va una decina di volte
    base_conf = ConfigObj('experiments_config.ini', unrepr=True)
    seed = base_conf['base_seed']['general']
    encryption = True
    threshold = 5

    base_conf['fast_argmin_calc'] = False
    base_conf['key_length'] = 256

    # when
    agents = setup_experiment(base_conf, seed, encryption)
    sp = agents['sp']

    # then
    # It should not raise exceptions
    sp.run_kmeans_until_no_changes()
    sp.calc_centroids_without_outliers(threshold)
