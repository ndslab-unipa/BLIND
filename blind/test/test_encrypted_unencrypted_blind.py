import pytest
import numpy.testing as npt
import sys
from pathlib import Path
from configobj import ConfigObj
from typing import List
import numpy as np

# Tests for the encrypted and unencrypted version of BLIND

# Add blind directory to sys.path
sys.path.append((Path().cwd()).as_posix())
from blind.unencrypted_blind import UnUser, UnServiceProvider
from blind.paillier_blind import User, KeyProvider, ServiceProvider

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

# Tests - Encrypted and Unencrypted systems should yield same results
def test_kmeans_5_users_3_steps(kp):
    # given
    # configuration
    conf = ConfigObj('base_config.ini', unrepr=True)
    conf['num_users'] = 5
    conf['num_tasks'] = 10
    conf['num_groups'] = 3
    conf['min_value'] = 0.0
    conf['max_value'] = 10.0

    num_steps = 3

    # random values
    values = []
    for i in range(conf['num_users']):
        user_values = rng.uniform(low=conf['min_value'], high=conf['max_value'],
                                size=conf['num_tasks'])
        values.append(list(user_values))

    # create users and sp
    users = [User(kp.pu, conf, v) for v in values]
    sp = ServiceProvider(kp.pu, kp, users, conf)

    # create un_users and un_sp
    un_users = [UnUser(conf, v) for v in values]
    un_sp = UnServiceProvider(un_users, conf)

    # set same initial centroids
    # un_sp.centroids = sp.centroids

    # when
    sp.run_kmeans(num_steps)
    un_sp.run_kmeans(num_steps)

    # then
    npt.assert_allclose(sp.centroids, un_sp.centroids)


def test_centroids_without_outliers(kp):
    # given
    # configuration
    conf = ConfigObj('base_config.ini', unrepr=True)
    conf['num_users'] = 5
    conf['num_tasks'] = 10
    conf['num_groups'] = 3
    conf['min_value'] = 0.0
    conf['max_value'] = 10.0

    num_steps = 3
    thresholds = np.arange(0.5, 10, 2.5)

    # random values
    values = []
    for i in range(conf['num_users']):
        user_values = rng.uniform(low=conf['min_value'], high=conf['max_value'],
                                size=conf['num_tasks'])
        values.append(list(user_values))

    # create users and sp
    users = [User(kp.pu, conf, v) for v in values]
    sp = ServiceProvider(kp.pu, kp, users, conf)

    # create un_users and un_sp
    un_users = [UnUser(conf, v) for v in values]
    un_sp = UnServiceProvider(un_users, conf)

    # set same initial centroids
    # un_sp.centroids = sp.centroids

    # when
    sp.run_kmeans(num_steps)
    un_sp.run_kmeans(num_steps)

    # then
    for threshold in thresholds:
        centroids_enc = sp.calc_centroids_without_outliers(threshold)
        centroids_unenc = un_sp.calc_centroids_without_outliers(threshold)

        npt.assert_allclose(centroids_enc, centroids_unenc)
