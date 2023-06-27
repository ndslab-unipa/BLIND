import numpy as np
import math
from typing import List, Tuple
from configobj import ConfigObj

# Unencrypted version of BLIND, for test purposes only.

class UnBaseAgent:
    def __init__(self, conf: ConfigObj):
        self.conf = conf
        self.num_tasks: int = conf['num_tasks']

class UnUser(UnBaseAgent):
    def __init__(self, conf: ConfigObj, values: List[float]):
        super().__init__(conf)

        self.values = self.generate_values() if values is None else values

    def generate_values(self, rng = None) -> List[float]:
        if rng is None:
            rng = np.random.default_rng()

        return list(rng.uniform(low=self.conf['min_value'],
                                high=self.conf['max_value'],
                                size=self.num_tasks))

    def __repr__(self):
        return f'User(values={self.values})'

class UnServiceProvider(UnBaseAgent):
    def __init__(self, users: List[UnUser], conf: ConfigObj):
        super().__init__(conf)

        self.users = users

        self.num_users: int = conf['num_users']
        self.k: int = conf['num_groups']
        self.iterations = 0

        self.__centroids = self.create_random_centroids()

        # List of user membership (index i represents the i-th cluster)
        self.membership: List[List[int]] = self.__empty_membership()

    @property
    def centroids(self):
        return self.__centroids

    @centroids.setter
    def centroids(self, val):
        self.__centroids = val
        self.num_tasks = len(self.__centroids[0])
        self.k = len(self.__centroids)

    def create_random_centroids(self) -> np.array:
        """Create random centroids to start the PPK-Means algorithm.
    
        Create k centroids, where k is the number of clusters.
        Each centroid is m-dimensional, where m is the number of tasks.
        
        Returns:
            np.array: Array of k random centroids.
        """
        seed = self.conf['base_seed']['kmeans']
        rng = np.random.default_rng(seed)

        return rng.uniform(low=self.conf['min_value'],
                           high=self.conf['max_value'],
                           size=(self.k, self.num_tasks))

    def calc_distance_squared(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate squared Euclidean distance between a vector of encrypted
        values and another unencrypted vector.

        This method is used by the ServiceProvider to calculate the distance
        between each user (represented by his values) and each centroid.
        
        Args:
            vector1 (List[float]): First vector
            vector2 (List[float]): Second vector

        Returns:
            TNum: Squared Euclidean distance between vector1 and vector2
        """
        if len(vector1) != len(vector2):
            raise ValueError('vector1 and vector2 should have same length.')

        diff_squared = [(v1 - v2) ** 2 for v1, v2 in zip(vector1, vector2)]
        return np.sum(diff_squared)

    def find_min_distance(self, vector: List[float], centroids: List[List[float]]) -> Tuple[int, float]:
        distances = [self.calc_distance_squared(vector, c) for c in centroids]
        min_distance = distances[0]
        min_index = 0

        for i, dist in enumerate(distances[1:]):
            if dist < min_distance:
                min_distance = dist
                # i starts from 0, actual indices starts from 1
                min_index = i + 1
        
        return min_index, min_distance

    def update_memberships(self) -> None:
        # reset memberships
        self.membership = self.__empty_membership()

        for i, user in enumerate(self.users):
            min_index, _ = self.find_min_distance(user.values, self.centroids)
            self.membership[min_index].append(i)
    
    def update_centroids(self) -> None:
        """Update centroids.
        """
        self.update_memberships()
        self.centroids = self.calc_updated_centroids()

    def calc_updated_centroids(self, include_user: List[bool] = None) -> List[List[float]]:
        """Return updated centroids using user values and current centroids.

        User i (u_i) values contribute to the calculation of the new centroid l
        only if u_i belongs to cluster l.
        
        Returns:
            List[TNum]: New centroids (encrypted)
        """
        assert(self.num_tasks == len(self.centroids[0]))
        assert(self.k == len(self.centroids))

        new_centroids = []
        for l in range(self.k):
            new_centroids.append([0.0 for i in range(self.num_tasks)])

        if not include_user:
            # include all users
            include_user = [True for i in range(len(self.users))]

        # for each centroid
        for l in range(self.k):
            num_users = 0

            # for each user in cluster l
            for user_index in self.membership[l]:
                values = self.users[user_index].values

                if include_user[user_index]:
                    num_users += 1

                    # for each coordinate (task)
                    for i in range(self.num_tasks):
                        new_centroids[l][i] += values[i]
        
            # divide each centroid coordinate by the number of users in that cluster
            # for each coordinate (task)
            for i in range(self.num_tasks):
                if num_users > 0:
                    new_centroids[l][i] /= num_users
                else:
                    # if there are no users in the l-th cluster
                    new_centroids[l][i] = self.centroids[l][i]

        return new_centroids

    def run_kmeans(self, num_steps: int) -> None:
        """Run unencrypted Privacy-Preserving K-Means algorithm for num_steps steps.
    
        Automatically update centroids.
        
        Args:
            num_steps (int): Number of steps to perform
        """
        for i in range(num_steps):
            self.update_centroids()

    def run_kmeans_until_no_changes(self) -> int:
        """Run unencrypted Privacy-Preserving K-Means algorithm until no further
        changes in centroids.
    
        Automatically update centroids.

        Returns:
            int: The number of iterations performed
        """
        changes = True
        while changes:
            changes = False
            self.iterations += 1
            old_centroids = self.centroids
            self.update_centroids()

            for old_c, new_c in zip(old_centroids, self.centroids):
                for old_value, new_value in zip(old_c, new_c):
                    if old_value != new_value:
                        changes = True
                        break
        
        return self.iterations

    def calc_centroids_without_outliers(self, threshold: float) -> List[List[float]]:
        # calculate min distance for each user
        min_distances = []
        for user in self.users:
            _, min_distance = self.find_min_distance(user.values, self.centroids)
            min_distances.append(min_distance)

        # compare min distance with threshold
        include_user = []
        for distance in min_distances:
            include = True if distance <= threshold else False
            include_user.append(include)

        new_centroids = self.calc_updated_centroids(include_user)

        return new_centroids

    def __empty_membership(self) -> List[List[int]]:
        return [[] for i in range(self.k)]
