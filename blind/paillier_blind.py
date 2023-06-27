import numpy as np
import math
from bisect import bisect_left
from typing import List, Sequence, Dict, Tuple, TypeVar
from configobj import ConfigObj

from phe import paillier
from phe.paillier import PaillierPublicKey, PaillierPrivateKey, EncryptedNumber

# Encrypted version of BLIND

# TypeVar for number (either int or float)
TNum = TypeVar('TNum', float, int)

class BaseAgent:
    def __init__(self, public_key: PaillierPublicKey, conf: ConfigObj):
        self.pu = public_key
        self.conf = conf
        self.num_tasks: int = conf['num_tasks']

        self.rng: np.random.Generator

    def generate_keypair(self, n_length: int) -> Tuple[PaillierPublicKey, PaillierPrivateKey]:
        return paillier.generate_paillier_keypair(n_length=n_length)

    def encrypt_value(self, value: TNum, pu: PaillierPublicKey = None) -> EncryptedNumber:
        pu = pu if pu is not None else self.pu
        return pu.encrypt(value)

    def encrypt_vector(self, vector: List[TNum], pu: PaillierPublicKey = None) -> List[EncryptedNumber]:
        pu = pu if pu is not None else self.pu
        return [pu.encrypt(x) for x in vector]

    def blind_value(self, value: EncryptedNumber, r: float = None) -> Tuple[EncryptedNumber, float]:
        if r is None:
            assert(self.rng)
            r = self.rng.uniform(self.conf['min_r'], self.conf['max_r'])

        value_blinded = value + r

        return value_blinded, r

    def blind_value_mult(self, value: EncryptedNumber, r: float = None,
                         only_positive=False) -> Tuple[EncryptedNumber, float]:
        if r is None:
            assert(self.rng)

            # Generates random number for blinding
            if only_positive:
                min_r = self.conf['min_r_sqrt']
                max_r = self.conf['max_r_sqrt']
            else:
                min_r = self.conf['min_r_mult']
                max_r = self.conf['max_r_mult']

            r = self.rng.uniform(min_r, max_r)
    
        return value * r, r

    def vectorize(self, value: TNum, min_v: float, max_v: float,
                  precision: int, alpha: int, beta: int,
                  public_key: PaillierPublicKey = None) -> Tuple[List[EncryptedNumber], List[float]]:
        """Vectorize value, using the provided public key.

        Implements algorithm from "Efficient Solutions to Two-Party and Multiparty
        Millionaires' Problem" - Liu et al.
        
        Args:
            value (TNum): Value to vectorize
            min_v (float): Minimum allowed value
            max_v (float): Maximum allowed value
            precision (int): Number of decimal digits to preserve
            alpha (int): alpha parameter
            beta (int): beta parameter
            public_key (PaillierPublicKey, optional): Public key used to encrypt.
              alpha and beta. If None, use self.pu. Defaults to None.
        
        Returns:
            Tuple[List[EncryptedNumber], List[float]]: encrypted x_vector and
              unencrypted u_vector 
        """
        # Clip value inside [min_v, max_v] range
        value = np.clip(value, min_v, max_v)

        if public_key is None:
            public_key = self.pu

        alpha_enc = public_key.encrypt(alpha)
        beta_enc = public_key.encrypt(beta)
        
        eps = 1e-12
        u_vector = list(np.arange(min_v, max_v + eps, 10 ** (-precision)))

        idx = bisect_left(u_vector, value)
        x_vector = [alpha_enc + 0 for _ in range(idx)] + \
                   [beta_enc + 0 for _ in range(len(u_vector) - idx)]

        return x_vector, u_vector
    

class User(BaseAgent):
    def __init__(self, public_key: PaillierPublicKey, conf: ConfigObj, values: List[float] = None):
        super().__init__(public_key, conf)

        self.values = self.generate_values() if values is None else values

    def generate_values(self, rng = None) -> List[float]:
        if not rng:
            rng = np.random.default_rng()

        return list(rng.uniform(low=self.conf['min_value'],
                                high=self.conf['max_value'],
                                size=self.num_tasks))

    def encrypt_values(self) -> List[EncryptedNumber]:
        return self.encrypt_vector(self.values)

    def encrypt_values_squared(self) -> List[EncryptedNumber]:
        return self.encrypt_vector([x ** 2 for x in self.values])
    
    def encrypt_sum_of_squared_values(self) -> EncryptedNumber:
        return self.encrypt_value(sum(x ** 2 for x in self.values))

    def __repr__(self):
        return f'User(values={self.values})'


class KeyProvider(BaseAgent):
    def __init__(self, conf: ConfigObj):
        public_key, private_key = self.generate_keypair(conf['key_length'])
        super().__init__(public_key, conf)

        self.__pr = private_key
        self.rng = np.random.default_rng(self.conf['base_seed']['kp_rand'])

    def decrypt_value(self, value: EncryptedNumber) -> TNum:
        return self.__pr.decrypt(value)

    def decrypt_vector(self, vector: List[EncryptedNumber]) -> List[TNum]:
        return [self.decrypt_value(x) for x in vector]

    def square_blinded(self, value_e_blind: EncryptedNumber) -> EncryptedNumber:
        """Decrypt the value received (blinded beforehand), square it, encrypt
        it and return it back.
        
        Args:
            value_e_blind (EncryptedNumber): value to square (blinded beforehand by the caller)
        
        Returns:
            EncryptedNumber: Value squared and encrypted back
        """
        value_blind = self.decrypt_value(value_e_blind)
        value_squared = value_blind ** 2
        return self.pu.encrypt(value_squared)
    
    def sqrt_blinded(self, value_e_blind: EncryptedNumber) -> EncryptedNumber:
        """Decrypt the value received (blinded beforehand), sqrt it, encrypt
        it and return it back.
        
        Args:
            value_e_blind (EncryptedNumber): value to sqrt (blinded beforehand by the caller)
        
        Returns:
            EncryptedNumber: Value sqrt and encrypted back
        """
        value_blind = self.decrypt_value(value_e_blind)
        value_squared = value_blind ** 0.5
        return self.pu.encrypt(value_squared)

    def multiply_blinded(self, x_blind: EncryptedNumber, y_blind: EncryptedNumber) -> EncryptedNumber:
        """Return the encrypted multiplication of the two values received.

        The caller should blind both values before calling this method.
        
        Args:
            x_blind (EncryptedNumber): First value to multiply
            y_blind (EncryptedNumber): Second value to multiply
        
        Returns:
            EncryptedNumber: Encrypted result of the multiplication
        """
        x = self.decrypt_value(x_blind)
        y = self.decrypt_value(y_blind)
        return self.encrypt_value(x * y)

    def inverse_blinded(self, value_e_blind: EncryptedNumber) -> EncryptedNumber:
        """Return the encrypted inverse of value_e_blind (1/value_e_blind).
        
        Args:
            value_e_blind (EncryptedNumber): Value to invert
        
        Returns:
            EncryptedNumber: Encrypted 1/value_e_blind
        """
        value_blind = self.decrypt_value(value_e_blind)
        return self.encrypt_value(1 / value_blind)

    def compare_vectorized(self, value_enc: EncryptedNumber, x_v: List[EncryptedNumber],
                           u_v: List[float]) -> EncryptedNumber:
        """Compare an encrypted (but decryptable) value to a vectorized one,
        represented by a x_v encrypted vector and a u_v base vector.
        
        Args:
            value_enc (EncryptedNumber): Value to compare
            x_v (List[EncryptedNumber]): Vectorized encrypted vector
            u_v (List[float]): Base vector (plaintext)
        
        Returns:
            EncryptedNumber: Encrypted value representing the comparison result
        """
        value_blind = self.decrypt_value(value_enc)

        # find first element of u_v which is greater than value_blind
        index = bisect_left(u_v, value_blind)
        # ensure that the index is valid
        index = np.clip(index, 0, len(u_v) - 1)

        # sum 0 to the result, to change its ciphertext
        result = x_v[index] + 0

        return result
    
    def compare_vectorized_result_blinded(self, value_enc: EncryptedNumber,
                                          x_v: List[EncryptedNumber],
                                          u_v: List[float]) -> Tuple[EncryptedNumber, EncryptedNumber]:
        """Variant of compare_vectorized that blinds the result before sending
        it to the SP.

        The method also returns the encrypted r used for blinding.
        
        Args:
            value_enc (EncryptedNumber): Value to compare
            x_v (List[EncryptedNumber]): Vectorized encrypted vector
            u_v (List[float]): Base vector (plaintext)
        
        Returns:
            Tuple[EncryptedNumber, EncryptedNumber]:
                Tuple with two values:
                    - result blinded, encrypted with SP's public key
                    - blind value, encrypted with KP's public key
        """
        gamma_enc_sp = self.compare_vectorized(value_enc, x_v, u_v)
        gamma_blind, r = self.blind_value(gamma_enc_sp)
        r_enc = self.encrypt_value(r)
        return gamma_blind, r_enc
    
    def shuffle_and_blind(self, perm_s_vect_s_enc: List[EncryptedNumber], perm_s_rand_s_enc: List[EncryptedNumber]) -> Tuple[List[EncryptedNumber], List[EncryptedNumber], List[EncryptedNumber]]:
        """First step of the algorithm to calculate encrypted min_indicator.

        Based on the algorithm presented in:
        "Privacy-Preserving Nearest Neighbor Methods: Comparing Signals without
        Revealing Them" - Rane, S.; Boufounos, P.T.
        (MITSUBISHI ELECTRIC RESEARCH LABORATORIES)
        
        Args:
            perm_s_vect_s_enc (List[EncryptedNumber]): Blinded and permuted values
            perm_s_rand_s_enc (List[EncryptedNumber]): Permuted and encrypted random values
        
        Returns:
            Tuple[List[EncryptedNumber], List[EncryptedNumber], List[EncryptedNumber]]:
                Tuple of 3 elements:
                    - a (encoded with KP public key)
                    - b (encoded with SP public key)
                    - permutation used by KP (encoded with KP public key)
                Please refer to "Privacy-Preserving Nearest Neighbor Methods:
                Comparing Signals without Revealing Them" for more info on the
                parameters meaning.
        """
        num_el = len(perm_s_vect_s_enc)

        # paper: E_k(pi_s(X'))
        # perm_s_vect_s_enc

        # paper: E_s(pi_s(R'))
        # perm_s_rand_s_enc

        # paper: pi_s(X')
        perm_s_vect_s = self.decrypt_vector(perm_s_vect_s_enc)
        
        # Generates random numbers for blinding

        # paper: R''
        rand_c = self.rng.uniform(self.conf['min_r'], self.conf['max_r'], num_el)

        # Subtract rand_c from perm_s_rand_s
        perm_s_rand_s_c = [x - y for x, y in zip(perm_s_rand_s_enc, rand_c)] 
        
        # Permute it
        
        # paper: pi_k
        perm_c = list(range(num_el))
        self.rng.shuffle(perm_c)
        # perm_c = list(self.rng.permutation(num_el))

        # paper: E_s(beta) = E_s(pi_k(pi_s(R') - R''))
        b_enc_sp = [perm_s_rand_s_c[new_idx] for new_idx in perm_c]

        # Calculate a
        a_not_perm = [x + y for x, y in zip(rand_c, perm_s_vect_s)]

        # paper: alpha and E_k(alpha)
        a = [a_not_perm[new_idx] for new_idx in perm_c]
        a_enc = self.encrypt_vector(a)

        # Encrypt perm_c and send it to SP
        perm_c_enc = self.encrypt_vector(perm_c)

        return a_enc, b_enc_sp, perm_c_enc

    def unshuffle_and_calc_indicator(self, index_c_s: int, perm_c_enc: List[EncryptedNumber]) -> List[EncryptedNumber]:
        """Second step of the algorithm to calculate encrypted min_indicator.

        Based on the algorithm presented in:
        "Privacy-Preserving Nearest Neighbor Methods: Comparing Signals without
        Revealing Them" - Rane, S.; Boufounos, P.T.
        (MITSUBISHI ELECTRIC RESEARCH LABORATORIES)
        
        Args:
            index_c_s (int): Argmin index, permuted two times
            perm_c_enc (List[EncryptedNumber]): permutation used by KP in the
                first step of the protocol (encoded with KP public key)
        
        Returns:
            List[EncryptedNumber]: encrypted min_indicator, with a permutation
                known by SP
        """
        perm_c = self.decrypt_vector(perm_c_enc)
        index_s = perm_c[index_c_s]

        indicator = [0.0] * len(perm_c)
        indicator[index_s] = 1.0
        
        indicator_enc = self.encrypt_vector(indicator)
        return indicator_enc
        

class ServiceProvider(BaseAgent):
    def __init__(self, public_key: PaillierPublicKey, kp: KeyProvider,
                 users: List[User], conf: ConfigObj,
                 ask_user_values: bool = True):
        """Init ServiceProvider.
        
        Args:
            public_key (PaillierPublicKey): Paillier public key
            kp (KeyProvider): KeyProvider
            users (List[User]): List of users
            conf (ConfigObj): Configuration object
            ask_user_values (bool): Whether to automatically ask users for their
                encrypted values
        """
        super().__init__(public_key, conf)

        self.kp = kp
        self.users = users

        self.num_users: int = conf['num_users']
        self.k: int = conf['num_groups']
        self.__centroids = self.create_random_centroids()

        self.convergence = False
        self.iterations = 0
        self.__all_a_vectors: List[List[EncryptedNumber]] = []
        self.__all_distances: List[List[EncryptedNumber]] = []

        # init public-private keypair used to compare values with a threshold
        pub_k, priv_k = self.generate_keypair(conf['key_length'])
        self.__pu_s: PaillierPublicKey = pub_k
        self.__pr_s: PaillierPrivateKey = priv_k

        if ask_user_values:
            self.ask_users_values()
            self.ask_users_sum_of_values_sq()
        else:
            self.users_values: List[EncryptedNumber] = []
            self.users_sum_values_sq: List[EncryptedNumber] = []
        
        self.rng = np.random.default_rng(self.conf['base_seed']['sp_rand'])

    @property
    def centroids(self):
        return self.__centroids

    @centroids.setter
    def centroids(self, val):
        self.__centroids = val
        self.num_tasks = len(self.__centroids[0])
        self.k = len(self.__centroids)

    def ask_users_values(self):
        """Ask all users to send their encrypted values.
        """
        self.users_values: List[EncryptedNumber] = [user.encrypt_values() for user in self.users]

    def ask_users_values_sq(self):
        """Ask all users to send their encrypted values squared.
        """
        self.users_values_sq: List[EncryptedNumber] = [user.encrypt_values_squared() for user in self.users]
    
    def ask_users_sum_of_values_sq(self):
        """Ask all users to send the sum of their encrypted squared values.
        """
        self.users_sum_values_sq: List[EncryptedNumber] = [user.encrypt_sum_of_squared_values() for user in self.users]

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

    def calc_squared(self, value: EncryptedNumber) -> EncryptedNumber:
        """Calculate value squared, using blinding tecniques.
        
        Args:
            value (EncryptedNumber): Encrypted version of value
        
        Returns:
            EncryptedNumber: Returns the encrypted version of value squared.
        """
        # Blind value and send it to KP to have it squared
        v_blind, r = self.blind_value(value)
        v_squared_blind = self.kp.square_blinded(v_blind)

        # Calculate value_squared (encrypted)
        r_squared = self.encrypt_value(r ** 2)
        result = v_squared_blind - 2 * value * r - r_squared
        return result
    
    def calc_sqrt(self, value: EncryptedNumber) -> EncryptedNumber:
        """Calculate square root of value, using blinding tecniques.
        
        Args:
            value (EncryptedNumber): Encrypted version of value
        
        Returns:
            EncryptedNumber: Returns the encrypted version of value squared.
        """
        # Blind value and send it to KP to calc its sqrt
        v_blind, r = self.blind_value_mult(value, only_positive=True)
        v_sqrt_blind = self.kp.sqrt_blinded(v_blind)
    
        # Calculate value_sqrt (encrypted)
        r_sqrt = r ** 0.5
        result = v_sqrt_blind / r_sqrt
        return result

    def inverse(self, value: EncryptedNumber) -> EncryptedNumber:
        """Calculate encrypted inverse of value, using blinding techniques.
        
        Args:
            value (EncryptedNumber): Value to invert
        
        Returns:
            EncryptedNumber: Encrypted inverse of value (1/value)
        """
        # Blind value and send it to KP to calc its sqrt
        v_blind, r = self.blind_value_mult(value)
        v_inverse_blind = self.kp.inverse_blinded(v_blind)
            
        # Calculate 1/value (encrypted)
        result = v_inverse_blind * r
        return result

    def multiply(self, x: EncryptedNumber, y: EncryptedNumber) -> EncryptedNumber:
        """Calculate encrypted product of value1 and value2, using blinding
        techniques.
        
        Args:
            x (EncryptedNumber): First value to multiply
            y (EncryptedNumber): Second value to multiply
        
        Returns:
            EncryptedNumber: Encrypted result of the mutiplication.
        """
        # Blind values and send them to KP to have them multiplied
        x_blind, r_x = self.blind_value(x)
        y_blind, r_y = self.blind_value(y)

        z = self.kp.multiply_blinded(x_blind, y_blind)

        # Calculate x * y (encrypted)
        result = z - (x * r_y + y * r_x + r_x * r_y)
        
        return result

    def divide(self, x: EncryptedNumber, y: EncryptedNumber) -> EncryptedNumber:
        """Calculate encrypted division of value1 and value2, using blinding
        techniques.
        
        Args:
            x (EncryptedNumber): Dividend
            y (EncryptedNumber): Divisor
        
        Returns:
            EncryptedNumber: Encrypted result of the mutiplication.
        """
        return self.multiply(x, self.inverse(y))

    def calc_distance_squared(self, vector1: List[EncryptedNumber], vector2: List[TNum]) -> EncryptedNumber:
        """Calculate encrypted squared Euclidean distance between a vector of encrypted
        values and another unencrypted vector.

        Args:
            vector1 (List[EncryptedNumber]): Vector of encrypted numbers
            vector2 (List[TNum]): Vector of unencrypted numbers

        Returns:
            EncryptedNumber: Squared Euclidean distance between vector1 and vector2
        """
        if len(vector1) != len(vector2):
            raise ValueError('vector1 and vector2 should have same length.')

        diff_squared = [self.calc_squared(v1 - v2) for v1, v2 in zip(vector1, vector2)]
        return sum(diff_squared)

    def calc_user_centroid_distance_squared(self, vector1: List[EncryptedNumber], sum_vector1_sq: EncryptedNumber, vector2: List[TNum]) -> EncryptedNumber:
        """Calculate encrypted squared Euclidean distance between user values and
        a centroid.

        This method is used by the ServiceProvider to calculate the encrypted
        distance between each user (represented by his values) and each centroid.

        User values and sum of squared user values are known by the SP in encrypted form.

        Args:
            vector1 (List[EncryptedNumber]): Vector of encrypted numbers
            sum_vector1_sq (EncryptedNumber): Sum of encrypted numbers squared
            vector2 (List[TNum]): Vector of unencrypted numbers

        Returns:
            EncryptedNumber: Squared Euclidean distance between vector1 and vector2
        """ 
        v1 = np.array(vector1)
        v2 = np.array(vector2)

        sum_v2_sq = float(sum(v2 ** 2))
        sum_crossed = sum(-2 * np.multiply(v1, v2))

        result = sum_vector1_sq + sum_v2_sq + sum_crossed

        return result

    def calc_distance(self, vector1: List[EncryptedNumber], vector2: List[TNum]) -> EncryptedNumber:
        """Calculate encrypted Euclidean distance between a vector of encrypted
        values and another unencrypted vector.
        
        Args:
            vector1 (List[EncryptedNumber]): Vector of encrypted numbers
            vector2 (List[TNum]): Vector of unencrypted numbers
    
        Returns:
            EncryptedNumber: Euclidean distance between vector1 and vector2
        """
        # TODO: use calc_user_centroid_distance_squared?
        return self.calc_sqrt(self.calc_distance_squared(vector1, vector2))
    
    def include_user(self, distances: List[EncryptedNumber], a_vector: List[EncryptedNumber], threshold: float):
        # calculate distance from nearest centroid (encrypted)
        min_distance = self.encrypt_value(0)
        
        # for each centroid
        for distance, a_l in zip(distances, a_vector):
            min_distance += self.multiply(distance, a_l)
        
        # include_user is E(0) if user is an outlier, E(1) otherwise
        return self.less_than_threshold_encrypted(min_distance, threshold)

    def calc_updated_centroids(self, all_a_vectors: List[EncryptedNumber] = None) -> List[List[EncryptedNumber]]:
        """Return updated encrypted centroids using user values and current centroids.

        User i (u_i) values contribute to the calculation of the new centroid l
        only if u_i belongs to cluster l.
        
        Returns:
            List[List[EncryptedNumber]]: New centroids (encrypted)
        """
        num_tasks = len(self.centroids[0])
        k = len(self.centroids)

        new_centroids = []
        for l in range(k):
            new_centroids.append(self.encrypt_vector([0] * num_tasks))

        # TODO: check if we can encrypt 0 one time and then re-randomize it
        zero_enc = self.encrypt_value(0)
        a_vectors_sum = [zero_enc + 0 for _ in range(k)]

        if all_a_vectors is None:
            all_a_vectors = [self.calc_vector_a(values, sum_sq) for values, sum_sq in zip(self.users_values, self.users_sum_values_sq)]
            self.__all_a_vectors = all_a_vectors

        # for each user
        for values, a_vector in zip(self.users_values, all_a_vectors):
            # for each centroid
            for l in range(k):
                # a_l is E(1) if user is in cluster l, E(0) otherwise
                a_l = a_vector[l]

                # update a_vectors_sum
                a_vectors_sum[l] += a_l
                
                # for each coordinate (task)
                for i in range(num_tasks):
                    new_centroids[l][i] += self.multiply(values[i], a_l)
                
        # divide each centroid coordinate by the number of users in that cluster
        # for each centroid
        for l in range(k):
            # for each coordinate
            for i in range(num_tasks):
                if not self.less_than_threshold(a_vectors_sum[l], 1):
                    new_centroids[l][i] = self.divide(new_centroids[l][i], a_vectors_sum[l])
                else:
                    # if there are no users in the l-th cluster
                    new_centroids[l][i] = self.encrypt_value(self.centroids[l][i])

        return new_centroids
    
    # FIXME: return value should be TNum
    def blind_and_decrypt_value(self, value: EncryptedNumber) -> float:
        """Blind value and ask the KeyProvider to decrypt it.

        The SP uses this method when it needs to have a value decrypted by the KP,
        whilst preventing the KP from discovering the value.
        
        Args:
            value (EncryptedNumber): Value to decrypt
        
        Returns:
            float: Value decrypted
        """
        blinded_enc, r = self.blind_value(value)
        blinded = self.kp.decrypt_value(blinded_enc)

        return blinded - r

    # FIXME: return value should be List[TNum]
    def blind_and_decrypt_vector(self, vector: List[EncryptedNumber]) -> List[float]:
        """Blind vector and ask the KeyProvider to decrypt it.

        The SP uses this method when it needs to have a vector decrypted by the KP,
        whilst preventing the KP from discovering the values of vector.
        
        Args:
            vector (EncryptedNumber): Vector to decrypt
        
        Returns:
            float: Vector decrypted
        """
        return [self.blind_and_decrypt_value(v) for v in vector]

    def update_centroids(self) -> None:
        """Update centroids using blinding techniques.
        """
        centroids_enc = self.calc_updated_centroids()
        self.centroids = [self.blind_and_decrypt_vector(c) for c in centroids_enc]
    
    def run_kmeans(self, num_steps: int) -> None:
        """Run Privacy-Preserving K-Means algorithm for num_steps steps.
    
        Automatically update centroids.
        
        Args:
            num_steps (int): Number of steps to perform
        """
        for i in range(num_steps):
            self.update_centroids()
    
    def run_kmeans_until_no_changes(self, max_iter=None) -> int:
        """Run Privacy-Preserving K-Means algorithm until no further
        changes in centroids.
    
        Automatically update centroids.

        Args:
            max_iter (int): Maximum number of iterations.

        Returns:
            int: The number of iterations performed or -1 if no convergence in
                 max_iter iterations.
        """
        if max_iter is None:
            max_iter = self.conf['max_iter']

        changes = True
        while changes and self.iterations < max_iter:
            changes = False
            self.iterations += 1
            print(f'K-Means:  iteration {self.iterations}')

            old_centroids = self.centroids
            self.update_centroids()

            for old_c, new_c in zip(old_centroids, self.centroids):
                for old_value, new_value in zip(old_c, new_c):
                    if abs(old_value - new_value) > 1e-4:
                        changes = True
                        break

        if not changes:
            self.convergence = True
        else:
            self.iterations = -1
        
        return self.iterations

    def less_than_threshold(self, value: EncryptedNumber, threshold: float,
                            min_v: float = None, max_v: float = None,
                            precision: int = None) -> bool:
        """Check if an encrypted value is less than a plaintext threshold.

        Implements algorithm from "Efficient Solutions to Two-Party and Multiparty
        Millionaires' Problem" - Liu et al.
        
        Args:
            value (EncryptedNumber): Value to compare
            threshold (float): Threshold to check
            min_v (float, optional): Minimum value the threshold can be. If None,
              use parameter in config. Defaults to None.
            max_v (float, optional): Maximum value the threshold can be. If None,
              use parameter in config. Defaults to None.
            precision (int, optional): Number of decimal digits to consider.
              If None, use parameter in config. Defaults to None.
        
        Raises:
            RuntimeError: Raises error if the value returned by the KeyProvider
              is not one of alpha or beta, as randomly selected by the ServiceProvider.
        
        Returns:
            bool: Whether value is less than threshold
        """
        # blind value
        value_blind, r = self.blind_value(value, r=self.rng.uniform(
                                            self.conf['vect_r_min'],
                                            self.conf['vect_r_max']))

        # blind threshold with the same r
        t_blind = threshold + r

        min_v = min_v if min_v is not None else self.conf['vect_min'] 
        max_v = max_v if max_v is not None else self.conf['vect_max']
        precision = precision if precision is not None else self.conf['vect_precision']

        # vectorize threshold
        alpha = self.rng.uniform(-100, 100)
        beta = self.rng.uniform(-100, 100)
        x_v, u_v = self.vectorize(t_blind,
                                  min_v=min_v,
                                  max_v=max_v,
                                  precision=precision,
                                  alpha=alpha,
                                  beta=beta,
                                  public_key=self.__pu_s)

        # send blinded value and vectorized threshold to KP
        result_enc = self.kp.compare_vectorized(value_blind, x_v, u_v)
        result = self.__pr_s.decrypt(result_enc)

        if result == alpha:     # value is less than threshold
            is_less = True
        elif result == beta:    # value is greater than threshold
            is_less = False
        else:
            raise RuntimeError('Are you sure the KP is honest but curious?')

        return is_less

    def less_than_threshold_encrypted(self, value: EncryptedNumber, threshold: float,
                            min_v: float = None, max_v: float = None,
                            precision: int = None) -> EncryptedNumber:
        """Check if an encrypted value is less than a plaintext threshold. The
        result returned is encrypted (1 if less than threshold, 0 otherwise).

        Implements algorithm from "Efficient Solutions to Two-Party and Multiparty
        Millionaires' Problem" - Liu et al.
        
        Args:
            value (EncryptedNumber): Value to compare
            threshold (float): Threshold to check
            min_v (float, optional): Minimum value the threshold can be. If None,
              use parameter in config. Defaults to None.
            max_v (float, optional): Maximum value the threshold can be. If None,
              use parameter in config. Defaults to None.
            precision (int, optional): Number of decimal digits to consider.
              If None, use parameter in config. Defaults to None.
        
        Raises:
            RuntimeError: Raises error if the value returned by the KeyProvider
              is not one of alpha or beta, as randomly selected by the ServiceProvider.
        
        Returns:
            bool: Whether value is less than threshold
        """
        # blind value
        value_blind, r = self.blind_value(value, r=self.rng.uniform(
                                            self.conf['vect_r_min'],
                                            self.conf['vect_r_max']))

        # blind threshold with the same r
        t_blind = threshold + r

        min_v = min_v if min_v is not None else self.conf['vect_min'] 
        max_v = max_v if max_v is not None else self.conf['vect_max']
        precision = precision if precision is not None else self.conf['vect_precision']

        # vectorize threshold
        gamma_rand = self.rng.uniform(-100, 100)
        alpha = gamma_rand + 1
        beta = gamma_rand
        x_v, u_v = self.vectorize(t_blind,
                                  min_v=min_v,
                                  max_v=max_v,
                                  precision=precision,
                                  alpha=alpha,
                                  beta=beta,
                                  public_key=self.__pu_s)

        # send blinded value and vectorized threshold to KP
        result_blind_enc, kp_rand_enc = self.kp.compare_vectorized_result_blinded(value_blind, x_v, u_v)
        result_blind = self.__pr_s.decrypt(result_blind_enc)

        # encrypted value: 1 if value is less than threshold, 0 otherwise
        result_enc = self.encrypt_value(result_blind) - kp_rand_enc - gamma_rand

        return result_enc

    def less_than_value(self, value1: EncryptedNumber, value2: EncryptedNumber,
                        min_v=None, max_v=None, precision=None) -> bool:
        """Check if an encrypted value is less than another encrypted value.

        It actually checks if (value1 - value2) is less than a threshold of 0,
        calling self.less_than_threshold method.
        
        Args:
            value1 (EncryptedNumber): First value to compare
            value2 (EncryptedNumber): Second value to compare
            min_v (float, optional): Minimum value the threshold can be. If None,
              use parameter in config. Defaults to None.
            max_v (float, optional): Maximum value the threshold can be. If None,
              use parameter in config. Defaults to None.
            precision (int, optional): Number of decimal digits to consider.
              If None, use parameter in config. Defaults to None.
        
        Returns:
            bool: Whether value1 is less than value2
        """
        diff = value1 - value2
        threshold = 0
        min_v = min_v if min_v is not None else self.conf['vect_min_neg']

        return self.less_than_threshold(diff, threshold, min_v=min_v, max_v=max_v,
                                        precision=precision)

    def less_than_value_encrypted(self, value1: EncryptedNumber, value2: EncryptedNumber,
                        min_v=None, max_v=None, precision=None) -> EncryptedNumber:
        """Check if an encrypted value is less than another encrypted value.

        It actually checks if (value1 - value2) is less than a threshold of 0,
        calling self.less_than_threshold_encrypted method.
        
        Args:
            value1 (EncryptedNumber): First value to compare
            value2 (EncryptedNumber): Second value to compare
            min_v (float, optional): Minimum value the threshold can be. If None,
              use parameter in config. Defaults to None.
            max_v (float, optional): Maximum value the threshold can be. If None,
              use parameter in config. Defaults to None.
            precision (int, optional): Number of decimal digits to consider.
              If None, use parameter in config. Defaults to None.
        
        Returns:
            bool: Whether value1 is less than value2
        """
        diff = value1 - value2
        threshold = 0
        min_v = min_v if min_v is not None else self.conf['vect_min_neg']

        return self.less_than_threshold_encrypted(diff, threshold, min_v=min_v,
                                                  max_v=max_v, precision=precision)

    def calc_vector_a(self, values_enc: List[EncryptedNumber], sum_values_sq_enc: EncryptedNumber) -> List[EncryptedNumber]:
        """Return encrypted vector "a" for a user.
        
        Args:
            values (List[EncryptedNumber]): Encrypted user values
            sum_values_enc (EncryptedNumber): Encrypted sum of user values squared
        
        Returns:
            List[EncryptedNumber]: Encrypted vector "a" (1 if member of that cluster, 0 otherwise)
        """
        if self.conf['fast_distance_calc'] is True:
            distances = [self.calc_user_centroid_distance_squared(values_enc, sum_values_sq_enc, centroid) for centroid in self.centroids]
        else:
            distances = [self.calc_distance_squared(values_enc, centroid) for centroid in self.centroids]

        a_vector = self.calc_min_indicator(distances)

        return a_vector

    def calc_min_indicator(self, vector: List[EncryptedNumber], precision: int = None) -> List[EncryptedNumber]:
        """Find the argmin_i in encrypted vector and return an encrypted min_indicator.

        The min_indicator returned is a vector with value 1 if the index i is
        argmin_i(vector), 0 otherwise.

        For example, if the unencrypted version of vector is 
            [1.2, 5.3, 0.4, 2.9]
        then the unencrypted version min_indicator will be
            [0,   0,   1,   0]
        
        Based on the algorithm presented in:
        "Privacy-Preserving Nearest Neighbor Methods: Comparing Signals without
        Revealing Them" - Rane, S.; Boufounos, P.T.
        (MITSUBISHI ELECTRIC RESEARCH LABORATORIES)
        
        Args:
            vector (List[EncryptedNumber]): List of values
            precision (int, optional): Number of decimal digits to consider.
              If None, use parameter in config. Defaults to None.
        
        Returns:
            List[EncryptedNumber]: Encrypted min_indicator vector
        """
        if self.conf['fast_argmin_calc'] is False:
            return self.calc_min_indicator_slow(vector, precision)

        num_el = len(vector)

        # Generates random numbers for blinding

        # paper: R'
        rand_s = self.rng.uniform(self.conf['min_r'], self.conf['max_r'], num_el)

        # paper: E_k(X')
        vect_s = [v_el - s_el for v_el, s_el in zip(vector, rand_s)]

        # paper: pi_s
        perm_s = list(self.rng.permutation(num_el))

        # paper: E_k(pi_s(X')) -> X encrypted, blinded and permuted with pi_s
        perm_s_vect_s = [vect_s[new_idx] for new_idx in perm_s]
        
        # paper: E_s(pi_s(R')) -> R' encrypted with PU_s and permuted with pi_s
        perm_s_rand_s = [rand_s[new_idx] for new_idx in perm_s]
        perm_s_rand_s_enc = self.encrypt_vector(perm_s_rand_s, pu=self.__pu_s)

        # paper: E_k(alpha), E_s(beta), E_k(pi_k)
        a_enc, b_enc_sp, perm_c_enc = self.kp.shuffle_and_blind(perm_s_vect_s, perm_s_rand_s_enc)

        # Decrypt b with SP's private key
        b = [self.__pr_s.decrypt(x) for x in b_enc_sp]

        a_plus_b_enc = [x + y for x, y in zip(a_enc, b)]

        # Argmin with double shuffle
        # paper: pi_k(pi_s(argmin))
        index_c_s = self.__calc_argmin_blinded(a_plus_b_enc, precision=precision)

        # Encrypted min indicator shuffled
        indicator_s_enc = self.kp.unshuffle_and_calc_indicator(index_c_s, perm_c_enc)

        # Unshuffle
        indicator_enc = [0] * num_el
        for i in range(num_el):
           indicator_enc[perm_s[i]] = indicator_s_enc[i] 


        return indicator_enc

    def calc_min_indicator_slow(self, vector: List[EncryptedNumber], precision: int = None) -> List[EncryptedNumber]:
        """Find the argmin_i in encrypted vector and return an encrypted min_indicator.

        The min_indicator returned is a vector with value 1 if the index i is
        argmin_i(vector), 0 otherwise.

        For example, if the unencrypted version of vector is 
            [1.2, 5.3, 0.4, 2.9]
        then the unencrypted version min_indicator will be
            [0,   0,   1,   0]
        
        Simple (but slow) implementation.
        
        Args:
            vector (List[EncryptedNumber]): List of values
            precision (int, optional): Number of decimal digits to consider.
              If None, use parameter in config. Defaults to None.
        
        Returns:
            List[EncryptedNumber]: Encrypted min_indicator vector
        """
        num_elem = len(vector)
        one_enc = self.encrypt_value(1)
        a_vector = [one_enc + 0 for i in range(num_elem)]

        # only the minimum value will be "less than" every other element
        # then, a_vector[i] will be E(1) only if i is the argmin
        for i in range(num_elem):
            for j in range(num_elem):
                if i != j:
                    # compare vector[i] to vector[j]
                    less_than = self.less_than_value_encrypted(vector[i], vector[j])
                    a_vector[i] = self.multiply(a_vector[i], less_than)

        return a_vector        

    def calc_min(self, vector: List[EncryptedNumber], min_indicator: List[EncryptedNumber]) -> EncryptedNumber:
        min_value = 0
        
        for elem, indicator in zip(vector, min_indicator):
            min_value += self.multiply(elem, indicator)
        
        return min_value

    def __calc_min_distance_old(self, values_enc: List[EncryptedNumber], sum_values_sq_enc: EncryptedNumber) -> EncryptedNumber:
        if self.conf['fast_distance_calc'] == True:
            distances = [self.calc_user_centroid_distance_squared(values_enc, sum_values_sq_enc, centroid) for centroid in self.centroids]
        else:
            distances = [self.calc_distance_squared(values_enc, centroid) for centroid in self.centroids]
        
        a_vector = self.calc_vector_a(values_enc, sum_values_sq_enc)

        min_distance = self.calc_min(distances, a_vector)
        return min_distance

    def calc_all_user_distances(self) -> List[List[EncryptedNumber]]:
        all_distances = []
        for values_enc, sum_values_sq_enc in zip(self.users_values, self.users_sum_values_sq):
            # calculate min distance for each user
            if self.conf['fast_distance_calc'] is True:
                distances = [self.calc_user_centroid_distance_squared(values_enc, sum_values_sq_enc, centroid) for centroid in self.centroids]
            else:
                distances = [self.calc_distance_squared(values_enc, centroid) for centroid in self.centroids]
            all_distances.append(distances)
        
        return all_distances

    def calc_centroids_without_outliers(self, threshold: float) -> List[EncryptedNumber]:
        if self.convergence:
            # last step all_a_vectors and all_distances are unchanged
            all_a_vectors = self.__all_a_vectors
            all_distances = self.__all_distances
        else:
            # Calculate all distances between users and centroids
            all_distances = self.calc_all_user_distances()
            self.__all_distances = all_distances

            # calculate all users' a_vector
            all_a_vectors = [self.calc_min_indicator(distances) for distances in all_distances]

        if self.conf['fast_outlier_removal'] is True:
            for distances, a_vector in zip(all_distances, all_a_vectors):
                include = self.include_user(distances, a_vector, threshold)

                # for each centroid, update a_vector
                for l in range(self.k):
                    a_vector[l] = self.multiply(a_vector[l], include)

            centroids_enc = self.calc_updated_centroids(all_a_vectors)
        else:
            # old minimum calculation (redundant a_vector and distances calc)
            min_distances = [self.__calc_min_distance_old(values_enc, sum_values_sq_enc) for values_enc, sum_values_sq_enc in zip(self.users_values, self.users_sum_values_sq)]

            # compare min distance with threshold
            zero_enc = self.encrypt_value(0)
            one_enc = self.encrypt_value(1)

            o_vect = [self.less_than_threshold_encrypted(d, threshold) for d in min_distances]

            # multiply users a_vector with o_vect
            new_all_a_vectors: List[List[EncryptedNumber]] = []
            
            # for each user
            for a_vector, o in zip(all_a_vectors, o_vect):
                new_a_vector = [self.multiply(a, o) for a in a_vector]
                new_all_a_vectors.append(new_a_vector)

            # Calc new centroids without outliers
            centroids_enc = self.calc_updated_centroids(new_all_a_vectors)
        
        centroids = [self.blind_and_decrypt_vector(c) for c in centroids_enc]

        return centroids
    
    # Private methods
    def __calc_vector_a_unsafe(self, values_enc: List[EncryptedNumber]) -> List[EncryptedNumber]:
        """Return encrypted vector "a" for a user (UNSAFE version).
        
        Args:
            values (List[EncryptedNumber]): Encrypted user values
        
        Returns:
            List[EncryptedNumber]: Encrypted vector "a" (1 if member of that cluster, 0 otherwise)
        """
        distances = [self.calc_distance_squared(values_enc, centroid) for centroid in self.centroids]
    
        # unsafe
        distances_plain = self.kp.decrypt_vector(distances)
        argmin = np.argmin(distances_plain)
    
        a_vector_plain = [0.0] * len(self.centroids)
        a_vector_plain[argmin] = 1.0
        a_vector = self.encrypt_vector(a_vector_plain)
    
        return a_vector

    def __calc_min_blinded(self, vector: List[EncryptedNumber], min_v: float = None,
                        max_v: float = None, precision: int = None) -> EncryptedNumber:
        """Return the encrypted minimum value of encrypted vector.
    
        The method is unsafe because the SP learns the minimum index.
        
        Args:
            vector (List[EncryptedNumber]): List of encrypted values
            min_v (float, optional): Minimum value the threshold can be. If None,
              use parameter in config. Defaults to None.
            max_v (float, optional): Maximum value the threshold can be. If None,
              use parameter in config. Defaults to None.
            precision (int, optional): Number of decimal digits to consider.
              If None, use parameter in config. Defaults to None.
        
        Returns:
            EncryptedNumber: Encrypted minimum of vector
        """        
        curr_min = vector[0]
    
        for elem in vector[1:]:
            if self.less_than_value(elem, curr_min, min_v=min_v,
                                    max_v=max_v, precision=precision):
                curr_min = elem
    
        return curr_min
    
    def __calc_argmin_blinded(self, vector: List[EncryptedNumber], min_v: float = None,
                        max_v: float = None, precision: int = None) -> int:
        """Return the encrypted argmin value of encrypted vector.
    
        The method is unsafe because the SP learns the minimum index.
        It must only be used after both SP and KP permute the vector.
        
        Args:
            vector (List[EncryptedNumber]): List of encrypted values
            min_v (float, optional): Minimum value the threshold can be. If None,
              use parameter in config. Defaults to None.
            max_v (float, optional): Maximum value the threshold can be. If None,
              use parameter in config. Defaults to None.
            precision (int, optional): Number of decimal digits to consider.
              If None, use parameter in config. Defaults to None.
        
        Returns:
            int: argmin of vector
        """        
        curr_min = vector[0]
        min_index = 0
    
        for i, elem in enumerate(vector[1:]):
            if self.less_than_value(elem, curr_min, min_v=min_v,
                                    max_v=max_v, precision=precision):
                curr_min = elem
                
                # enumerate index (i) start from 0 instead of 1
                min_index = i + 1
    
        return min_index


    def __repr__(self):
        return f'ServiceProvider(centroids={self.centroids})'
