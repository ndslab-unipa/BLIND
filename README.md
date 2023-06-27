# BLIND

BLIND is a novel privacy-preserving truth discovery system designed specifically for mobile crowdsensing.

This system introduces an advanced model that optimizes the Quality of Information (QoI) by effectively clustering user observations while maintaining their privacy.

The primary objective of BLIND is to aggregate user data in a confidential manner, ensuring the privacy of individual participants and providing reliable results to a service provider (SP) without revealing sensitive information about users.

## Features

- Privacy-Preserving Clustering: BLIND utilizes a modified version of the K-Means algorithm to group user data into clusters in a way that guarantees the utmost privacy.
- Outlier Detection: The system identifies and excludes outliers from the clustering process, enhancing the accuracy of the results.
- Centroid Calculation: New centroids are computed for each cluster after discarding the outliers, enabling a more precise representation of the data.
- Secure Privacy-Preserving Protocol: A novel secure protocol is implemented to cluster user data in a mobile crowdsensing system without compromising confidential information.
- Role-Based System: BLIND involves three main actors: users, a service provider (SP), and a key provider (KP). Each actor plays a distinct role to ensure the privacy and reliability of the system.

## Actors

1. Users: Participants of the mobile crowdsensing system who provide data observations.
2. Service Provider (SP): The entity responsible for collecting aggregated results from users while safeguarding their individual privacy.
3. Key Provider (KP): A third-party entity that facilitates secure computations and protects the privacy of users' data.

It is assumed that the Service Provider and the Key Provider are semi-honest (honest but curious) and that they cannot collude to violate user privacy.

## How It Works

1. Clustering: User data is grouped into clusters using a privacy-preserving adaptation of the K-Means algorithm.
2. Outlier Removal: Outliers, if any, are detected and excluded from the clustering process to enhance data accuracy.
3. Centroid Calculation: New centroids are computed for each cluster after removing the discarded outliers.
4. Truth Value Computation: The SP calculates a final truth value for each task based on the computed centroids.
5. Privacy Protection: Throughout the entire process, users' values remain secret and known only to themselves. The SP does not possess information about users' cluster membership or outlier status, ensuring their privacy.

## Key Provider (KP)

The KP plays a supporting role in the system and is responsible for maintaining the security and privacy of users' data. Key aspects of the KP's involvement include:

- Public-Private Key Pair: The KP generates a secure public-private key pair that is used to encrypt users' secret values.
- Computation Assistance: The SP may request the KP to perform computations required for updating cluster centroids. The KP carries out these calculations while ensuring that no confidential information is compromised.
- Blinding Techniques: To prevent the KP from deciphering users' secret information, the SP employs blinding techniques that maintain the confidentiality of user and SP data.

It is important to note that the KP does not gain access to any sensitive information about users' data, cluster membership, outlier identities, or centroids.

## Result Aggregation and Sharing

After the truth values are computed, the SP gains access to the aggregate values for each task. Depending on the nature of the service provided, the SP may decide whether to share the final result with the participating users. This flexibility allows the SP to tailor the system's output based on specific usage scenarios.

BLIND ensures that privacy is maintained throughout the process, enabling accurate and reliable results for mobile crowdsensing while protecting the individual privacy of users and their data.

## Code Structure

The provided code includes both an encrypted and unencrypted version of BLIND, along with unit tests and configuration files necessary for utilizing it.

## Installation

The project leverages the `python-paillier` library, which can be found at [https://pypi.org/project/phe/](https://pypi.org/project/phe/), to encrypt and decrypt values using the partially homomorphic Paillier cryptosystem.

To use the BLIND library, you need to have the `python-paillier` library installed. You can install it by running the following command:

```bash
pip install phe
```

## Dependencies

The `requirements.txt` file specifies all the dependencies of the Python project. It lists the external libraries or packages required for the project to run successfully. To install the dependencies listed in the `requirements.txt` file, use the following command:

```shell
pip install -r requirements.txt
```


## Functionality

The BLIND library offers the following utility functions, that can be useful on their own for other privacy-preserving calculations:

- Efficiently solve the two-party millionaires problem.
- Calculate the encrypted product of two encrypted values using blinding techniques.
- Calculate the encrypted division of two encrypted values using blinding techniques.
- Calculate the encrypted squared Euclidean distance between a vector of encrypted values and another unencrypted vector.
- Calculate the encrypted squared Euclidean distance between a vector of encrypted values and another encrypted vector.
- Check if an encrypted value is less than a plaintext threshold.
- Check if an encrypted value is less than another encrypted value.
- Find the `argmin_i` of an encrypted vector and return an encrypted `min_indicator` vector. The `min_indicator` vector has a value of 1 if the index `i` is the `argmin_i` of the vector, and 0 otherwise.
- Run the privacy-preserving K-Means algorithm for a fixed number of steps or until there are no further changes in the centroids.
