import numpy as np


def state_to_vector(state,num_cities):
    """
    Convert the state dictionary to a vector for the neural network.
    :param state: State dictionary containing 'unvisited_cities', 'current_path', 'current_city', 'total_distance'
    :return: A concatenated vector representing the state
    """
    # Create zero vectors for unvisited cities and current path
    unvisited_cities_vector = np.zeros(num_cities)
    for city in state['unvisited_cities']:
        unvisited_cities_vector[city] = 1

    current_path_vector = np.zeros(num_cities)
    for city in state['current_path']:
        current_path_vector[city] = 1

    # Extract current city and total distance
    current_city = state['current_city']
    total_distance = state['total_distance']

    # Concatenate the vectors
    return np.concatenate([unvisited_cities_vector, current_path_vector, [current_city, total_distance]])



