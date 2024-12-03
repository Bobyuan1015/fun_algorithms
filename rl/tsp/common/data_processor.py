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




def adjust_reward(reward_params, reward_strategy, history_best_distance):
    """
    Adjust the reward based on the reward strategy.
    :param reward_params: Dictionary with parameters needed for reward calculation
    :return: Adjusted reward
    """
    distance = reward_params["distance"]
    done = reward_params["done"]
    total_distance = reward_params["total_distance"]
    visited = reward_params["visited"]
    env = reward_params["env"]

    # Define reward strategies based on the distance and other parameters
    if reward_strategy == "negative_distance":
        reward = -distance
    elif reward_strategy == "negative_distance_with_return_bonus":
        reward = -distance + (10 if done else 0)
    elif reward_strategy == "negative_distance_return_with_best_bonus":
        reward = -distance + (10 if done and total_distance < history_best_distance else 0)
    elif reward_strategy == "zero_step_final_bonus":
        reward = 0 if not done else 50 / total_distance
    elif reward_strategy == "zero_step_final_bonus_best_path":
        reward = 0 if not done else (
                50 / total_distance + (10 if total_distance < history_best_distance else 0))
    elif reward_strategy == "positive_final_bonus":
        reward = 0 if not done else (100 if total_distance < history_best_distance else 50)
    elif reward_strategy == "adaptive_reward":
        reward = -distance + (2 * (len(visited) / env.num_cities))
    elif reward_strategy == "dynamic_penalty_reduction":
        reduction_factor = len(visited) / env.num_cities
        reward = -distance * (1 - reduction_factor)
    elif reward_strategy == "segment_bonus":
        reward = 5 if len(visited) % 5 == 0 else -distance
    elif reward_strategy == "average_baseline":
        avg_distance = np.mean(
            [env.distance_matrix[visited[i], visited[i + 1]] for i in range(len(visited) - 1)]) if len(
            visited) > 1 else distance
        reward = (5 if total_distance < avg_distance else -5) if done else -distance
    elif reward_strategy == "feedback_adjustment":
        reward = 20 if done and total_distance < history_best_distance else -distance
    elif reward_strategy == "heuristic_mst":
        mst_reward = env._compute_mst_reward()
        reward = mst_reward - distance
    elif reward_strategy == "curiosity_driven":
        unvisited_cities = set(range(env.num_cities)) - set(visited)
        if unvisited_cities:
            farthest_city = max(unvisited_cities, key=lambda city: env.distance_matrix[env.current_city, city])
            reward = max(env.distance_matrix[env.current_city, farthest_city] - distance, -distance)
        else:
            reward = -distance
    elif reward_strategy == "diversity_driven":
        reward = len(set(visited)) / env.num_cities * -distance
    else:
        reward = 0

    return reward
