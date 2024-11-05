# Base Data Science snippet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from delivery import *
from tqdm import tqdm_notebook

env = DeliveryEnvironment(n_stops = 10,method = "distance")
agent = DeliveryQAgent(env.observation_space,env.action_space)
run_n_episodes(env,agent,"training_500_stops.gif")
env.render()