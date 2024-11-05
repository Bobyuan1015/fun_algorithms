import torch
from tensordict import TensorDict
from rl4co.envs import TSPEnv
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.models.rl import REINFORCE
from rl4co.utils.trainer import RL4COTrainer
from lightning.pytorch.loggers import WandbLogger

# WandB 导入
import wandb
from torch.distributed.elastic.agent.server.local_elastic_agent import logger

# 初始化 WandB 为离线模式

# wandb.init(project="tsp", entity="fang1015", mode='offline')  # 替换为你的用户名或团队名称
wandb.init(project="tsp", entity="fang1015")  # 替换为你的用户名或团队名称
logger = WandbLogger(project="tsp", name="TSP_Training")

# Utils: download and load TSPLib instances in RL4CO
import requests, tarfile, os, gzip, shutil
from tqdm.auto import tqdm
from tsplib95.loaders import load_problem, load_solution

def download_and_extract_tsplib(url, directory="tsplib", delete_after_unzip=True):
    os.makedirs(directory, exist_ok=True)

    # Download with progress bar
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open("tsplib.tar.gz", 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                pbar.update(len(chunk))

    # Extract tar.gz
    with tarfile.open("tsplib.tar.gz", 'r:gz') as tar:
        tar.extractall(directory)

    # Decompress .gz files inside directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".gz"):
                path = os.path.join(root, file)
                with gzip.open(path, 'rb') as f_in, open(path[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(path)

    if delete_after_unzip:
        os.remove("tsplib.tar.gz")

# Download and extract all tsp files under tsplib directory
download_and_extract_tsplib("http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz")

# Load the problem from TSPLib
tsplib_dir = './tsplib'  # modify this to the directory of your prepared files
files = os.listdir(tsplib_dir)
problem_files_full = [file for file in files if file.endswith('.tsp')]

# Load the optimal solution files from TSPLib
solution_files = [file for file in files if file.endswith('.opt.tour')]

problems = []
# Load only problems with solution files
for sol_file in solution_files:
    prob_file = sol_file.replace('.opt.tour', '.tsp')
    problem = load_problem(os.path.join(tsplib_dir, prob_file))

    # NOTE: in some problem files (e.g. hk48), the node coordinates are not available
    # we temporarily skip these problems
    if not len(problem.node_coords):
        continue

    node_coords = torch.tensor([v for v in problem.node_coords.values()])
    solution = load_solution(os.path.join(tsplib_dir, sol_file))

    problems.append({
        "name": sol_file.replace('.opt.tour', ''),
        "node_coords": node_coords,
        "solution": solution.tours[0],
        "dimension": problem.dimension
    })

# order by dimension
problems = sorted(problems, key=lambda x: x['dimension'])[:1]

# Utils function: we will normalize the coordinates of the VRP instances
def normalize_coord(coord: torch.Tensor) -> torch.Tensor:
    x, y = coord[:, 0], coord[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_scaled = (x - x_min) / (x_max - x_min)
    y_scaled = (y - y_min) / (y_max - y.min())
    coord_scaled = torch.stack([x_scaled, y_scaled], dim=1)
    return coord_scaled

def tsplib_to_td(problem, normalize=True):
    coords = torch.tensor(problem['node_coords']).float()
    coords_norm = normalize_coord(coords) if normalize else coords
    td = TensorDict({
        'locs': coords_norm,
    })
    td = td[None]  # add batch dimension, in this case just 1
    return td

# Test an untrained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RL4CO env based on TorchRL
env = TSPEnv(generator_params={'num_loc': 50})

# Policy: neural network, in this case with encoder-decoder architecture
policy = AttentionModelPolicy(env_name=env.name).to(device)

# RL Model: REINFORCE and greedy rollout baseline
model = REINFORCE(env,
                  policy,
                  baseline="rollout",
                  batch_size=512,
                  train_data_size=100_000,
                  val_data_size=10_000,
                  optimizer_kwargs={"lr": 1e-4},
                  )

# 记录测试模型的结果
tds, actions = [], []

policy = policy.eval()
for problem in problems:
    with torch.inference_mode():
        td_reset = env.reset(tsplib_to_td(problem)).to(device)
        out = policy(td_reset.clone(), env, decode_type="greedy")
        unnormalized_td = env.reset(tsplib_to_td(problem, normalize=False)).to(device)
        cost = -env.get_reward(unnormalized_td, out["actions"]).item()  # unnormalized cost

    bks_sol = (torch.tensor(problem['solution'], device=device, dtype=torch.int64) - 1)[None]
    bks_cost = -env.get_reward(unnormalized_td, bks_sol)

    tds.append(tsplib_to_td(problem))
    actions.append(out["actions"])

    gap = (cost - bks_cost.item()) / bks_cost.item()

    print(f"Problem: {problem['name']:<15} Cost: {cost:<14.4f} BKS: {bks_cost.item():<10.4f}\t Gap: {gap:.2%}")

# Train
trainer = RL4COTrainer(
    max_epochs=1000,
    accelerator="gpu",
    devices=1,
    logger=logger
)

# 记录训练过程中的指标
for epoch in range(1000):  # 假设你要训练 1000 个 epochs
    trainer.fit(model)

    # 在每个 epoch 结束后记录训练结果
    wandb.log({"epoch": epoch, "loss": model.loss.item()})  # 替换为实际损失值

# Test trained model
tds, actions = [], []

policy = model.policy.eval().to(device)
for problem in problems:
    with torch.inference_mode():
        td_reset = env.reset(tsplib_to_td(problem)).to(device)
        out = policy(td_reset.clone(), env, decode_type="greedy")
        unnormalized_td = env.reset(tsplib_to_td(problem, normalize=False)).to(device)
        cost = -env.get_reward(unnormalized_td, out["actions"]).item()  # unnormalized cost

    bks_sol = (torch.tensor(problem['solution'], device=device, dtype=torch.int64) - 1)[None]
    bks_cost = -env.get_reward(unnormalized_td, bks_sol)

    tds.append(tsplib_to_td(problem))
    actions.append(out["actions"])
    gap = (cost - bks_cost.item()) / bks_cost.item()

    print(f"Problem: {problem['name']:<15} Cost: {cost:<14.4f} BKS: {bks_cost.item():<10.4f}\t Gap: {gap:.2%} predict={out['actions']} true={problem['solution']}")

    # 记录测试结果
    wandb.log({
        "problem_name": problem['name'],
        "predicted_cost": cost,
        "bks_cost": bks_cost.item(),
        "gap": gap,
        "predicted_actions": out["actions"].tolist(),
        "true_solution": problem['solution']
    })

# 完成后结束 WandB 记录
wandb.finish()
