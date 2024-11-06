import torch
from rl4co.envs import TSPEnv
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.models.rl import REINFORCE
from rl4co.utils.trainer import RL4COTrainer
from lightning.pytorch.loggers import WandbLogger
import wandb
from rl.tsp.tsplib95_generator import generate_data_by_tsplib, tsplib_to_td


def test(data):
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
    for problem in data:
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

def train_with_rl4co(data):
    # wandb.init(project="tsp-old", entity="fang1015", mode='offline')  # 替换为你的用户名或团队名称
    wandb.init(project="tsp-old", entity="fang1015")  # 替换为你的用户名或团队名称
    logger = WandbLogger(project="tsp-old", name="TSP_Training")
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
    for problem in data:
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

if __name__ == "__main__":
    data = generate_data_by_tsplib()
    test(data)
    train_with_rl4co(data)

