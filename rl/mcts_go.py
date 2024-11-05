class Node:
    def __init__(self, state, parent=None, prior=0):
        self.state = state  # 当前棋盘状态
        self.parent = parent  # 父节点
        self.children = {}  # 子节点，键为动作，值为节点
        self.visit_count = 0  # 被访问次数
        self.value_sum = 0  # 值的累计和，用于计算均值
        self.prior = prior  # 通过策略网络给出的先验概率
        self.is_expanded = False  # 节点是否已经扩展
        self.is_terminal = False  # 是否为终止状态（如胜利或平局）

    @property
    def value(self):
        # 返回该节点的平均值
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, legal_moves, priors):
        # 扩展节点，创建其子节点
        for move in legal_moves:
            if move not in self.children:
                self.children[move] = Node(state=next_state(self.state, move),
                                           parent=self, prior=priors[move])
        self.is_expanded = True


class MCTS:
    def __init__(self, policy_network, value_network):
        self.policy_network = policy_network  # 策略网络
        self.value_network = value_network  # 价值网络

    def search(self, root):
        # 执行一次搜索
        node = root

        # 1. SELECTION: 使用 UCT 公式选择最优子节点，直到到达未扩展节点
        while node.is_expanded and not node.is_terminal:
            node = self.select_child(node)

        # 2. EXPANSION: 如果未扩展且不是终止节点，扩展该节点
        if not node.is_terminal:
            self.expand_node(node)

        # 3. SIMULATION: 使用价值网络估计当前节点的价值
        value = self.simulate(node)

        # 4. BACKPROPAGATION: 从当前节点向上回传模拟结果
        self.backpropagate(node, value)

    def select_child(self, node):
        # 使用 UCT 公式选择最佳子节点， current-->child   take the factor of 访问次数、经过时的value
        best_score = -float('inf')
        best_child = None

        for move, child in node.children.items():
            uct_score = self.uct(child, node)
            if uct_score > best_score:
                best_score = uct_score
                best_child = child

        return best_child

    def uct(self, child, parent):
        # Upper Confidence Bound applied to Trees (UCT) 公式
        c = 1.0  # 探索的权衡系数
        exploration_term = c * child.prior * (parent.visit_count ** 0.5 / (1 + child.visit_count))
        return child.value + exploration_term

    def expand_node(self, node):
        # 调用策略网络预测当前节点的策略概率（先验概率）
        legal_moves = get_legal_moves(node.state)
        priors = self.policy_network.predict(node.state)  # 返回每个合法动作的概率

        # 扩展节点
        node.expand(legal_moves, priors)

        # 检查当前节点是否为终止状态（胜利、失败或平局）
        node.is_terminal = is_terminal(node.state)

    def simulate(self, node):
        # 使用价值网络评估该节点的价值
        if node.is_terminal:
            return get_terminal_value(node.state)  # 终局直接返回结果（胜负）

        # 否则使用价值网络预测
        return self.value_network.predict(node.state)

    def backpropagate(self, node, value):
        # 回传过程，从叶子节点到根节点，更新所有节点的值和访问次数
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent


# MCTS的主循环
def MCTS_main(root, num_simulations, policy_network, value_network):
    mcts = MCTS(policy_network, value_network)

    # 进行多次模拟，构建蒙特卡洛树
    for _ in range(num_simulations):
        mcts.search(root)

    # 选择访问次数最多的动作（即最优动作）
    best_move = max(root.children.items(), key=lambda child: child[1].visit_count)[0]

    return best_move


# 模拟运行中的辅助函数
def next_state(state, move):
    """根据当前状态和动作，返回下一个状态"""
    # 实现棋盘状态转换逻辑
    pass


def get_legal_moves(state):
    """返回当前状态下所有合法的动作"""
    # 实现获取合法动作的逻辑
    pass


def is_terminal(state):
    """判断该状态是否为终局状态"""
    # 实现终局判断逻辑（胜利、失败或平局）
    pass


def get_terminal_value(state):
    """返回终局状态的值，1代表胜利，-1代表失败，0代表平局"""
    # 实现终局的值返回逻辑
    pass
