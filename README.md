## Todo list

- [x] Debug the different versions of the TSP code developed in three stages
- [x] Optimize the environment definitions in the code, such as the reward definition
- [x] Standardize different agents
- [ ] Implement hierarchical reinforcement learning
- [ ] Think about innovative points in the paper
- [ ] Organize experiment results  

## Issues shot
pip install pandas numpy tensorflow tqdm imageio tqdm imageio matplotlib scipy  
pip install torch torchvision torchaudio   -i  https://pypi.tuna.tsinghua.edu.cn/simple
pip install gym[classic_control] -i https://pypi.tuna.tsinghua.edu.cn/simple

---------
cd ~/anaconda3/lib
mv libstdc++.so libstdc++.so-bk
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ./
ln -s libstdc++.so.6 libstdc++.so
ln -s libstdc++.so.6 libstdc++.so.6.0.19
