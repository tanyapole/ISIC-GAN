exec bash
conda activate nduginets
cd ~/master-diploma/
tmux new-session -d -s jupyter-1234 'jupyter notebook --no-browser --port 1234'
tmux ls