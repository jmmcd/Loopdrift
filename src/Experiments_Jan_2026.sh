python experiments_qutrit.py --parameter-sweep --steps 201 --num-runs 30 --sweep-output-dir ../results/QutritWalkSweepResults --seed 42
python classical_walk.py --steps 200 --num-walks 30 --output ../results/ClassicalWalkResults --seed 42
python visualize_tonnetz.py --output qutrit_walk_up_LPR_0.pdf ../results/QutritWalkSweepResults/exp1_order_LPR_run01.csv
python visualize_tonnetz.py --output qutrit_walk_up_PRL_0.pdf ../results/QutritWalkSweepResults/exp1_order_PRL_run01.csv
python visualize_tonnetz.py --output qutrit_walk_up_RLP_0.pdf ../results/QutritWalkSweepResults/exp1_order_RLP_run01.csv
python visualize_tonnetz.py --output classical_walk_0.pdf ../results/ClassicalWalkResults/ClassicalWalkResults_1