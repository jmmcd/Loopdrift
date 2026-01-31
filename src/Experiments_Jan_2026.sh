# python experiments_qutrit.py --parameter-sweep --steps 201 --num-runs 30 --sweep-output-dir ../results/QutritWalkSweepResults --seed 42
# python classical_walk.py --steps 200 --num-walks 30 --output ../results/ClassicalWalkResults --seed 42
# python visualize_tonnetz.py --output qutrit_walk_up_LPR_0.pdf ../results/QutritWalkSweepResults/exp1_order_LPR_run01.csv
# python visualize_tonnetz.py --output qutrit_walk_up_PRL_0.pdf ../results/QutritWalkSweepResults/exp1_order_PRL_run01.csv
# python visualize_tonnetz.py --output qutrit_walk_up_RLP_0.pdf ../results/QutritWalkSweepResults/exp1_order_RLP_run01.csv

python visualize_tonnetz.py --output qutrit_walk_up_LPR_0.pdf ../results/QutritWalkSweepResults/exp1_order_LPR_run01.csv
python visualize_tonnetz.py --output qutrit_walk_up_LPR_1.pdf ../results/QutritWalkSweepResults/exp1_order_LPR_run02.csv
python visualize_tonnetz.py --output qutrit_walk_up_LPR_2.pdf ../results/QutritWalkSweepResults/exp1_order_LPR_run03.csv
python visualize_tonnetz.py --output qutrit_walk_up_LPR_3.pdf ../results/QutritWalkSweepResults/exp1_order_LPR_run04.csv
python visualize_tonnetz.py --output qutrit_walk_up_LPR_4.pdf ../results/QutritWalkSweepResults/exp1_order_LPR_run05.csv


# python visualize_tonnetz.py --output classical_walk_1.pdf ../results/ClassicalWalkResults/ClassicalWalkResults_1
# python visualize_tonnetz.py --output classical_walk_2.pdf ../results/ClassicalWalkResults/ClassicalWalkResults_2
# python visualize_tonnetz.py --output classical_walk_3.pdf ../results/ClassicalWalkResults/ClassicalWalkResults_3
# python visualize_tonnetz.py --output classical_walk_4.pdf ../results/ClassicalWalkResults/ClassicalWalkResults_4
# python visualize_tonnetz.py --output classical_walk_5.pdf ../results/ClassicalWalkResults/ClassicalWalkResults_5