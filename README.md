# DQN-for-solving-the-N-Puzzle
Using a deep Q-network to train an agent to solve the N-puzzle


Author: Dudley Spence
Title: USING DEEP REINFORCEMENT LEARNING TO CREATE AN AGENT THAT CAN LEARN TO SOLVE THE N-PUZZLE

This is a project using deep reinforcement learning specifically a deep q-network (DQN) to learn to solve the N-puzzle. 

The network can be trained to solve any N-puzzle but the 8-puzzle and 15-puzzle are suggested.


The project contains 6 files:

Training_Loop.py
Environment.py
Replay_Buffer.py
DQNAgent.py
Evaluate.py
Puzzle_Solver.py


preliminary installation commands:

pip install tensorflow
pip install keras
pip install tqdm
pip install matplotlib
pip install numpy
pip install collections
pip install python


Training Guide:

•	“N” is the puzzle size you are wanting to train on so for the 8-puzzle, N=8. Difficulty is an estimate of the 
	optimum number of moves for the puzzles the agent will be using for training. 

•	Lower difficulty will result in faster training as the puzzles are easier but also much faster to generate. Bear 
	in mind if the difficulty is set higher than the maximum difficulty for that puzzle the puzzle boards will never 
	be generated and training won’t progress.

•	There are already pre-trained saves included in the software, one for the 8-Puzzle and one for the 15-puzzle should
	you wish to use the pre-trained networks and continue the training.

•	To begin training: open the command prompt and run the command…

	python Training_Loop.py

•	From there you will be asked to input the N value for the puzzles you would like to train using. For example, for 
	the 8-puzzle you should input 8.

•	You will be then asked to provide a training difficulty. The difficulty should not be set higher than 21 for the 
	8-puzzle and 50 for the 15-puzzle. I would advise a training difficulty of 15 for the 8-puzzle.

•	Finally, you will be asked if you wish to restart training or continue from the last training checkpoint. You must 
	type any of “yes”, “no”, “y” or “n”. If you select yes to restart training, all previous training data for that 
	puzzle size will be deleted.

•	The training should then begin and will automatically save the network parameters for future use.

•	The first time the network is trained I would advise not interrupting training until epsilon decay has finished and 
	the epsilon value, stated in the in-situ evaluations, has reach the final epsilon value (default is 0.3).



To make changes to the other hyper-parameters of the DQN, open the Training_Loop.py file and at the bottom of the script choose 
the desired network hyper-parameters.




Network Evaluation Guide:

•	To evaluate the network training simply run the command… 

	python Evaluate.py

•	Input the size N of the puzzle you would like the network to use for evaluation. 






Use the trained network to solve puzzles of your choice:

•	Firstly, you will need to convert the puzzle you wish to solve into the correct format.

•	Figure 8 shows an example 8-puzzle that can be written as a list of length N. e.g. figure 8 = 123456780

•	Once you have converted your puzzle into the correct format. On the command line run the command…

	python Puzzle_Solver.py 

•	Then type in the reformatted puzzle and press enter to begin attempting to solve the puzzle.



