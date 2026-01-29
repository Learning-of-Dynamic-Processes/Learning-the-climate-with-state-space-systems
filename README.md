# Learning the climate of dynamical systems with state space systems

SOURCE
------
Code base comes from https://github.com/Learning-of-Dynamic-Processes/coldstart.git
"Data-driven cold starting of good reservoirs"
L. Grigoryeva et al., (2023).
*ArXiv* arXiv:2403.10325 
(https://arxiv.org/abs/2403.10325)

INSTALLATION
---------


Clone repository and install dependencies

    git clone https://github.com/Learning-of-Dynamic-Processes/coldstart.git

USAGE
---------

- Run 
This python package contains functions to initialize reservoir computers for two examples, the Brusselator and the Lorenz system.

- Run `brusselator_optuna.py` to find a good set of reservoir hyperparamerters for the Brusselator system. Paste the best set of parameters into `brusselator/config.py`.
- Run `brusselator_train.py` to create a reservoir using the parameters in `brusselator/config.py`.
- Use `brusselator.ipynb` to create the cold start map and the Brusselator figures shown in the paper.

Likewise, for the Lorenz system:

- Run `lorenz_optuna.py` to find a good set of reservoir hyperparamerters for the Lorenz system. Paste the best set of parameters into `lorenz/config.py`.
- Run `lorenz_train.py` to create a reservoir using the parameters in `lorenz/config.py`.
- Use `lorenz.ipynb` to create the cold start map and the Lorenz figures shown in the paper.

To recreate results from section 4.1 of the article, use the code in `coldstart_fig_9/`.
