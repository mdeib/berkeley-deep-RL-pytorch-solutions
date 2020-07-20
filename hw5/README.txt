1) The code structure for this homeowrk was heavily modified in order to match the structure of the previous three homeworks. 
To this end the PDF does not give the most accurate location instructions but should still be referred to for questions and guidance.
The logging procedure in particular was changed to match the previous assignments.

2) Code:

Code to look at:

- scripts/train_ac_exploration_f18.py
- envs/pointmass.py
- infrastructure/rl_trainer.py (Has been changed for this homework)
- infrastructure/utils.py (Has been changed foir this homework)

Code to fill in as part of HW:

- agents/ac_agent.py (new Exploratory_ACAgent class added)
- exploration/exploration.py
- exploration/density_model.py

3) Commands to run for each problem:

##########################
### P1 Hist PointMass  ###
##########################

python cs285/scripts/train_ac_exploration_f18.py PointMass-v0 -n 100 -b 1000 -e 3 --density_model none -s 8 --exp_name PM_bc0_s8
python cs285/scripts/train_ac_exploration_f18.py PointMass-v0 -n 100 -b 1000 -e 3 --density_model hist -bc 0.01 -s 8 --exp_name PM_hist_bc0.01_s8

##########################
###  P2 RBF PointMass  ###
##########################

python cs285/scripts/train_ac_exploration_f18.py PointMass-v0 -n 100 -b 1000 -e 3 --density_model rbf -bc 0.01 -s 8 -sig 0.2 --exp_name PM_rbf_bc0.01_s8_sig0.2

##########################
###  P3 EX2 PointMass  ###
##########################

python cs285/scripts/train_ac_exploration_f18.py PointMass-v0 -n 100 -b 1000 -e 3 --density_model ex2 -s 8 -bc 0.05 -kl 0.1 -dlr 0.001 -dh 8 --exp_name PM_ex2_s8_bc0.05_kl0.1_dlr0.001_dh8

###########################
###    P4 HalfCheetah   ###
###########################

python cs285/scripts/train_ac_exploration_f18.py sparse-cheetah-cs285-v1 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 --density_model none --exp_name HC_bc0
python cs285/scripts/train_ac_exploration_f18.py sparse-cheetah-cs285-v1 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 --density_model ex2 -bc 0.001 -kl 0.1 -dlr 0.005 -dti 1000 --exp_name HC_bc0.001_kl0.1_dlr0.005_dti1000
python cs285/scripts/train_ac_exploration_f18.py sparse-cheetah-cs285-v1 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 --density_model ex2 -bc 0.0001 -kl 0.1 -dlr 0.005 -dti 10000 --exp_name HC_bc0.0001_kl0.1_dlr0.005_dti10000

4) Visualize saved tensorboard event file:

$ cd cs285/data/<your_log_dir>
$ tensorboard --logdir .

Then, navigate to shown url to see scalar summaries as plots (in 'scalar' tab), as well as videos (in 'images' tab)