# README

## Project Description

This project implements a 2D Drone controlled by a NN trained with a deep reinforcement learning algorithm. 

The drone is repeatedly exposed to disturbances, which are represented by abrupt changes in location and speed. The drone's goal is to return to the origin of the coordinates and minimize its speed.

## How to run the demonstration?

### Step 1: Recreate conda environment
Navigate to root of directory "two_link_robot" and execute this command:

conda env create -f environment.yml

### Step 2: Activate conda environment

conda activate myenv_two_link_robot

### Step 3: Run
To demonstrate the best saved model, run: 

python3 perform.py

The model has already been trained.

## Author
Konstantin König, Dec 31st 2024
konstantin.koenig@aol.com