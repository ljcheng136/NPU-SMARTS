# Example : Conservative Q-Learning
1. An example solution for Track-2 offline learning based model development is presented here. This example uses Convervative Q-learning (CQL) method from [d3rlpy](https://github.com/takuseno/d3rlpy) offline RL library.
    + **The policy here has not yet been trained to fully solve the task environments.**
    + **This example is only meant to demonstrate one potential method of developing an offline learning using Waymo dataset. Here, any offline learning method may be used to develop the policy.**
1. Additional example solutions to Track2 developed using offline learning methods are available [here](https://github.com/smarts-project/smarts-project.rl/tree/master/neurips2022).

## Setup
+ Use `python3.8` to develop your model.
    ```bash
    $ cd <path>/SMARTS/competition/track2/train
    $ python3.8 -m venv ./.venv
    $ source ./.venv/bin/activate
    $ pip install --upgrade pip
    $ pip install -e .
    ```

## Notes on the used Observation, Action, and Reward
+ Observations: A 3-channel RGB birds eye-view image plus an extended channel containing the location of the goal, is used as the input for to the model. Hence, the model input is of the form (4, 256, 256).
+ Actions: The policy outputs dx, dy, and dh, which are the delta values per step in x, y direction and heading for the ego vehicle in its birds eye-view image coordinate system. Since dx and dy can not be directly obtained from SMARTS observation, we have to get displacement change in global coordinate first and use a rotation matrix with respect to the heading to get dx and dy.
+ Rewards: The reward uses the default reward in SMARTS which is the distance travelled per step plus an extra reward for reaching the goal. Since there is no "goal" concept in the training set, we use the last point of each trajectory as the goal position during training. 

## Train
1. Train
    ```bash
    $ cd <path>/SMARTS/competition/track2/train
    $ python3.8 train.py --input_dir=<path_to_data> --ouput_dir=<path_to_saved_model>
    ```

    The default value for `input_dir` and `output_dir` are `"/SMARTS/competition/offline_dataset/"` and `/SMARTS/competition/track2/submission/`.
 1. Since we can not load too many images in the training dataset at each time, we are training using data in one scenario at each time. After the end of each training iteration, the model will be saved in `<path>/SMARTS/competition/track2/train/d3rlpy_logs/<scenario_index>`. The next trainig iteration will keep training on the latest trained model. And at the end of the training, the last model will be copied to `/SMARTS/competition/track2/submission/`.