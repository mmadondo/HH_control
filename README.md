# Learning Control Policies of Hodgkin-Huxley Neuronal Dynamics

We present a neural network approach for closed-loop deep brain stimulation (DBS). We cast the problem of finding an optimal neurostimulation strategy as a control problem. In this setting, control policies aim to optimize therapeutic outcomes by tailoring the parameters of a DBS system, typically via electrical stimulation, in real time based on the patient's ongoing neuronal activity. We approximate the value function offline using a neural network to enable generating controls (stimuli) in real time via the feedback form. The neuronal activity is characterized by a nonlinear, stiff system of differential equations as dictated by the Hodgkin-Huxley model. Our training process leverages the relationship between Pontryagin's maximum principle and Hamilton-Jacobi-Bellman equations to update the value function estimates simultaneously. Our numerical experiments illustrate the accuracy of our approach for out-of-distribution samples and the robustness to moderate shocks and disturbances in the system.


## Getting Started
**NOTE**: All commands in this document should run in the commandline/terminal unless stated otherwise.

### Local Solution Method
**NOTE**: Requires MATLAB. Run the main file, which implements the all-at-once [Interior Point Method](https://doi.org/10.1080/10556788.2013.858156) (IPM):
```commandline
>> run_local_solution
```
This will generate several plots for both normal and pathological activity. Navigate to the `../experiments/local_solution/` folder to see plots.

### Semi-Global Solution Method

1. Create a virtual environment:
```commandline
# setup virtual environment
conda env create -f dbs_env.yml

# load virtual environment
conda activate dbs_env
```

2. Check that you can run the DBS problem file:
```commandline
python DBSProblem.py
```
**RESULTS:**  navigate to the `../experiments/oc/run` folder to see plots.

3. Train NN-based controller
    - With default parameters:    
    ```commandline
        python train.py
    ```
    - With specific settings:
    ```commandline
        python train.py --optim adam --n_iters 500
    ```

    **RESULTS:**  navigate to the `../experiments/oc/run` folder to see plots.

4. Evaluate NN-based controller
    ```commandline
        python eval.py
    ```

    **RESULTS:**  navigate to the `../experiments/oc/eval` folder to see plots.

## Citing this work
If you found any part of this project helpful in your work, please cite as

```
@article{madondo2023learning,
  title={Learning Control Policies of Hodgkin-Huxley Neuronal Dynamics},
  author={Madondo, Malvern and Verma, Deepanshu and Ruthotto, Lars and Yong, Nicholas Au},
  journal={arXiv preprint arXiv:2311.07563},
  year={2023}
}
```