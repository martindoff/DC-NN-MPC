# DC-NN-MPC
Computationally tractable learning-based nonlinear tube MPC using difference of convex neural network dynamic approximation 


<br />
<p align="center">
   <img src="https://github.com/martindoff/DC-NN-MPC/blob/main/plot/NN.png" alt="Logo" width="400" height="300">
  <p align="center">
   Input-Convex Neural Network architecture whose kernel weights $\Theta$ are constrained to be non-negative and activation functions are convex non-decreasing (e.g. ReLU) 
    <br />  
  </p>
</p>

<!-- ABOUT THE PROJECT -->
## About The Project

Learning-based robust tube-based MPC of dynamic systems approximated by difference-of-convex (DC) Neural Network models. Successive linearisations of the learned dynamics in DC form are performed to express the MPC scheme as a sequence of convex programs.  Convexity in the learned dynamics is exploited to bound successive linearisation errors tightly and treat them as bounded disturbances in a robust MPC scheme. Application to a PVTOL aircraft model. This novel computationally tractabe tube-based MPC algorithm is presented in the paper "Computationally tractable nonlinear robust MPC via DC programming" by Martin Doff-Sotta and Mark Cannon. It is an extension of our previous work [here](https://github.com/martindoff/DC-TMPC) and [here](https://ora.ox.ac.uk/objects/uuid:a3a0130b-5387-44b3-97ae-1c9795b91a42/download_file?safe_filename=Doff-Sotta_and_Cannon_2022_Difference_of_convex.pdf&file_format=application%2Fpdf&type_of_work=Conference+item)

### DC Neural Network 
In order to derive a sequence of convex programs, the dynamics are learned in DC form using [DC Neural Network models](https://github.com/martindoff/DC-Deep-Neural-Network). These are obtained by stacking two [Input-Convex Neural Networks](http://proceedings.mlr.press/v70/amos17b/amos17b.pdf) models whose kernel weights are constrained to be non-negative and activation functions are convex non-decreasing (e.g. ReLU). 

### Built With

* Python 3
* CVXPY
* MOSEK
* Keras

### Prerequisites

You need to install the following:
* numpy
* matplotlib
* pydot
* [tensorflow / keras](https://keras.io/getting_started/)
* [CVXPY](https://www.cvxpy.org/install/)
* [MOSEK](https://docs.mosek.com/10.2/install/installation.html)

In a terminal command line, run the following to install all modules at once

   ```sh
   pip3 install numpy matplotlib tensorflow cvxpy Mosek 
   ```
Using solver `Mosek` requires a license. Follow the instructions [here](https://docs.mosek.com/10.2/install/installation.html) to obtain it and position it correctly in your system.

### Running the code

1. To clone the repository in the current location, run the following in the terminal
   ```sh
   git clone https://github.com/martindoff/DC-NN-MPC.git
   ```
2. Go to directory 
   ```sh
   cd DC-NN-MPC
   ```
   
3. Run the program
   ```python
   python3 main.py
   ```
### Options 
   
1. To load an existing model, set the `load` variable in `main.py` to `True`
```python
   load = True
   ``` 
   
   Set the variable to `False` if the model has to be (re)trained. 

2. The tube parameterisation can be chosen via the `set_param` variable in `main.py`. For a tube cross section parameterised by means of elementwise bounds, set

   ```python
   set_param = 'elem'
   ```
   For a tube parameterised by means of simplex sets:

   ```python
   set_param = 'splx'
   ```

### Results

The DC decomposition of the system dynamics $f = f_1 - f_2$, where $f_1, f_2$ are convex is illustrated below


<br />
<p align="center">
   <img src="https://github.com/martindoff/DC-NN-MPC/blob/main/DC.png" alt="Logo" width="600" height="300">
  <p align="center">
   DC decomposition with DC Neural Networks
    <br />  
  </p>
</p>

The closed-loop solution is presented below


<br />
<p align="center">
   <img src="https://github.com/martindoff/DC-NN-MPC/blob/main/tmpc_plot.png" alt="Logo" width="400" height="300">
  <p align="center">
   Closed-loop tube MPC solution
    <br />  
  </p>
</p>

