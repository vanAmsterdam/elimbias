# Define a structural causal model

The structural causal model used in the publication is printed below.
The setup is easy:

1. Define the background noise variables (i.e. variables not caused by any other variable)

   * **variable** is the internal name
   * **label** is a longer name; when these corerspond to *measurements* of the images (like here size and variance), they will be used to sample images
   * **type** noise vs dependent
   * **distribution** where to draw the variable from; for the dependent variables, this is the *conditional* distribution
   * **param_1** and **param_2** are the canonical parameters for the distribution (e.g. location and scale for Normal)

2. Define the relationships between the noise variables and the dependent variables, using b_... columns to define coefficients from the noise variable to ... in a linear model.



| variable | label     | type      | distribution | variable_model | param_1 | param_2 | b_x | b_t   | b_y |
|----------|-----------|-----------|--------------|----------------|---------|---------|-----|-------|-----|
| u1       | u1        | noise     | Normal       |                | 0       | 0.7071  | 1   | 0     | -2  |
| u2       | u2        | noise     | Normal       |                | 0       | 0.7071  | -1  | 1.828 | 0   |
| z        | variance  | noise     | Normal       |                | 0       | 1       | 0   | 0     | -1  |
| n_x      | noise_x   | noise     | Normal       |                | 0       | 0.05    | 1   | 0     | 0   |
| n_t      | noise_t   | noise     | Normal       |                | 0       | 0.05    | 0   | 1     | 0   |
| n_y      | noise_y   | noise     | Normal       |                | 0       | 0.05    | 0   | 0     | 1   |
| x        | size      | dependent | Normal       | Linear         | 0       |         | 0   | 0     | 0   |
| t        | treatment | dependent | Bernoulli    | Logistic       | -0.5    |         | 0   | 0     | 1   |
| y        | survival  | dependent | Normal       | Linear         | -0.5    |         | 0   | 0     | 0   |