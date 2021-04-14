# Ant Colony Optimization - CPU optimization

Implementation of Ant Colony algorithm for the purpose of image edge detection.

Algorithm stages:
1. Initialization:
* Setting the parameters,
* Assigning the initial pheromone values.

2. Schedule activities:
    1. Construct ant solutions from a finite set of solution components (fully connected graph that represents the problem to be solved).
    Ants traverse the graph until each has made the target number of construction steps. The solution construction process starts with an empty partial solution, which is extended at each construction step by adding a solution component. The solution component is chosen from a set of nodes neighboring the current position in the graph. The choice of solution components is done probabilistically. The exact decision rule for choosing the solution components varies across different ACO variants. The most common decision rule is the one used in the original AS.
    2. Do daemon actions: Once solutions have been constructed, there might be a need to perform additional actions before updating the pheromone values. Such actions, usually called daemon actions, are those that cannot be performed by a single ant. Normally, these are problem specific or centralized actions to improve the solution or search process.
    3. Update pheromones: After each construction process and after the daemon actions have been performed, the pheromone values are updated. The goal of the pheromone update is to increase the pheromone values associated with good solutions and decrease those associated with bad ones. This is normally done by decreasing all the pheromone values (evaporation) and increasing the pheromone values associated with the good solutions (deposit).