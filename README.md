# neat-braitenberg-vrep

Evolution of a Braitenberg like vehicle using NEAT and VREP.

### Project

The goal of the project is to showcase how to evolve Braitenberg like agents in parallel that could navigate within a specific environment without hitting obstacles using neuroevolution, particularly `neat-python`. The algorithm maintains a population of neural networks which is subjected to natural selection and mutation. In this work `V-REP` robotic simulator is used to create and simulate the environment. It exposes a remote API that allows controlling the simulation from the external client-side application - `robot.py`
robot controll module.

`neat-python` library implements `parallel.ParallelEvaluator` however it had to be extended in order to be able use it properly with `V-REP`. Each worker has to connect to an appropriate `V-REP` instance.

### How To Run?

Install `V-REP` and change the `vrep_abspath` in `neuroevolution.py` vrep absolute path to `V-REP` accordingly. Optional, configure the neat algorithm in `config.ini` and evolutionary params in `settings.py`.

```
# install dependencies
pipenv install

# run neuroevolution
pipenv run python neuroevolution.py


Neuroevolutionary program started!

 ****** Running generation 0 ******

thread_id = Worker Thread #0 client_id = 0
thread_id = Worker Thread #1 client_id = 1
thread_id = Worker Thread #2 client_id = 2
thread_id = Worker Thread #3 client_id = 3
Worker Thread #0 genome_id: 1 fitness: 53.822867
Worker Thread #3 genome_id: 4 fitness: 35.223197
Worker Thread #1 genome_id: 2 fitness: 12.990576
Worker Thread #2 genome_id: 3 fitness: 53.216213
...

```

![alt text](/static/vrep.instances.png "Spawing 4 V-REP Instances")
