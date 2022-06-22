# grid2gif

Generates animated gif using [grid2game](https://github.com/BDonnot/grid2game).

## Usage
You only need to provide a [grid2op](https://github.com/rte-france/Grid2Op) environment and a list of actions:
```python
g2g = Grid2Gif("educ_case14_storage", test=True)
agent = DoNothingAgent(g2g.env.action_space)
g2g.mk_gif(agent, path="images/DN_agent.gif", max_iter=288)
```
Then, grid2gif will create a gif where each frame is an observation of the environment
after executing an action.

## Example
#### Do Nothing Agent
![](images/dn_agent.gif)

#### Expert agent
![](images/expert_agent.gif)
