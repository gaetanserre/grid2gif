from grid2game.plot.plot_grid import PlotGrids
import grid2op
import gif
from tqdm import tqdm
from grid2op.Agent import DoNothingAgent
from lightsim2grid import LightSimBackend
from PPO_agent.my_baseline import make_agent

class Grid2Gif:
  def __init__(self, env_name, **kwargs):
    self.env = grid2op.make(env_name, **kwargs)
    self.env.seed(1)
    self.plot_grids = PlotGrids(self.env.observation_space)
  
  def set_line_info(self, line_info):
    """
    This function adds a legend along the powerlines

    Parameters
    ----------
    line_info: `str`
              "rho" for capacity\\
              "name" for name\\
              "th_lim" for thermal limit\\
              "cooldown" for cooldown\\
              "timestep_overflow" for # step overflow\\
              "none" for `None` (default)
    """
    self.plot_grids.line_info = line_info
  
  def set_load_info(self, load_info: str):
    """
    This function adds a legend along the loads

    Parameters
    ----------
    load_info: `str`
              "p" for MW\\
              "v" for kV\\
              "q" for MVAr\\
              "name" for name\\
              "none" for `None` (default)
    """
    self.plot_grids.load_info = load_info
  
  def set_gen_info(self, gen_info: str):
    """
    This function adds a legend along the generators

    Parameters
    ----------
    gen_info: `str`
              "p" for MW\\
              "v" for kV\\
              "q" for MVAr\\
              "ramp_down" for ramp_down\\
              "ramp_up" for ramp_up\\
              "target_dispatch" for target_dispatch\\
              "actual_dispatch" for actual_dispatch\\
              "type" for generator type\\
              "name" for name\\
              "none" for `None` (default)
    """
    self.plot_grids.gen_info = gen_info
  
  def set_storage_info(self, storage_info: str):
    """
    This function adds a legend along the storages

    Parameters
    ----------
    storage_info: `str`
              "p" for MW\\
              "MWh" for MWh\\
              "name" for name\\
              "none" for `None` (default)
    """
    self.plot_grids.storage_info = storage_info
  
  def _get_fig(self):
    obs = self.env.get_obs()
    self.plot_grids.init_figs(obs, obs)
    return self.plot_grids.figure_rt
  
  def plot(self):
    self._get_fig().show()
  
  def mk_gif(self, agent, path: str, max_iter: int, duration=100, **kwargs):
    """
    This function generates an animated gif using an agent

    Parameters
    ----------
    agent: :class:`Agent`
           a gym-like agent
    path: `str`
        the path where to store the animated gif
    max_iter: `int`
        the maximum number of steps to compute
    duration: `int`
            how much time in ms one frame is displayed
    """
    print(self.env.chronics_handler.real_data.get_id())
    @gif.frame
    def compute_fig():
      return self._get_fig()
    
    frames = []
    done = False
    obs = self.env.get_obs()
    reward = None

    battery1 = []
    battery2 = []

    for _ in tqdm(range(max_iter)):
      action = agent.act(obs, reward, done)
      obs, reward, done, _ = self.env.step(action)
      if done:
        break
      frames.append(compute_fig())
      battery1.append(obs.storage_power[0])
      battery2.append(obs.storage_power[1])
    
    for _ in range(20):
      frames.append(frames[-1])

    gif.save(frames, path, duration=duration, **kwargs)

    return battery1, battery2

if __name__ == "__main__":
  g2g = Grid2Gif("input_data_local", backend=LightSimBackend())
  g2g.set_line_info("rho")
  g2g.set_storage_info("MWh")
  g2g.set_gen_info("p")

  agent = make_agent(g2g.env, "PPO_agent")
  g2g.mk_gif(agent, path="images/RL_agent.gif", max_iter=2017)