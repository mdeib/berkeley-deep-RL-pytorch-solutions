from gym.envs.registration import register
register(
    id='sparse-cheetah-cs285-v1',
    entry_point='cs285.envs.sparse_half_cheetah:HalfCheetahEnv',
    max_episode_steps=1000,
)
from cs285.envs.sparse_half_cheetah import HalfCheetahEnv
