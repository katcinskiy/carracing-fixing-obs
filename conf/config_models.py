from dataclasses import dataclass


@dataclass
class AlgoConfig:
    policy: str
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    verbose: int
    tensorboard_log: str


@dataclass
class TrainConfig:
    total_timesteps: int
    n_envs: int
    tensorboard_log_dir: str

    brake_penalty_reward: float
    brake_penalty_th: float

    steer_penalty_actions_len: int
    steer_penalty_th: int
    steer_penalty_reward: float


@dataclass
class EvalConfig:
    video_save_path: str
    model_save_path: str
    eval_freq: int
    n_eval_episodes: int
    deterministic: bool


@dataclass
class WrapperConfig:
    use_custom: bool
    trajectory_color_max_speed: int
    trajectory_thickness: int
    draw_for_last: int
    use_frame_stack: bool
    frame_stack_count: int


@dataclass
class EnvConfig:
    wrapper: WrapperConfig
    max_episode_steps: int
    domain_randomize: bool
    continuous: bool
    obs_width: int
    obs_height: int


@dataclass
class AppConfig:
    name: str
    log_dir: str
    env: EnvConfig
    train: TrainConfig
    eval: EvalConfig

    algo_name: str
    algo_cfg: AlgoConfig
