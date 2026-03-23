class Path_Hyperparameter:
    random_seed = 42

    # training hyper-parameter
    epochs: int = 300
    batch_size: int = 2
    inference_ratio = 2
    learning_rate: float = 1e-3
    factor = 0.1
    patience = 12
    warm_up_step = 1000
    weight_decay: float = 1e-3
    amp: bool = True
    load: str = None
    max_norm: float = 20.0

    # dataloader hyper-parameter
    num_workers: int = 4
    prefetch_factor: int = 2

    # evaluate and test hyper-parameter
    evaluate_epoch: int = 1
    evaluate_inteval: int = 1
    test_epoch: int = 30
    stage_epoch = [0, 0, 0, 0, 0]
    save_checkpoint: bool = True
    save_interval: int = 5
    save_best_model: bool = True

    # model hyper-parameter
    drop_path_rate = 0.2
    dims = 96
    depths = [2, 2, 9, 2]
    ssm_d_state = 16
    ssm_dt_rank = "auto"
    ssm_ratio = 2.0
    mlp_ratio = 4.0

    # data parameter
    image_size = 256
    downsample_raito = 1
    dataset_name = "deepglobe"
    root_dir = r"D:\project\pythonProject\Road_Identification\SAM2-UNet"
    image_dir_name = "data"
    label_dir_name = "seg"
    threshold: float = 0.5

    # output parameter
    log_dir = "./logs"
    checkpoint_dir = "./checkpoints"
    project_name = "deepglobe_road_binary"

    def state_dict(self):
        return {
            k: getattr(self, k)
            for k, _ in Path_Hyperparameter.__dict__.items()
            if not k.startswith("_")
        }


ph = Path_Hyperparameter()
