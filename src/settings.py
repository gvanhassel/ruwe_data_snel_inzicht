from pathlib import Path


root = Path(__file__).resolve().parent.parent
# print(root)
data_dir = root / "data"
tensor_dir = data_dir / "pickle"

class Settings:
    def __init__(
        self,
        root,
        model_dir,
        data_dir,
        tensor_dir,
        logger_dir,
    ):
        self.root = root
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.tensor_dir = tensor_dir
        self.logger_dir = logger_dir

projectsettings = Settings(
    root=root,
    model_dir=root / "results/model",
    data_dir=data_dir,
    tensor_dir=tensor_dir,
    logger_dir=root / "results/log",
)


