import pprint
from pathlib import Path
import sys

from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf


def main() -> None:
    """Compose and run the A2C data pipeline using Hydra configuration."""
    project_root = Path(__file__).resolve().parent.parent
    sys.path.append(str(project_root))
    with initialize(config_path="../conf/data", version_base=None):
        cfg = compose(
            config_name="a2c_data",
            overrides=[
                "symbols=[BTC,ETH]",
                "time_freq=1d",
                "feature_engineering.technical_indicators.strategies=[momentum]",
                "feature_engineering.quantstats_features.enabled=false",
            ],
        )

        print("=" * 80)
        print("HYDRA CONFIGURATION CONSTRUCTION")
        print("=" * 80)
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        pprint.pprint(config_dict)

        # Instantiate pipeline components from the Hydra config
        data_handler = instantiate(cfg)

    # Run the data pipeline and display output
    df = data_handler.get_data()
    print(df.head())
    print("Data shape:", df.shape)


if __name__ == "__main__":
    main()
