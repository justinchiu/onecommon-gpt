import numpy as np


def get_configs(size):
    return np.unpackbits(np.arange(size, dtype=np.uint8)[:, None], axis=1)[..., 1:]


def config_to_int(configs):
    return np.packbits(
        np.flip(configs, -1),
        axis=-1,
        bitorder="little",
    )[..., 0]


if __name__ == "__main__":
    print(config_to_int(get_configs(128)))
