import numpy as np
import json


def generate(n_obs=20, max_v=2, map_size=200, max_size=10, min_size=3, starting_area=20):
    pos = np.random.uniform(starting_area, map_size, size=(n_obs, 2))
    v = np.random.uniform(-max_v, max_v, size=(n_obs, 2))
    size = np.random.uniform(min_size, max_size, size=n_obs)
    obs = []
    for i in range(n_obs):
        obs.append({
            "start_x": int(pos[i, 0]),
            "start_y": int(pos[i, 1]),
            "v_x": np.round(v[i, 0], 1),
            "v_y": np.round(v[i, 1], 1),
            "size": int(size[i]),
        })
    return obs


if __name__ == '__main__':
    obs = generate(map_size=100, max_size=5, min_size=3, max_v=1, n_obs=13)
    out = './config/obs_rich.json'
    with open(out, 'r') as f:
        data = json.load(f)
    with open(out, 'w') as f:
        data['obstacles'] = obs
        json.dump(data, f, indent=2)
    print("DONE")
