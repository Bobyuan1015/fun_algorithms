import torch
from tensordict import TensorDict
import glob
import requests, tarfile, os, gzip, shutil
from tqdm.auto import tqdm
from tsplib95.loaders import load_problem, load_solution
from confs.path_conf import system_data_dir

def download_and_extract_tsplib(url, directory="tsplib", delete_after_unzip=True):
    os.makedirs(directory, exist_ok=True)
    if  glob.glob(os.path.join(directory, "*.tsp")):
        print(f'tsp files exist， do not download')
        return


    # Download with progress bar
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(f"tsplib.tar.gz", 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                pbar.update(len(chunk))

    # Extract tar.gz
    with tarfile.open("tsplib.tar.gz", 'r:gz') as tar:
        tar.extractall(directory)

    # Decompress .gz files inside directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".gz"):
                path = os.path.join(root, file)
                with gzip.open(path, 'rb') as f_in, open(path[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(path)

    if delete_after_unzip:
        os.remove("tsplib.tar.gz")


# Utils function: we will normalize the coordinates of the VRP instances
def normalize_coord(coord: torch.Tensor) -> torch.Tensor:
    x, y = coord[:, 0], coord[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_scaled = (x - x_min) / (x_max - x_min)
    y_scaled = (y - y_min) / (y_max - y.min())
    coord_scaled = torch.stack([x_scaled, y_scaled], dim=1)
    return coord_scaled

def tsplib_to_td(problem, normalize=True):
    coords = torch.tensor(problem['node_coords']).float()
    coords_norm = normalize_coord(coords) if normalize else coords
    td = TensorDict({
        'locs': coords_norm,
    })
    td = td[None]  # add batch dimension, in this case just 1
    return td

def generate_data_by_tsplib():
    tsplib_dir = f'{system_data_dir}/tsp/tsplib/'  # modify this to the directory of your prepared files
    # Download and extract all tsp files under tsplib directory
    download_and_extract_tsplib("http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz",
                                directory=tsplib_dir)

    # Load the problem from TSPLib

    files = os.listdir(tsplib_dir)
    problem_files_full = [file for file in files if file.endswith('.tsp')]

    # Load the optimal solution files from TSPLib
    solution_files = [file for file in files if file.endswith('.opt.tour')]

    problems = []
    # Load only problems with solution files
    for sol_file in solution_files:
        prob_file = sol_file.replace('.opt.tour', '.tsp')
        problem = load_problem(os.path.join(tsplib_dir, prob_file))

        # NOTE: in some problem files (e.g. hk48), the node coordinates are not available
        # we temporarily skip these problems
        if not len(problem.node_coords):
            continue

        node_coords = torch.tensor([v for v in problem.node_coords.values()])
        solution = load_solution(os.path.join(tsplib_dir, sol_file))

        problems.append({
            "name": sol_file.replace('.opt.tour', ''),
            "node_coords": node_coords,
            "solution": solution.tours[0],
            "dimension": problem.dimension
        })

    # order by dimension
    problems = sorted(problems, key=lambda x: x['dimension'])
    return problems


if __name__ == "__main__":
    data = generate_data_by_tsplib()
    print()
#     {'dimension': 16, 'name': 'ulysses16', 'node_coords': tensor([[38.2400, 20.4200],
#         [39.5700, 26.1500],
#         [40.5600, 25.3200],
#         [36.2600, 23.1200],
#         [33.48...700, 15.1300],
#         [38.1500, 15.3500],
#         [37.5100, 15.1700],
#         [35.4900, 14.3200],
#         [39.3600, 19.5600]]), 'solution': [1, 14, 13, 12, 7, 6, 15, 5, 11, 9, 10, 16, 3, 2, 4, 8]}



