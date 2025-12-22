import csv
import numpy as np
import os

filename=os.getcwd()

csv_path = os.path.join(filename, "results.csv")

with np.load(os.path.join(filename, "evaluations.npz")) as data:
    timesteps = data['timesteps']
    results = data['results']

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "result"])   # header row

        for j in range(len(timesteps)):
            writer.writerow([timesteps[j], results[j][0]])