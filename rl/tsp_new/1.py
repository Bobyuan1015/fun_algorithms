def render(self, distance, path, true_distance, save=False):
    x, y = zip(*self.locations)
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, color="red", label="Locations")
    plt.plot(x, y, "o")

    # Add arrows to show visit order and display location values
    for i in range(1, len(path)):
        start_idx = path[i - 1]
        end_idx = path[i]
        plt.annotate(
            '', xy=self.locations[end_idx], xytext=self.locations[start_idx],
            arrowprops=dict(arrowstyle="->", color="blue", lw=1.5)
        )
        # Display the coordinates for each location
        plt.text(self.locations[start_idx][0], self.locations[start_idx][1],
                 f"{path[i]}", fontsize=9, ha="right")
    # Display the last point's coordinates as well
    plt.text(self.locations[path[-1]][0], self.locations[path[-1]][1],
             f"{self.locations[path[-1]]}", fontsize=9, ha="right")

    # Label the start and end points
    plt.annotate("Start", xy=self.locations[path[0]],
                 xytext=(self.locations[path[0]][0] - 0.5,
                         self.locations[path[0]][1] - 0.5),
                 color="green", weight="bold", fontsize=12)
    plt.annotate("End", xy=self.locations[path[-1]],
                 xytext=(self.locations[path[-1]][0] + 0.5,
                         self.locations[path[-1]][1] + 0.5),
                 color="red", weight="bold", fontsize=12)

    plt.legend()
    plt.title(f"TSP Path, distance={distance} true={true_distance}", fontsize=9)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)

    # Save the figure if requested
    if save:
        plt.savefig("TSP_path.png")

    plt.show()