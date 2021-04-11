import os
import sys
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

if __name__ == '__main__':

	colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
	by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
	sorted_names = [name for hsv, name in by_hsv]

	res_path = sys.argv[1]
	points = []
	labels = []
	nn_out = []
	n_pts = 0
	with open(res_path) as f:
		for pt in f.readlines():
			n_pts += 1
			coords = pt.strip('\n').split(' ')
			for i in range(len(coords) - 2):
				if len(points) > i:
					points[i].append(float(coords[i]))
				else:
					points.append([float(coords[i])])
			labels.append(int(coords[-1:][0]))
			nn_out.append(int(coords[-2:-1][0]))
		n_classes = len(set(labels))

		for i in range(n_pts):
			if labels[i] == nn_out[i]:
				plt.plot(points[0][i], points[1][i], color = colors[sorted_names[labels[i] * 10]], marker = 'o')
			else:
				plt.plot(points[0][i], points[1][i], color = colors[sorted_names[nn_out[i] * 10]], marker = 's')
	plt.show()