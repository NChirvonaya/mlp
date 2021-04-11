import os
import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':

	if len(sys.argv) < 2:
		print('Enter output dir!\n')
		exit(0)

	out_dir = sys.argv[1]
	errs_path = os.path.join(out_dir, 'errs.txt')
	errs_list = []
	try:
		with open(errs_path) as f:
			for err in f.readlines():
				err = err.strip('\n')
				errs_list.append(float(err))
			plt.subplot(1, 2, 1)
			plt.plot(errs_list)
			plt.title('Train error')
	except Exception as e:
		raise e

	errs_path = os.path.join(out_dir, 'errs_val.txt')
	errs_list = []
	try:
		with open(errs_path) as f:
			for err in f.readlines():
				err = err.strip('\n')
				errs_list.append(float(err))
			plt.subplot(1, 2, 2)
			plt.plot(errs_list)
			plt.title('Validation error')
			plt.show()
	except Exception as e:
		raise e
	