from PIL import Image
import matplotlib.pyplot as plt


# Log images
def log_input_image(x, opts):
	return tensor2im(x)


def tensor2im(var):
	# var shape: (3, H, W)
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))


def vis_faces(log_hooks):
	display_count = len(log_hooks)
	fig = plt.figure(figsize=(8, 4 * display_count))
	gs = fig.add_gridspec(display_count, 3)
	for i in range(display_count):
		hooks_dict = log_hooks[i]
		fig.add_subplot(gs[i, 0])
		if 'diff_input' in hooks_dict:
			vis_faces_with_id(hooks_dict, fig, gs, i)
		else:
			vis_faces_no_id(hooks_dict, fig, gs, i)
	plt.tight_layout()
	return fig

def vis_all(log_hooks):
	display_count = len(log_hooks)
	display_imgs = len(log_hooks[0])

	font_dict = dict(fontsize=25, family='Times New Roman', weight='bold')
	fig = plt.figure(figsize=(4 * display_imgs, 4 * display_count))
	gs = fig.add_gridspec(display_count, display_imgs)

	for i in range(display_count):
		hooks_dict = log_hooks[i]

		index = 0
		for key in hooks_dict:
			fig.add_subplot(gs[i, index])
			plt.imshow(hooks_dict[key])
			plt.xticks([])
			plt.yticks([])
			if i == 0:
				plt.title(key, fontdict=font_dict)
			index += 1

	plt.tight_layout()
	return fig

def vis_faces_with_id(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'])
	plt.title('Input\nOut Sim={:.2f}'.format(float(hooks_dict['diff_input'])))
	# plt.axes('off')
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['target_face'])
	plt.title('Target\nIn={:.2f}, Out={:.2f}'.format(float(hooks_dict['diff_views']),
	                                                 float(hooks_dict['diff_target'])))
	# plt.axes('off')
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Output\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))
	# plt.axes('off')


def vis_faces_no_id(hooks_dict, fig, gs, i):

	plt.imshow(hooks_dict['input_face'], cmap="gray")
	plt.xticks([])
	plt.yticks([])
	plt.title('Input')
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['target_face'])
	plt.xticks([])
	plt.yticks([])
	plt.title('Target')
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['output_face'])
	plt.xticks([])
	plt.yticks([])
	plt.title('Output')

