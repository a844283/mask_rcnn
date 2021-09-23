# detect coffeebeans in images with mask rcnn model
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset


class CoffeeBeanDataset(Dataset):
	# load the dataset
	def load_dataset(self, dataset_dir, is_train=True):
		self.add_class("dataset", 1, "healthy")
		self.add_class("dataset", 2, "black" )
		self.add_class("dataset", 3, "sour" )
		self.add_class("dataset", 4, "moldy")
		self.add_class("dataset", 5, "faded_crystallized")
		self.add_class("dataset", 6, "faded_streaked")
		self.add_class("dataset", 7, "faded_old")
		self.add_class("dataset", 8, "faded_buttery")
		self.add_class("dataset", 9, "faded_overdried")
		self.add_class("dataset", 10, "cut")
		self.add_class("dataset", 11, "slight_insect_damage")
		self.add_class("dataset", 12, "severe_insect_damage")
		self.add_class("dataset", 13, "shrunk")
		self.add_class("dataset", 14, "immature")
		self.add_class("dataset", 15, "crushed")
		self.add_class("dataset", 16, "underdried")
		self.add_class("dataset", 17, "dry_cherry")
		self.add_class("dataset", 18, "parchment")
		self.add_class("dataset", 19, "shell")
		self.add_class("dataset", 20, "token")

		train_images_dir = dataset_dir + 'train/JPEGImages/'
		train_annotations_dir = dataset_dir + 'train/Annotations/'
		test_images_dir = dataset_dir + 'test/JPEGImages/'
		test_annotations_dir = dataset_dir + 'test/Annotations/'
		if is_train:
			images_dir = train_images_dir
			annotations_dir = train_annotations_dir
		else:
			images_dir = test_images_dir
			annotations_dir = test_annotations_dir
		for filename in listdir(images_dir):
			image_id = filename[:-4]
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

	def extract_boxes(self, filename):
		# load and parse the file
		root = ElementTree.parse(filename).getroot()
		boxes = list()
		for objs_name in root.findall('.//object'):
			obj_name = objs_name.find('name').text
			for box in objs_name.findall('./bndbox'):
				xmin = int(float(box.find('xmin').text))
				ymin = int(float(box.find('ymin').text))
				xmax = int(float(box.find('xmax').text))
				ymax = int(float(box.find('ymax').text))
				coors = [xmin, ymin, xmax, ymax, obj_name]
				boxes.append(coors)
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	# load the masks for an image
	def load_mask(self, image_id):
		info = self.image_info[image_id]
		path = info['annotation']
		boxes, w, h = self.extract_boxes(path)
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			masks[row_s:row_e, col_s:col_e, i] = 1
			class_ids.append(self.class_names.index(box[4]))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

# define a configuration for the model
class CoffeeBeanConfig(Config):
	NAME = "cb"
	NUM_CLASSES = 1 + 20
	STEPS_PER_EPOCH = 5

if __name__ == '__main__':
	train_set = CoffeeBeanDataset()
	train_set.load_dataset('../coffeebean/', is_train=True)
	train_set.prepare()
	test_set = CoffeeBeanDataset()
	test_set.load_dataset('../coffeebean/', is_train=False)
	test_set.prepare()
	config = CoffeeBeanConfig()
	config.display()
	model = MaskRCNN(mode='training', model_dir='./', config=config)
	# load weights
	model.load_weights('mask_rcnn_balloon.h5', by_name=True,
										  exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
	# train weights
	model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')


