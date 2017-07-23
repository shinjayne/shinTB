from config import config
from shintb import graph_drawer ,svt_data_loader, default_box_control, runner

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

graphdrawer = graph_drawer.GraphDrawer(config)

dataloader = svt_data_loader.SVTDataLoader('./svt1/train.xml', './svt1/test.xml')

dbcontrol = default_box_control.DefaultBoxControl(config, graphdrawer)

runner = runner.Runner(config, graphdrawer, dataloader, dbcontrol)

if __name__ == "__main__":
	flags.DEFINE_string("mode", "train", "train, image")

	if FLAGS.mode == "train":
		runner.train()
	elif FLAGS.mode == "image":
		runner.image()
