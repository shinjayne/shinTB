import tensorflow as tf

from config import config
from shintb import graph_drawer, svt_data_loader, default_box_control, runner, output_drawer

flags = tf.app.flags
FLAGS = flags.FLAGS

graphdrawer = graph_drawer.GraphDrawer(config)

dataloader = svt_data_loader.SVTDataLoader(config["train_data_xml"], config['test_data_xml'])

dbcontrol = default_box_control.DefaultBoxControl(config, graphdrawer)

outputdrawer = output_drawer.OutputDrawer(config, dbcontrol)

runner = runner.Runner(config, graphdrawer, dataloader, dbcontrol, outputdrawer)

if __name__ == "__main__":
	flags.DEFINE_string("mode", "train", "train,test ,image")
	flags.DEFINE_string("jobname", None, "job name for saving ckpt file")
	flags.DEFINE_integer("iter", 100000, "iteration for job")

	if FLAGS.mode == "train":
		if FLAGS.jobname ==None :
			raise FileNotFoundError("jobname 을 입력하지 않았습니다")
		else :
			runner.train(FLAGS.jobname, FLAGS.iter)

	elif FLAGS.mode == "test":
		runner.test(FLAGS.iter)

	elif FLAGS.mode == "image":
		runner.image()