import tensorflow as tf
import time
from shintb.utils.learning_rate_maker import learning_rate_maker

class Runner :
    def __init__(self, config, graphdrawer, dataloader, dbcontrol):
        print("Runner instance initialing ...")
        print(">>> Collecting config info, GraphDrawer instance, DataLoader instance , DefaultBoxControl instance... Done!")
        self.sess = tf.Session(graph = tf.get_default_graph())
        self.saver = tf.train.Saver()

        self.config = config
        self.graph = graphdrawer
        self.dataloader = dataloader
        self.dbcontrol = dbcontrol
        print("Complete !!!")

    def train(self):
        print("Runner starts training ...")

        c = self.config
        g = self.graph
        print(tf.global_variables())
        # self.sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(c["model_dir"])

        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("!! RESTORE SAVED VARIBALES IN 'model_dir' !! : restored %s" % ckpt.model_checkpoint_path)

        while True:

            start_t = time.time()

            print(">>>", c["batch_size"] ," random batch data selected by DataLoader...", end=" ")
            train_imgs, train_gtboxes = self.dataloader.nextBatch(c["batch_size"])
            print("Done!")
            pred_conf, pred_loc, global_step = self.sess.run([g.pred_conf, g.pred_loc, g.global_step],
                                                             feed_dict = {g.imgs_ph : train_imgs,
                                                                          g.batch_norm_ph : False})

            print(">>>calculating (positive, negative, true_location) with jaccard overlap by DefaultBoxControl ...", end=" ")
            positive, negative , true_loc = self.dbcontrol.calculate_pos_neg_trueloc(train_gtboxes, pred_conf)
            print("Done!")


            learning_rate = learning_rate_maker(global_step, c["learning_rate_list"])

            self.sess.run(g.train,
                          feed_dict = {g.imgs_ph : train_imgs,
                                       g.true_loc_ph : true_loc,
                                       g.positive_ph : positive,
                                       g.negative_ph : negative,
                                       g.learning_rate_ph : learning_rate,
                                       g.batch_norm_ph : True
                                       }
                          )
            total_loss, global_step = self.sess.run([g.total_loss, g.global_step],
                                                                         feed_dict={g.imgs_ph: train_imgs,
                                                                                    g.true_loc_ph: true_loc,
                                                                                    g.positive_ph: positive,
                                                                                    g.negative_ph: negative,
                                                                                    g.learning_rate_ph: learning_rate,
                                                                                    g.batch_norm_ph: True
                                                                                    }
                                                                         )

            during_t = time.time() - start_t

            print("GLOBAL STEP :",global_step, "/ LEARNING RATE :", learning_rate, "/ LOSS :", total_loss, " (",during_t, "secs)" )

            if global_step % 1000 == 0 :
                self.saver.save(self.sess, "%s.ckpt" % c["model_dir"],global_step)

