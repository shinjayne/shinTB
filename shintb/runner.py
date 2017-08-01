import tensorflow as tf
import cv2

import time
from shintb.utils.learning_rate_maker import learning_rate_maker


class Runner :
    def __init__(self, config, graphdrawer, dataloader, dbcontrol, outputdrawer):
        print("Runner instance initialing ...")
        print(">>> Collecting config info, GraphDrawer instance, DataLoader instance , DefaultBoxControl instance... Done!")
        #self.sess = tf.Session(graph = tf.get_default_graph(), config=tf.ConfigProto(log_device_placement=True))
        self.sess = tf.Session(graph=tf.get_default_graph())
        self.saver = tf.train.Saver()  # tensorflow ckpt poto buff
          # tensorflow summary proto buff

        self.config = config
        self.graph = graphdrawer
        self.dataloader = dataloader
        self.dbcontrol = dbcontrol
        self.outputdrawer = outputdrawer
        print("Complete !!!")

    def train(self, jobname, iter):
        print("Runner starts training job...")
        self.writer = tf.summary.FileWriter(logdir=(self.config["saved_dir"] + "/" + jobname))


        c = self.config
        g = self.graph
        #print(tf.global_variables()[0])
        self.sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(c["saved_dir"])

        if ckpt and ckpt.model_checkpoint_path:
            last_step = int(ckpt.model_checkpoint_path.split("-")[-1])
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("!! RESTORE SAVED VARIBALES !! : restored %s" % ckpt.model_checkpoint_path)
            print("!! RESTORE SAVED VARIBALES !! : restart from step " , last_step)
            print("!! RESTORE SAVED VARIBALES !! : restored %s" % ckpt.model_checkpoint_path)
            print("!! RESTORE SAVED VARIBALES !! : restored %s" % ckpt.model_checkpoint_path)
            print("!! RESTORE SAVED VARIBALES !! : restored %s" % ckpt.model_checkpoint_path)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        step = 0
        while step < iter :

            start_t = time.time()

            print(">>> 8 random batch data selected by DataLoader...", end=" ")
            train_imgs, train_gtboxes = self.dataloader.nextBatch(c["batch_size"])
            print("Done!")

            print(">>> from graph, get pred_conf, pred_loc ...", end =" ")
            pred_conf, pred_loc, global_step = self.sess.run([g.pred_conf, g.pred_loc, g.global_step],
                                                             feed_dict = {g.imgs_ph : train_imgs,
                                                                          g.batch_norm_ph : False})
            print("Done!")

            print(">>>calculating (positive, negative, true_location) with jaccard overlap by DefaultBoxControl ...", end=" ")
            positive, negative , true_loc = self.dbcontrol.calculate_pos_neg_trueloc(pred_conf, train_gtboxes)
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
                ckpt_path = self.saver.save(self.sess, "%s.ckpt" % (c["saved_dir"]+"/"+jobname),global_step)
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                print("SAVED AT : ", ckpt_path)
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

            summ_proto_buff=self.sess.run(g.summaries, feed_dict={g.imgs_ph: train_imgs,
                                                                                    g.true_loc_ph: true_loc,
                                                                                    g.positive_ph: positive,
                                                                                    g.negative_ph: negative,
                                                                                    g.learning_rate_ph: learning_rate,
                                                                                    g.batch_norm_ph: True
                                                                                    })
            self.writer.add_summary(summ_proto_buff, global_step = global_step)
            step += 1

        ckpt_path = self.saver.save(self.sess, "%s.ckpt" % (c["saved_dir"]+"/"+jobname), global_step)

        print("::: TRAINING JOB FINISHED ::: \n::: CKPT SAVED AT", ckpt_path," ::: \n:::  SUMMARY LOG SAVED AT ", (c["saved_dir"]+"/"+jobname) )
        print("TIP : LAUNCH tensorboard WITH ...  >> tensorboard --logdir=",(c["saved_dir"]+"/"+jobname))


    def test(self, iter):

        print("Runner starts finding texts on test dataset ...")
        c = self.config
        g = self.graph
        self.sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(c["saved_dir"])

        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("!! RESTORE SAVED VARIBALES !! : restored %s" % ckpt.model_checkpoint_path)
            print("!! RESTORE SAVED VARIBALES !! : restored %s" % ckpt.model_checkpoint_path)
            print("!! RESTORE SAVED VARIBALES !! : restored %s" % ckpt.model_checkpoint_path)
            print("!! RESTORE SAVED VARIBALES !! : restored %s" % ckpt.model_checkpoint_path)
            print("!! RESTORE SAVED VARIBALES !! : restored %s" % ckpt.model_checkpoint_path)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

            cv2.namedWindow("outputs", cv2.WINDOW_NORMAL)

            step = 0
            while step < iter:

                print(">>>  3 random batch data selected by DataLoader...", end=" ")
                test_imgs, test_gtboxes = self.dataloader.nextBatch(3, 'test')
                print("Done!")

                print("IMGS INPUT: ", test_imgs[0], test_imgs.shape)
                print("GROUND TRUTH BOX INFO :", test_gtboxes[0])

                pred_conf, pred_loc, global_step =  self.sess.run([g.pred_conf, g.pred_loc, g.global_step],
                                                                    feed_dict={g.imgs_ph: test_imgs,
                                                                    g.batch_norm_ph: False})

                print("GRAPH INPUT:", self.sess.run(g.input, feed_dict={g.imgs_ph: test_imgs}))

                out1, out2, out3, out4, out5, out6 = self.sess.run([g.out1,g.out2,g.out3,g.out4,g.out5,g.out6],
                                                                    feed_dict={g.imgs_ph: test_imgs,
                                                                    g.batch_norm_ph: False})

                conv1_1, conv1_2, conv4_3, conv7 = self.sess.run([g.conv1_1,g.conv1_2,g.conv4_3, g.conv7],
                                                                    feed_dict={g.imgs_ph: test_imgs,
                                                                    g.batch_norm_ph: False})

                print("CONV1_1 INFO : ", conv1_1)
                print("CONV1_2 INFO : ", conv1_2)
                print("CONV4_3 INFO : ", conv4_3)
                print("CONV7 INFO : ", conv7)


                print("MAP POINT 1 INFO : ", out1)
                #print("MAP POINT 2 INFO : ", out2)
                #print("MAP POINT 3 INFO : ", out3)
                #print("MAP POINT 4 INFO : ", out4)
                #print("MAP POINT 5 INFO : ", out5)
                #print("MAP POINT 6 INFO : ", out6)


                print("PREDICTED CONFIDENCE : ", pred_conf[0], pred_conf.shape)
                print("PREDICTED LOCATION :", pred_loc[0], pred_loc.shape)

                output_boxes, output_confidence = self.outputdrawer.format_output(pred_conf[0], pred_loc[0])

                self.outputdrawer.draw_outputs(test_imgs[0], output_boxes, output_confidence, wait=1)

                step += 1








        else :
            raise FileNotFoundError("ckpt저장 폴더에서 불러올 ckpt 파일을 찾지 못했습니다")





    def image(self):
        print("Runner starts finding texts on test dataset ...")
        c = self.config
        g = self.graph
        self.sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(c["saved_dir"])

        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("!! RESTORE SAVED VARIBALES !! : restored %s" % ckpt.model_checkpoint_path)
            print("!! RESTORE SAVED VARIBALES !! : restored %s" % ckpt.model_checkpoint_path)
            print("!! RESTORE SAVED VARIBALES !! : restored %s" % ckpt.model_checkpoint_path)
            print("!! RESTORE SAVED VARIBALES !! : restored %s" % ckpt.model_checkpoint_path)
            print("!! RESTORE SAVED VARIBALES !! : restored %s" % ckpt.model_checkpoint_path)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

            #cv2.namedWindow("outputs", cv2.WINDOW_NORMAL)




            print(">>> 1 random batch data selected by DataLoader...", end=" ")
            test_img, test_gtbox = self.dataloader.nextBatch(1, 'test')
            print("Done!")

            pred_conf, pred_loc, global_step =  self.sess.run([g.pred_conf, g.pred_loc, g.global_step],
                                                                    feed_dict={g.imgs_ph: test_img,
                                                                    g.batch_norm_ph: False})




            output_boxes, output_confidence = self.outputdrawer.format_output(pred_conf[0,:,:], pred_loc[0,:,:])

            #nmsed_list = self.outputdrawer.postprocess_boxes(output_boxes, output_confidence, 0.01, 0.45)



            self.outputdrawer.draw_outputs(test_img, output_boxes , output_confidence , wait=1)










        else :
            raise FileNotFoundError("ckpt저장 폴더에서 불러올 ckpt 파일을 찾지 못했습니다")



