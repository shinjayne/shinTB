import shintb.utils.box_calculation as boxcal
import numpy as np
import tensorflow as tf


class DefaultBoxControl:
    def __init__(self, config, graphdrawer):
        print("Default Box Control instance Initialing...")
        self.config = config
        self.graph = graphdrawer

        # shape of out1, out2, out3, out4, out5, out6 in textbox layer
        # list( [?, y(i), x(i), 72] for i in range(6) )
        self.out_shape = self.calculate_out_shape()
        print(">>>Check textbox layers(6) shape : \n", self.out_shape, "Done!")

        self.default_boxes = self.initialize_default_boxes()

        print("Complete !!!")

    def calculate_out_shape(self):
        g = self.graph
        outs = [g.out1, g.out2, g.out3, g.out4, g.out5, g.out6]
        return [out.get_shape().as_list() for out in outs]

    def calculate_box_scale(self, k):
        c = self.config
        s_min = c["box_s_min"]
        s_max = 0.95
        m = 6.0

        s_k = s_min + (s_max - s_min) * (k - 1.0) / (m - 1.0)  # equation 2

        return s_k

    def initialize_default_boxes(self):
        c = self.config

        print(">>>Initializing Default Boxes ...")

        boxes = []

        for o_i in range(len(self.out_shape)):  # 6
            print("default boxes in map point ", o_i, " :", end=" ")
            layer_boxes = []
            layer_shape = self.out_shape[o_i]  # [?, y(i), x(i), 72]
            s_k = self.calculate_box_scale(o_i + 1)
            for x in range(layer_shape[2]):
                x_boxes = []
                for y in range(layer_shape[1]):
                    y_boxes = []
                    rs = c["box_ratios"]
                    for i in range(len(rs)):  # 12
                        scale = s_k
                        default_w = scale * np.sqrt(rs[i])
                        default_h = scale / np.sqrt(rs[i])
                        c_x = (x + 0.5) / float(layer_shape[2])  # 0~1scaling 해준다
                        c_y = (y + 0.5) / float(layer_shape[1])  # 0~1scaling 해준다
                        y_boxes.append([c_x, c_y, default_w, default_h])
                        c_y = (y + 1.0) / float(layer_shape[1])
                        y_boxes.append([c_x, c_y, default_w, default_h])
                    x_boxes.append(y_boxes)
                layer_boxes.append(x_boxes)
            boxes.append(layer_boxes)
            print("list size = [x:", layer_shape[2], "][y:", layer_shape[1], "][boxes:", len(c["box_ratios"]) * 2, "][[c_x, c_y, default_w, default_h]]")

        print("Done!")

        return boxes


    # pred_conf : [?, 23280, 2]
    def calculate_pos_neg_trueloc(self, pred_conf, gtboxes):
        print("calculate_pos_neg_trueloc START")
        c = self.config

        pos_neg_trueloc = [None for i in range(c["batch_size"])]

        for batch_i in range(c["batch_size"]) :
            matches = self.matching_dbboxes_gtboxes_in_batch_i(pred_conf[batch_i], gtboxes[batch_i])

            pos_i, neg_i, _, true_loc_i = self.prepare_pos_neg_trueloc_in_matches(matches)

            pos_neg_trueloc[batch_i] = (pos_i, neg_i, _, true_loc_i)

        positive, negative, _true_conf ,true_loc = [np.stack(m) for m in zip(*pos_neg_trueloc)]

        print("calculate_pos_neg_trueloc END")
        return positive, negative, true_loc

    # pred_conf_i : [23280, 2]
    def matching_dbboxes_gtboxes_in_batch_i(self, pred_conf_i, gtboxes_i):
        print("matching_dbboxes_gtboxes_in_batch_i START")
        c = self.config

        #default box들의 좌표와 번째 수를 나열해놓은 리스트
        dbtracker = []
        for o_i in range(len(c["layer_boxes"])):
            for x in range(self.out_shape[o_i][2]):
                for y in range(self.out_shape[o_i][1]):
                    for i in range(c["layer_boxes"][o_i]):
                        dbtracker.append([o_i, x, y, i])


        # c.layer_boxes = [12,12,12,12,12,12]
        # o : each Map Location !
        # matches.shape = [6, out_shape[o][2]x , out_shape[o][1]y , 12]
        matches = [[[[None for i in range(c["layer_boxes"][o])] for y in range(self.out_shape[o][1])] for y in
                    range(self.out_shape[o][2])]
                   for o in range(len(c["layer_boxes"]))]

        positive_count = 0

        # iteration : for each ground_truth_box
        for (gt_box, box_id) in  gtboxes_i :
            #################################
            # about this ground_truth_box ...
            #################################
            top_match = (None, 0)

            # o : each Map Location !
            for o in range(len(c["layer_boxes"])):
                ##########################
                # in this Map Location ...
                ##########################

                # gt_box = [x,y,w,h] 안의 값들은 scaled 되어있으므로
                # 이 scaled 되어있는 값들을 this Map Location 에 맞게 곱해서 부풀려준다

                x1 = max(int(gt_box[0] / (1.0 / self.out_shape[o][2])), 0)
                y1 = max(int(gt_box[1] / (1.0 / self.out_shape[o][1])), 0)
                x2 = min(int((gt_box[0] + gt_box[2]) / (1.0 / self.out_shape[o][2])) + 2, self.out_shape[o][2])
                y2 = min(int((gt_box[1] + gt_box[3]) / (1.0 / self.out_shape[o][1])) + 2, self.out_shape[o][1])

                #				o`th Map
                #	 -------------------------------
                #	|								|
                #	|								|
                #	|								|
                # 	|	(x1,y2)--------(x2,y2)		|
                # 	|	|                    |		|
                # 	|	|       gt_area      |		|
                # 	|	|                    |		|
                # 	|	(x1,y1)--------(x2,y1)		|
                #	|								|
                #	|								|
                #	|								|
                #	 -------------------------------
                #
                #
                # Map 에서 이 area 안에 있는 모든 픽셀들에 대하여
                # 그 픽셀의 default box들과 ground_truth box 의 Jaccard Overlap 을 구한다

                for y in range(y1, y2):
                    for x in range(x1, x2):
                        for i in range(c["layer_boxes"][o]):

                            # c.defaults : contains informations of default boxes
                            # c.defaults[map_location][x][y][12]
                            # (x,y) 에서의 12 번째 default_box 정보

                            box = self.default_boxes[o][x][y][i]
                            # 이 box 안에도 scaled 된 default box (x,y,w,h) 정보가 들어있으므로
                            # 바로 gt_box 와 비교가능

                            # gt_box is corner, box is center-based so convert
                            jacc = boxcal.calc_jaccard(gt_box, boxcal.center2cornerbox(box))
                            if jacc >= 0.5:
                                matches[o][x][y][i] = (gt_box, box_id)
                                positive_count += 1
                            if jacc > top_match[1]:
                                top_match = ([o, x, y, i], jacc)

            top_box = top_match[0]
            # if box's jaccard is <0.5 but is the best
            if top_box is not None and matches[top_box[0]][top_box[1]][top_box[2]][top_box[3]] is None:
                positive_count += 1
                matches[top_box[0]][top_box[1]][top_box[2]][top_box[3]] = (gt_box, box_id)
        ###################
        # End of Iteration !
        #
        # if ... matches[map_loc][x][y][12] = (gt_box, box_id) or None
        # 		map_location 의 (x,y) 에서 12번째 default_box 랑 (gt_box, box_id) 가 match 되었음
        #
        # if ... matches[map_loc][x][y][12] =  None
        #		map_location 의 (x,y) 에서 12번째 default_box 는 매치된 gt_box 가 없음
        ###################



        ###########################
        # --HARD NEGATIVE MINING---
        ###########################

        # < default box 와 ground_truth box 의 Jaccad Overlap 을 기준으로 >
        # positive : overlap > 0.5  넘는 default box
        # negative : overlap < 0.5 인 아이들
        # HARD NEGATIVE MINING : (pos : neg = 1 : 3) 되도록 negative 갯수 조정
        # c["neg/pos"] = 3
        # negative 선별 기준??? : pred_conf 의 confidence loss 가 높은 순으로 (random 아님)

        # negative 가 왜 필요한가 ?? : confidence loss function 에서 사용됩니다

        # positive / negative 선별된 박스들 제외하고는 None 으로 제거됩니다

        negative_max = positive_count * c["neg/pos"]
        negative_count = 0

        # pred_conf_i : [ 23280, 2]
        top_confidences = self.get_top_confidences(pred_conf_i, negative_max)
        # top confidence 들의 index list 를 반환

        for i in top_confidences:
            # n번째 index 를 가진것이 matches 에 어디에 위치하는지 보기위해
            # [6][x][y][12] 형식으로 변환
            dbtrack = dbtracker[i]

            # top confidence 의 index를 따라가 ..,
            if matches[dbtrack[0]][dbtrack[1]][dbtrack[2]][dbtrack[3]] is None : # and np.argmax(pred_conf[i]) != classes
                matches[dbtrack[0]][dbtrack[1]][dbtrack[2]][dbtrack[3]] = -1
                # negative인 부분은 -1 로 채워준다

                # matches에
                # 결과적으로 positive 는 (gt_box, box_id)
                # negative 는 -1
                # 나머지는 None

                negative_count += 1

                if negative_count >= negative_max:
                    break

        #print("%i positives" % positive_count)
        #print("%i/%i negatives" % (negative_count, negative_max))
        print("matching_dbboxes_gtboxes_in_batch_i END")
        return matches

    #pred_conf_i : [23280, 2]
    def get_top_confidences(self, pred_conf_i, negative_max):
        print("get_top_confidences START")
        confidences = []

        # pred_conf_i : [23280, 2]
        #logits : [c1, c2]

        # 오버플로우 방지!
        def softmax_without_overflow(a, reduce_axis=1) :
            c = np.max(a, reduce_axis)
            c = np.reshape(c, [-1,1])
            exp_a = np.exp(a - c)
            sum_exp_a = a.sum(reduce_axis)
            sum_exp_a = np.reshape(sum_exp_a, [-1, 1])

            return exp_a / (sum_exp_a + 1.0e-4)

        pred_conf_i_softmax =  softmax_without_overflow(pred_conf_i)
        # pred_conf_i_softmax = tf.nn.softmax(pred_conf_i, -1)

        for probs in pred_conf_i_softmax :
            top_confidence = np.amax(probs)
            confidences.append(top_confidence)

        '''
        for logits in pred_conf:
            #probs = (1.0/np.exp(-logits)) / np.add(np.sum((1.0/np.exp(-logits))), 1e-3)
            probs = tf.nn.softmax(logits)
            # probs : [softmax_c1, softmax_c2]
            top_label = np.amax(probs)
            confidences.append(top_label)
        '''
        # top_confidences = sorted(confidences, key=lambda tup: tup[1])[::-1]

        k = min(negative_max, len(confidences))
        top_confidences = np.argpartition(np.asarray(confidences), -k)[-k:]

        print("get_top_confidences END")
        return top_confidences

    # prepare_feed  (이미지 하나에 대하여)
    # input : 한 이미지의 matches (shape [map_loc][x][y][12] ) Jaccad overlap 수행한 결과들
    # output :  a_positives, a_negatives, a_true_labels, a_true_locs
    # a_positives (23280, )
    # a_negatives (23280, )
    # a_true_labels (23280, )
    # a_true_locs (23280, 4)
    def prepare_pos_neg_trueloc_in_matches(self, matches):
        print("prepare_pos_neg_trueloc_in_matches START")
        c = self.config

        positives_list = []
        negatives_list = []
        true_labels_list = []
        true_locs_list = []

        # o : Map Location
        for o in range(len(c["layer_boxes"])):  # 6
            # c.out_shapes[o]
            for x in range(self.out_shape[o][2]):  # x
                for y in range(self.out_shape[o][1]):  # y
                    for i in range(c["layer_boxes"][o]):  # 12
                        match = matches[o][x][y][i]

                        if isinstance(match, tuple):  # there is a ground truth assigned to this default box
                            positives_list.append(1)
                            negatives_list.append(0)
                            true_labels_list.append(match[1])  # groundtruth id
                            default = self.default_boxes[o][x][y][i]  # default box
                            true_locs_list.append(boxcal.calc_offsets(default, boxcal.corner2centerbox(match[0])))
                        elif match == -1:  # this default box was chosen to be a negative
                            positives_list.append(0)
                            negatives_list.append(1)
                            # c.classes = 1
                            true_labels_list.append(1)  # background class
                            # 보아하니 background 이면 confidence = 1
                            # 그리고 text 이면 confidence = 0  인듯
                            true_locs_list.append([0] * 4)

                        # 아래의 경우는 발생하지 않는게 정상이다
                        else:  # no influence for this training step
                            positives_list.append(0)
                            negatives_list.append(0)
                            true_labels_list.append(1)  # background class
                            true_locs_list.append([0] * 4)

        a_positives = np.asarray(positives_list)
        a_negatives = np.asarray(negatives_list)
        a_true_labels = np.asarray(true_labels_list)
        a_true_locs = np.asarray(true_locs_list)

        print("prepare_pos_neg_trueloc_in_matches END  ")
        return a_positives, a_negatives, a_true_labels, a_true_locs


