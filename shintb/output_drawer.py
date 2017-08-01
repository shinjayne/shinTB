import numpy as np
import colorsys
import cv2

import shintb.utils.box_calculation as boxcal


class OutputDrawer:
    def __init__(self, config, dbcontrol):
        self.config = config
        self.dbcontrol = dbcontrol


    # for one image
    # pred_conf : [23830, 2]
    # pred_loc : [23830, 4]
    #

    def format_output(self, pred_conf, pred_loc, boxes=None, confidences=None):

        c = self.config

        if boxes is None:

            #[6, x, y, 12] shape list

            boxes = [
                [[[None for i in range(c["layer_boxes"][o])] for x in range(self.dbcontrol.out_shape[o][1])] for y in
                 range(self.dbcontrol.out_shape[o][2])]
                for o in range(len(c["layer_boxes"]))]

        if confidences is None:

            confidences = []

        index = 0  # 1 index -> 1 box (among 23280 boxes)

        # 6
        for o_i in range(len(c["layer_boxes"])):
            # x
            for x in range(self.dbcontrol.out_shape[o_i][2]):
                # y
                for y in range(self.dbcontrol.out_shape[o_i][1]):
                    # 12
                    for i in range(c["layer_boxes"][o_i]):

                        # for one image
                        # pred_conf : [23830, 2] (logits)
                        # pred_loc : [23830, 4]

                        diffs = pred_loc[index] #[dx,dy,dw,dh]
                        original = self.dbcontrol.default_boxes[o_i][x][y][i] #[x,y,w,h]

                        c_x = original[0] + original[2] * diffs[0]   # x+ w*dx
                        c_y = original[1] + original[3] * diffs[1]  #y + y*dy
                        w = original[2] * np.exp(diffs[2])     # w * exp(dw)
                        h = original[3] * np.exp(diffs[3])   # h * exp(dh)

                        boxes[o_i][x][y][i] = [c_x, c_y, w, h]
                        logits = pred_conf[index]  # [c1, c2]
                        # if np.argmax(logits) != classes+1:
                        info = ([o_i, x, y, i],
                                np.amax(np.exp(logits) / (np.sum(np.exp(logits)) + 1e-3)),
                                np.argmax(logits))
                        # indices, max probability, corresponding label

                        # if len(confidences) < index+1:
                        # 	confidences.append(info)
                        # else:
                        # 	confidences[index] = info
                        # else:
                        #    logits = pred_conf[index][:-1]
                        #    confidences.append(([o_i, x, y, i], np.amax(np.exp(logits) / (np.sum(np.exp(logits)) + 1e-3)),
                        #                        np.argmax(logits)))
                        confidences.append(info)
                        index += 1

        # sorted_confidences = sorted(confidences, key=lambda tup: tup[1])[::-1]

        return boxes, confidences


    def draw_outputs(self, img, boxes, confidences, wait=1):
        I = img * 255.0

        # nms = non_max_suppression_fast(np.asarray(filtered_boxes), 1.00)
        picks = self.postprocess_boxes(boxes, confidences)

        print("PICKED BOXES INFO :", picks)

        for box, conf, top_label in picks:  # [filtered[i] for i in picks]:
            if top_label != 1:
                # print("%f: %s %s" % (conf, coco.i2name[top_label], box))

                c = colorsys.hsv_to_rgb(((top_label * 17) % 255) / 255.0, 1.0, 1.0)
                c = tuple([255 * c[i] for i in range(3)])

        I = cv2.cvtColor(I.astype(np.uint8), cv2.COLOR_RGB2BGR)

        for box, conf, top_label in picks :
            x, y, w, h = box[0] ,box[1], box[2], box[3]
            rect_start = (x,y)
            rect_end = (x+w, y+h)
            I = cv2.rectangle(I, rect_start, rect_end, (255, 0, 0) , 5 )

            print("Textboxes information!")
            print("rect_start : ", rect_start , "// rect_end :", rect_end)
            print("confidence: ", conf)

        #doing GOOD #I = cv2.rectangle(I, (10,10), (100,100), (255,0,0), 5) #test color
        cv2.imshow("outputs", I )
        cv2.waitKey(wait)



    def basic_nms(self, boxes, thres=0.45):
        re = []

        def pass_nms(c, lab):
            for box_, conf_, top_label_ in re :
                #if lab == top_label_ and boxcal.calc_jaccard(c, box_) > thres:
                if lab == 0 and boxcal.calc_jaccard(c, box_) < thres:
                    return False
            return True



        for box, conf, top_label in boxes:
            # top_label = 0 : text // top_label=1 : background
            if top_label != 1 and pass_nms(box, top_label):
                re.append((box, conf, top_label))

                # re.append(index)
                if len(re) >= 200:
                    break

        return re  #[(corneredbox,conf,top_label), ...]


    # center to corner process
    def postprocess_boxes(self, boxes, confidences, min_conf=0.001, nms=0.45):
        filtered = []

        for box, conf, top_label in confidences:
            if conf >= min_conf:
                coords = boxes[box[0]][box[1]][box[2]][box[3]]
                coords = boxcal.center2cornerbox(coords)

                filtered.append((coords, conf, top_label))

        print("FILTERED BOXES INFO :", filtered)

        return self.basic_nms(filtered, nms)

