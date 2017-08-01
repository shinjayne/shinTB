# Configuration for Model Tuning
config = {
	#Dataset
	"train_data_xml" : "svt1/train.xml",
	"test_data_xml" : "svt1/test.xml" ,
    #image size
    "image_size" : 300.0,
    ##################
    # In Learning
    ##################
    # Restoring variables from ckpt file
    "saved_dir" : "saved",
    # Random images Batch size in training
    "batch_size" : 1,
    # Learning rate until step 4000, 180000, 240000, and after
    "learning_rate_list" : [8e-4, 1e-3, 1e-4, 1e-5] ,
    # Whether Doing Batch Normalization in training phase
    "batch_norm" : True,
    # Negative and Positive Ratio in Overlapping
    "neg/pos": 3,

    ########################
    # In Default Box Shaping
    ########################

    # Layer boxes
    "layer_boxes" : [12, 12, 12, 12, 12, 12] ,
    # Default box Ratio at every single Map Point
    "box_ratios" : [1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
    # Default box Scale minimum
    "box_s_min" : 0.1 ,
    # Total Box number
    "total_boxes" : 23280 ,



}
