import pandas as pd





listx= list(range(1,4320))

predlist=[]
output = pd.DataFrame({'Id': listx})


import os, sys

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


for x in listx:
    path = 'test/'+str(x+1)+'.jpg'
    print (path)

    # change this as you see fit
    image_path = path

    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("retrained_labels.txt")]
   
    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
                               {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        # print('the top result is' + label_lines[node_id])
        flag = 0
        for node_id in top_k:

            while flag == 0:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                predlist.append(int(human_string[:3]))
                print('%s' % (human_string))

                flag = 1  # we only want the top prediction

output['Prediction']=predlist
output.to_csv('outputtest.csv')



