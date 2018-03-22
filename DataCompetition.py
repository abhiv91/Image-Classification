
#Lets first conduct Data Augmentation on our dataset.

import pandas as pd

import os, sys

import tensorflow as tf
from PIL import Image
from PIL import ImageFilter
def dataAugment():
#list of all classes
    lisxtx = ['001.Black_footed_Albatross', '002.Laysan_Albatross', '004.Groove_billed_Ani', '010.Red_winged_Blackbird',
          '011.Rusty_Blackbird', '013.Bobolink', '014.Indigo_Bunting', '021.Eastern_Towhee', '025.Pelagic_Cormorant',
          '026.Bronzed_Cowbird', '027.Shiny_Cowbird', '029.American_Crow', '030.Fish_Crow', '031.Black_billed_Cuckoo',
          '035.Purple_Finch', '036.Northern_Flicker', '038.Great_Crested_Flycatcher', '040.Olive_sided_Flycatcher',
          '041.Scissor_tailed_Flycatcher', '042.Vermilion_Flycatcher', '044.Frigatebird', '045.Northern_Fulmar',
          '046.Gadwall', '047.American_Goldfinch', '048.European_Goldfinch', '049.Boat_tailed_Grackle',
          '050.Eared_Grebe', '051.Horned_Grebe', '052.Pied_billed_Grebe', '053.Western_Grebe', '054.Blue_Grosbeak',
          '055.Evening_Grosbeak', '056.Pine_Grosbeak', '057.Rose_breasted_Grosbeak', '059.California_Gull',
          '061.Heermann_Gull', '062.Herring_Gull', '063.Ivory_Gull', '064.Ring_billed_Gull', '066.Western_Gull',
          '067.Anna_Hummingbird', '068.Ruby_throated_Hummingbird', '069.Rufous_Hummingbird', '070.Green_Violetear',
          '071.Long_tailed_Jaeger', '072.Pomarine_Jaeger', '073.Blue_Jay', '074.Florida_Jay', '076.Dark_eyed_Junco',
          '077.Tropical_Kingbird', '079.Belted_Kingfisher', '080.Green_Kingfisher', '081.Pied_Kingfisher',
          '082.Ringed_Kingfisher', '083.White_breasted_Kingfisher', '085.Horned_Lark', '086.Pacific_Loon',
          '087.Mallard', '088.Western_Meadowlark', '089.Hooded_Merganser', '090.Red_breasted_Merganser',
          '091.Mockingbird', '092.Nighthawk', '093.Clark_Nutcracker', '094.White_breasted_Nuthatch',
          '095.Baltimore_Oriole', '096.Hooded_Oriole', '098.Scott_Oriole', '099.Ovenbird', '100.Brown_Pelican',
          '102.Western_Wood_Pewee', '103.Sayornis', '104.American_Pipit', '106.Horned_Puffin', '108.White_necked_Raven',
          '109.American_Redstart', '110.Geococcyx', '111.Loggerhead_Shrike', '112.Great_Grey_Shrike',
          '114.Black_throated_Sparrow', '116.Chipping_Sparrow', '118.House_Sparrow', '120.Fox_Sparrow',
          '121.Grasshopper_Sparrow', '122.Harris_Sparrow', '123.Henslow_Sparrow', '127.Savannah_Sparrow',
          '128.Seaside_Sparrow', '129.Song_Sparrow', '130.Tree_Sparrow', '131.Vesper_Sparrow',
          '132.White_crowned_Sparrow', '133.White_throated_Sparrow', '134.Cape_Glossy_Starling', '136.Barn_Swallow',
          '137.Cliff_Swallow', '138.Tree_Swallow', '139.Scarlet_Tanager', '140.Summer_Tanager', '142.Black_Tern',
          '143.Caspian_Tern', '144.Common_Tern', '145.Elegant_Tern', '146.Forsters_Tern', '147.Least_Tern',
          '148.Green_tailed_Towhee', '150.Sage_Thrasher', '152.Blue_headed_Vireo', '154.Red_eyed_Vireo',
          '155.Warbling_Vireo', '156.White_eyed_Vireo', '158.Bay_breasted_Warbler', '159.Black_and_white_Warbler',
          '161.Blue_winged_Warbler', '162.Canada_Warbler', '163.Cape_May_Warbler', '164.Cerulean_Warbler',
          '165.Chestnut_sided_Warbler', '167.Hooded_Warbler', '170.Mourning_Warbler', '171.Myrtle_Warbler',
          '172.Nashville_Warbler', '173.Orange_crowned_Warbler', '174.Palm_Warbler', '175.Pine_Warbler',
          '176.Prairie_Warbler', '177.Prothonotary_Warbler', '180.Wilson_Warbler', '182.Yellow_Warbler',
          '183.Northern_Waterthrush', '184.Louisiana_Waterthrush', '185.Bohemian_Waxwing', '186.Cedar_Waxwing',
          '188.Pileated_Woodpecker', '189.Red_bellied_Woodpecker', '191.Red_headed_Woodpecker', '192.Downy_Woodpecker',
          '193.Bewick_Wren', '194.Cactus_Wren', '195.Carolina_Wren', '197.Marsh_Wren', '198.Rock_Wren',
          '199.Winter_Wren', '200.Common_Yellowthroat']
# rotate and transform the images. Gaussian Blur has been removed after experimentation
    for name in lisxtx:
        path = '/home/ubuntu/tf_files/train/' + name + '/'
        files = os.listdir(path)
        i = 1
        print(path)
        for file in files:
            os.rename(os.path.join(path, file), os.path.join(path, str(i) + '.jpg'))
            im = Image.open(path + str(i) + '.jpg')
            im.rotate(180).save(path + str(i) + 'rotate.jpg')
            #removed gaussian blur
            im.transpose(Image.FLIP_LEFT_RIGHT).save(path + str(i) + "FLIPLR.jpg")
            im.transpose(Image.FLIP_TOP_BOTTOM).save(path + str(i) + "FLIPtb.jpg")

            i = i + 1

dataAugment()
#curl -O https://raw.githubusercontent.com/tensorflow/tensorflow/r1.1/tensorflow/examples/image_retraining/retrain.py
'''
The above script is from the tensorflow Library but not part of the package. Run in commandline where the data is stored. Input the following Parameters:

python retrain.py \
--bottleneck_dir=bottlenecks \
--how_many_training_steps=10000 \
--model_dir=inception \
--learning_rate=0.85 \
--summaries_dir=training_summaries/basic \
--output_graph=retrained_graph.pb \
--output_labels=retrained_labels.txt \
--print_misclassified_test_images True \
--testing_percentage 25 \
--test_batch_size -1 \
--image_dir=train

This will train Inception's CNN on the layer of images and keep 25% as training data. The Hyper Parameters have been experimented with and chosen by me for the Birds Data set for maximum Accuracy witht the test set.
'''

def predict():
    listx = list(range(1,4321))

    predlist = []
    output = pd.DataFrame({'Id': listx})


    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #adapted from Tensorflow examples and documentation, make sure to put this outside the for loop
    with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    for x in listx:
        #this only works since our test files are enumerated
        path = 'test/' + str(x + 1) + '.jpg'
        print(path)
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

                    flag = 1  # we only want the top prediction to append onto our list

    output['Prediction'] = predlist
    output.to_csv('output.csv')#ignore the index column (delete it with any csv editor)



#once your model is trained in the directory,
predict()