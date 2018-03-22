
import os
import re

import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import sklearn
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt

import pickle
model_dir = '/home/ubuntu/tf_files/inception/'

listx =['062.Herring_Gull', '013.Bobolink', '027.Shiny_Cowbird', '129.Song_Sparrow', '111.Loggerhead_Shrike', '052.Pied_billed_Grebe', '002.Laysan_Albatross', '161.Blue_winged_Warbler', '040.Olive_sided_Flycatcher', '046.Gadwall', '088.Western_Meadowlark', '172.Nashville_Warbler', '139.Scarlet_Tanager', '195.Carolina_Wren', '134.Cape_Glossy_Starling', '095.Baltimore_Oriole', '130.Tree_Sparrow', '085.Horned_Lark', '123.Henslow_Sparrow', '133.White_throated_Sparrow', '103.Sayornis', '081.Pied_Kingfisher', '185.Bohemian_Waxwing', '171.Myrtle_Warbler', '150.Sage_Thrasher', '066.Western_Gull', '158.Bay_breasted_Warbler', '194.Cactus_Wren', '061.Heermann_Gull', '051.Horned_Grebe', '087.Mallard', '067.Anna_Hummingbird', '148.Green_tailed_Towhee', '189.Red_bellied_Woodpecker', '120.Fox_Sparrow', '144.Common_Tern', '193.Bewick_Wren', '154.Red_eyed_Vireo', '182.Yellow_Warbler', '036.Northern_Flicker', '071.Long_tailed_Jaeger', '030.Fish_Crow', '152.Blue_headed_Vireo', '089.Hooded_Merganser', '138.Tree_Swallow', '180.Wilson_Warbler', '077.Tropical_Kingbird', '035.Purple_Finch', '102.Western_Wood_Pewee', '162.Canada_Warbler', '200.Common_Yellowthroat', '197.Marsh_Wren', '159.Black_and_white_Warbler', '191.Red_headed_Woodpecker', '004.Groove_billed_Ani', '011.Rusty_Blackbird', '192.Downy_Woodpecker', '176.Prairie_Warbler', '156.White_eyed_Vireo', '142.Black_Tern', '050.Eared_Grebe', '137.Cliff_Swallow', '001.Black_footed_Albatross', '056.Pine_Grosbeak', '054.Blue_Grosbeak', '116.Chipping_Sparrow', '076.Dark_eyed_Junco', '059.California_Gull', '155.Warbling_Vireo', '143.Caspian_Tern', '047.American_Goldfinch', '072.Pomarine_Jaeger', '070.Green_Violetear', '063.Ivory_Gull', '094.White_breasted_Nuthatch', '080.Green_Kingfisher', '165.Chestnut_sided_Warbler', '029.American_Crow', '167.Hooded_Warbler', '164.Cerulean_Warbler', '199.Winter_Wren', '096.Hooded_Oriole', '057.Rose_breasted_Grosbeak', '175.Pine_Warbler', '053.Western_Grebe', '136.Barn_Swallow', '140.Summer_Tanager', '049.Boat_tailed_Grackle', '079.Belted_Kingfisher', '110.Geococcyx', '184.Louisiana_Waterthrush', '106.Horned_Puffin', '118.House_Sparrow', '104.American_Pipit', '010.Red_winged_Blackbird', '174.Palm_Warbler', '041.Scissor_tailed_Flycatcher', '045.Northern_Fulmar', '173.Orange_crowned_Warbler', '026.Bronzed_Cowbird', '122.Harris_Sparrow', '069.Rufous_Hummingbird', '128.Seaside_Sparrow', '091.Mockingbird', '112.Great_Grey_Shrike', '108.White_necked_Raven', '186.Cedar_Waxwing', '038.Great_Crested_Flycatcher', '127.Savannah_Sparrow', '042.Vermilion_Flycatcher', '099.Ovenbird', '100.Brown_Pelican', '145.Elegant_Tern', '163.Cape_May_Warbler', '082.Ringed_Kingfisher', '177.Prothonotary_Warbler', '092.Nighthawk', '090.Red_breasted_Merganser', '048.European_Goldfinch', '132.White_crowned_Sparrow', '183.Northern_Waterthrush', '031.Black_billed_Cuckoo', '025.Pelagic_Cormorant', '198.Rock_Wren', '093.Clark_Nutcracker', '114.Black_throated_Sparrow', '086.Pacific_Loon', '188.Pileated_Woodpecker', '121.Grasshopper_Sparrow', '109.American_Redstart', '074.Florida_Jay', '083.White_breasted_Kingfisher', '021.Eastern_Towhee', '146.Forsters_Tern', '044.Frigatebird', '073.Blue_Jay', '170.Mourning_Warbler', '131.Vesper_Sparrow', '014.Indigo_Bunting', '055.Evening_Grosbeak', '098.Scott_Oriole', '068.Ruby_throated_Hummingbird', '064.Ring_billed_Gull', '147.Least_Tern']
images_dir = '/home/ubuntu/tf_files/train/111.Loggerhead_Shrike/'
print(os.listdir(images_dir))


list_images = [images_dir+f for f in os.listdir(images_dir) if re.search('jpg|JPG', f)]
print(list_images)


def create_graph():
    with gfile.FastGFile(os.path.join(
    model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')



def extract_features(list_images):
    nb_features = 2048
    features = np.empty((len(list_images),nb_features))
    labels = []

    create_graph()

    with tf.Session() as sess:

        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for ind, image in enumerate(list_images):
            if (ind%100 == 0):
                print('Processing %s...' % (image))
            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)

            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(next_to_last_tensor,
                {'DecodeJpeg/contents:0': image_data})
            features[ind,:] = np.squeeze(predictions)
            labels.append(re.split('_\d+',image.split('/')[1])[0])

    return features, labels


features,labels = extract_features(list_images)
pickle.dump(features, open('features', 'wb'))
pickle.dump(labels, open('labels', 'wb'))
features = pickle.load(open('features'))
labels = pickle.load(open('labels'))
newFile = open('labelsfeatures.txt','w')
pickle.dump(features, newFile)
pickle.dump(labels, newFile)
newFile.close()
'''
X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.2, random_state=42)

clf = LinearSVC(multi_class='crammer_singer' ,C=10 )
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

def plot_confusion_matrix(y_true,y_pred):
    cm_array = confusion_matrix(y_true,y_pred)
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    plt.imshow(cm_array[:-1,:-1], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix", fontsize=16)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Number of images', rotation=270, labelpad=30, fontsize=12)
    xtick_marks = np.arange(len(true_labels))
    ytick_marks = np.arange(len(pred_labels))
    plt.xticks(xtick_marks, true_labels, rotation=90)
    plt.yticks(ytick_marks,pred_labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size


print("Accuracy: {0:0.1f}%".format(accuracy_score(y_test,y_pred)*100))
plot_confusion_matrix(y_test,y_pred)
'''