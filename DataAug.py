from PIL import Image
from PIL import ImageFilter
import os

lisxtx = [ '086.Pacific_Loon', '087.Mallard', '088.Western_Meadowlark', '089.Hooded_Merganser', '090.Red_breasted_Merganser', '091.Mockingbird', '092.Nighthawk', '093.Clark_Nutcracker', '094.White_breasted_Nuthatch', '095.Baltimore_Oriole', '096.Hooded_Oriole', '098.Scott_Oriole', '099.Ovenbird', '100.Brown_Pelican',  '102.Western_Wood_Pewee', '103.Sayornis', '104.American_Pipit',  '106.Horned_Puffin',  '108.White_necked_Raven', '109.American_Redstart', '110.Geococcyx', '111.Loggerhead_Shrike', '112.Great_Grey_Shrike',  '114.Black_throated_Sparrow',  '116.Chipping_Sparrow',  '118.House_Sparrow',  '120.Fox_Sparrow', '121.Grasshopper_Sparrow', '122.Harris_Sparrow', '123.Henslow_Sparrow', '127.Savannah_Sparrow', '128.Seaside_Sparrow', '129.Song_Sparrow', '130.Tree_Sparrow', '131.Vesper_Sparrow', '132.White_crowned_Sparrow', '133.White_throated_Sparrow', '134.Cape_Glossy_Starling',  '136.Barn_Swallow', '137.Cliff_Swallow', '138.Tree_Swallow', '139.Scarlet_Tanager', '140.Summer_Tanager', '142.Black_Tern', '143.Caspian_Tern', '144.Common_Tern', '145.Elegant_Tern', '146.Forsters_Tern', '147.Least_Tern', '148.Green_tailed_Towhee',  '150.Sage_Thrasher',  '152.Blue_headed_Vireo',  '154.Red_eyed_Vireo', '155.Warbling_Vireo', '156.White_eyed_Vireo',  '158.Bay_breasted_Warbler', '159.Black_and_white_Warbler',  '161.Blue_winged_Warbler', '162.Canada_Warbler', '163.Cape_May_Warbler', '164.Cerulean_Warbler', '165.Chestnut_sided_Warbler', '167.Hooded_Warbler', '170.Mourning_Warbler', '171.Myrtle_Warbler', '172.Nashville_Warbler', '173.Orange_crowned_Warbler', '174.Palm_Warbler', '175.Pine_Warbler', '176.Prairie_Warbler', '177.Prothonotary_Warbler', '180.Wilson_Warbler',  '182.Yellow_Warbler', '183.Northern_Waterthrush', '184.Louisiana_Waterthrush', '185.Bohemian_Waxwing', '186.Cedar_Waxwing',  '188.Pileated_Woodpecker', '189.Red_bellied_Woodpecker', '190.Red_cockaded_Woodpecker', '191.Red_headed_Woodpecker', '192.Downy_Woodpecker', '193.Bewick_Wren', '194.Cactus_Wren', '195.Carolina_Wren', '197.Marsh_Wren', '198.Rock_Wren', '199.Winter_Wren', '200.Common_Yellowthroat']

for name in lisxtx:
    path = '/home/ubuntu/tf_files/train/'+ name +'/'
    files = os.listdir(path)
    i = 1
    print(path)
    for file in files:
    
        os.rename(os.path.join(path, file), os.path.join(path, str(i)+'.jpg'))
        im = Image.open(path+str(i)+'.jpg')
        im.rotate(180).save(path+str(i)+'rotate.jpg')
    
        im.transpose(Image.FLIP_LEFT_RIGHT).save(path+str(i)+"FLIPLR.jpg")
        im.transpose(Image.FLIP_TOP_BOTTOM).save(path+str(i)+"FLIPtb.jpg")
        im.filter(ImageFilter.GaussianBlur(5)).save(path+str(i)+"blur.jpg")
        i = i+1



