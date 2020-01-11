using FileIO, Images
img = load("Examples/Cactus/train/0a1b6731bda8f1b6a807fffc743f8d22.jpg")
size(img)
arr = channelview(img)
size(arr)
