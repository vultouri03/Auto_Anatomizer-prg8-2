This is the Auto anatomizer, a program that can be use to translate you own body into a sketch that can be used when making art.

it uses Media pipe for pose landmarking and an ML5 neural network in order to check if you are striking the correct pose.

INSTALATION

In order to use this program on your own device you only need to download this repository to you pc and then launch the index.html file. All necesarry libraries and frameworks are already integrated into the app.

If you don't want to download it the full application can be used on this link: https://vultouri03.github.io/Auto_Anatomizer-prg8-2/

Known issues

when opening the console an error might appear telling you that the array to classify a pose is empty, this is because of the fact that when opening the app there is one frame where the NN is faster than the pose landmarker causing one pose to classify to empty. this has no consequences for the working of the app.

The same issue happens with some other console errors related to that same issue, they don't interfere with the working of the app and are only visual in the console
