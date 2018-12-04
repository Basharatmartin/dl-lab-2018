import numpy as np

LEFT =1
RIGHT = 2
STRAIGHT = 0
ACCELERATE =3
BRAKE = 4

def one_hot(labels):
    """
    this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(np.shape(labels) + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1.0
    return one_hot_labels

def rgb2gray(rgb):
    """ 
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    #gray = 2 * gray.astype('float32') - 1
    return gray.astype('float32') 

def action_to_id(a):
    """ 
    this method discretizes the actions.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    if all(a == [-1.0, 0.0, 0.0]): return LEFT               # LEFT: 1
    elif all(a == [1.0, 0.0, 0.0]): return RIGHT             # RIGHT: 2
    elif all(a == [0.0, 1.0, 0.0]): return ACCELERATE        # ACCELERATE: 3
    elif all(a == [0.0, 0.0, 0.2]): return BRAKE             # BRAKE: 4
    else:       
        return STRAIGHT                                      # STRAIGHT = 0

def id_to_action(a):
	
	nr_classes = 3 
	labels_action = np.zeros((a.shape[0], nr_classes))
	labels_action[a==LEFT] = [-1.0, 0.0, 0.0]
	labels_action[a==RIGHT] = [1.0, 0.0, 0.0]
	labels_action[a==STRAIGHT] = [0.0, 0.0, 0.0] 	## Accelerate too
	labels_action[a==ACCELERATE] =[0.0, 1.0, 0.0]
	labels_action[a==BRAKE] = [0.0, 0.0, 0.2]

	return labels_action

