import numpy as np
import nltk
from PIL import Image

from pickle import load
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.sequence import pad_sequences

from keras.models import load_model

import matplotlib.pyplot as plt
import argparse

import warnings
warnings.filterwarnings("ignore")


# Inspired from Data-Flair Tutorial
def extract_features(img_path):
    try:
        image = Image.open(img_path) # try to open the image
    except:
        print("Error, cannot find image!") # if fails, catch error and return a print statement

    image_model = InceptionV3(include_top = False, pooling='avg')
    og_image = Image.open(img_path) # open it 
    image = og_image.resize((299,299)) # resize it to be an image of 299, 299
        
    image = np.expand_dims(image, axis=0) / 255 # add an extra dimension and convert pixel RGB values to decimals
        
    feature = image_model.predict(image) # extracts relevant features
    return og_image, feature

# Inspired from Data-Flair Tutorial
def next_word(prediction, tokenizer): # used to find word in the word_index dictionary,
    for word, index in tokenizer.word_index.items():
        if prediction == index:
            return word
    return None # return none if word does not exist


# inspired from Data-Flair Tutorial
def generate_caption(img_caption_model, tokenizer, feature_vector, max_length):
    caption = 'start' # caption will start with start
    i = 0
    while i < max_length: # 
        sequence = tokenizer.texts_to_sequences([caption])[0] # convert caption to a sequence
        sequence = pad_sequences([sequence], maxlen=max_length) # make the sequence an array of length 33
        prediction = img_caption_model.predict([feature_vector, sequence], verbose=0) # predict what word needs to be generated next based on our feature vector and sequence
        prediction = np.argmax(prediction) # choose the most likely prediction
        word_to_add = next_word(prediction, tokenizer) # see if it exists in our dictionary 
        if word_to_add is None: # if it does not, leave forloop
            break
        elif word_to_add == 'end': # if the word is end, add that to our caption and leave forloop
            caption += ' ' + 'end'
            break
        else: 
            caption += ' ' + word_to_add # add the word and keep on going
        i += 1
    return caption # return the completed caption


def evaluation(reference, generated):
    reference = reference.split() # splits into array
    #print(reference)
    generated = generated.split() # splits into array
    
    generated = generated[1:] # removes start
    generated = generated[:-1] # removes end
    #print(generated)
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], generated, weights=(1,0,0,0)) # Gives us the BLEU Score for Evaluating our Function
    return BLEUscore

def main(args):
    img_path = args.image # get image path
    print(img_path)
    reference_caption = ' '.join(args.reference) # get reference caption
    print(reference_caption)
    max_length = 33 # max length of caption
    tokenizer = load(open('tokenizer.p', 'rb')) # get tokenizer
    img_caption_model = load_model('models/model_9.h5') # best trained model
    image, feature_vector = extract_features(img_path) # get original image, and the feature vector associated with that image
    caption = generate_caption(img_caption_model, tokenizer, feature_vector, max_length) # generate the caption

    print(f"Reference Caption: {reference_caption}")
    print(f"Caption Generated: {caption}")
    print(f"BLEU Score: {evaluation(reference_caption, caption)}")

    
    plt.imshow(image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser() # used to get Command line elements
    parser.add_argument('--image', type=str, required=True, help='Image Path') # allows user to add an image path
    parser.add_argument('--reference', type=str, required=True, help='Reference Caption', nargs='+') # allows user to add a reference caption
    args = parser.parse_args() # allows the arguements to be passed into main
    main(args)