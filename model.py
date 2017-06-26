from keras.models import Sequential,load_model
from keras.layers import Lambda, Conv2D, Dropout, MaxPool2D, Dense, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import plot_model
import dataset as ds
import utils
import matplotlib.pyplot as plt

models_path = './models/'
output_images = './'

def cornerFinder(input_shape, name='corner_cnn', load_weights=None, debug=False):

    model = Sequential(name=name)
    model.add(Lambda(lambda x: x / 255, input_shape=input_shape))  # normalize
    model.add(Conv2D(8, kernel_size=(5, 5), activation="relu", strides=(2,2) ))
    model.add(Conv2D(16, kernel_size=(5, 5), activation="relu", strides=(2,2) ))
    model.add(Conv2D(32, kernel_size=(5, 5), activation="relu", strides=(2,2) ))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(8,activation="sigmoid"))


    if load_weights is not None:
        print('Loading weights', load_weights)
        model.load_weights(load_weights)
    else:
        print('Loading weights failed', load_weights)

    if debug:
        model_img_path = output_images + model.name + '.png'
        plot_model(model, to_file=model_img_path, show_shapes=True)
        utils.showImage(model_img_path)

    return model


def train(dataset, epochs=30, batch_size=32, load_weights=None, debug=False):
    timestamp = utils.standardDatetime()
    # load dataset generator and metrics
    gen_train, gen_valid, info = ds.loadDatasetGenerators(dataset, batch_size=batch_size)
    print(info)

    # create the model a eventually preload the weights (set to None or remove to disable)
    model = cornerFinder(input_shape=info['input_shape'], load_weights=load_weights, debug=debug)
    model_name = model.name +'_'+ timestamp

    # Intermediate model filename template

    filepath = models_path + model_name + "_{epoch:02d}-{val_loss:.5f}.h5"
    # save model after every epoc, only if improved the val_loss.
    # very handy (with a proper environment (2 GPUs anywhere) you can test your model while it still train)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    # detect stop in gain on val_loss between epocs and terminate early, avoiding unnecessary computation cycles.
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_grads=True, write_images=True)
    callbacks_list = [checkpoint, earlystopping, tensorboard]

    model.compile(loss="mse", optimizer="adam")
    history_object = model.fit_generator(gen_train, info['n_train_batch'], verbose=1, epochs=epochs,
                                         validation_data=gen_valid, validation_steps=info['n_valid_batch'],
                                         callbacks=callbacks_list)

    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error accuracy')
    plt.ylabel('mean squared error accuracy')
    plt.xlabel('epoch')
    plt.legend(['Training accuracy', 'Validation accuracy'], loc='upper right')
    plt.show()

    return model_name

def loadModel(path=None):
    if path is None:
        return loadLastModel()
    else:
        return load_model(path)

def lastModel():
    return utils.lastFile(models_path+'*.h5')

def loadLastModel():
    return loadModel(lastModel())