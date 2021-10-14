from tensorflow.keras.layers import (
    Input,
    Conv1D,
    Conv2D,
    Conv3D,
    MaxPooling1D,
    MaxPooling2D,
    MaxPool3D,
    UpSampling3D,
    UpSampling2D,
    Dense,
    ZeroPadding2D,
    ZeroPadding3D,
    Flatten,
    DepthwiseConv2D,
    Dropout,
    Reshape,
)
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras.layers import dot
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from deepinterpolation.generic import JsonLoader


def autoencoder_single_256(path_json):
    def local_network_function(input_img):
        # encoder
        # input = 512 x 512 x number_img_in (wide and thin)
        conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(
            input_img
        )  # 512 x 512 x 32
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
        conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(
            pool1
        )  # 256 x 256 x 64
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64#
        conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(
            pool2
        )  # 128 x 128 x 128 (small and thick)

        # decoder
        conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(
            conv3
        )  # 128 x 128 x 128
        up1 = UpSampling2D((2, 2))(conv4)  # 14 x 14 x 128
        conv5 = Conv2D(128, (3, 3), activation="relu", padding="same")(
            up1
        )  # 256 x 256 x 64
        up2 = UpSampling2D((2, 2))(conv5)  # 28 x 28 x 64
        decoded = Conv2D(1, (3, 3), activation=None, padding="same")(
            up2
        )  # 512 x 512 x 1

        return decoded

    return local_network_function


def unet_single_256(path_json):
    def local_network_function(input_img):
        # encoder
        # input = 512 x 512 x number_img_in (wide and thin)
        conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(
            input_img
        )  # 512 x 512 x 32
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
        conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(
            pool1
        )  # 256 x 256 x 64
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64#
        conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(
            pool2
        )  # 128 x 128 x 128 (small and thick)

        # decoder
        conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(
            conv3
        )  # 128 x 128 x 128
        up1 = UpSampling2D((2, 2))(conv4)  # 14 x 14 x 128

        conc_up_1 = Concatenate()([up1, conv2])
        conv5 = Conv2D(128, (3, 3), activation="relu", padding="same")(
            up1
        )  # 256 x 256 x 64
        up2 = UpSampling2D((2, 2))(conv5)  # 28 x 28 x 64

        conc_up_2 = Concatenate()([up2, conv1])
        decoded = Conv2D(1, (3, 3), activation=None, padding="same")(
            conc_up_2
        )  # 512 x 512 x

        return decoded

    return local_network_function


def fmri_unet_denoiser(path_json):
    def local_network_function(input_img):
        # encoder
        conv1 = Conv3D(8, (3, 3, 3), activation="relu", padding="same")(input_img)
        pool1 = MaxPool3D(pool_size=(2, 2, 2))(conv1)

        conv2 = Conv3D(16, (3, 3, 3), activation="relu", padding="same")(pool1)
        pool2 = MaxPool3D(pool_size=(2, 2, 2))(conv2)

        conv3 = Conv3D(32, (3, 3, 3), activation="relu", padding="same")(pool2)

        # decoder
        up1 = UpSampling3D((2, 2, 2))(conv3)
        up1 = ZeroPadding3D(padding=((0, 1), (0, 1), (0, 1)))(up1)

        conc_up_1 = Concatenate()([up1, conv2])

        conv4 = Conv3D(16, (3, 3, 3), activation="relu", padding="same")(conc_up_1)

        up2 = UpSampling3D((2, 2, 2))(conv4)
        up2 = ZeroPadding3D(padding=((0, 1), (0, 1), (0, 1)))(up2)

        conc_up_2 = Concatenate()([up2, conv1])

        conv5 = Conv3D(8, (3, 3, 3), activation="relu", padding="same")(conc_up_2)

        out = Conv3D(1, (1, 1, 1), activation=None, padding="same")(conv5)
        return out

    return local_network_function


def fmri_flexible_architecture(path_json):
    def local_network_function(input_img, hp):
        # encoder
        in_conv = input_img
        out_conv = input_img

        broad_activation = hp.Choice("unit_activation", values=["relu", "elu"])

        for nb_conv in range(hp.Choice(f"nb_conv_layers", values=[0, 1, 2])):
            conv_interm = Conv3D(
                hp.Choice(
                    f"conv_{nb_conv}_units", values=[32, 64, 128, 256], default=64
                ),
                (2, 2, 2),
                activation=broad_activation,
                padding="same",
            )(in_conv)
            out_conv = MaxPool3D(pool_size=(2, 2, 2))(conv_interm)
            in_conv = out_conv

        in_dense = out_conv

        for nb_dense in range(hp.Choice(f"nb_dense_layers", values=[2, 4, 6])):
            out_dense = Dense(
                hp.Choice(
                    f"dense_{nb_dense}_units", values=[32, 64, 128, 256], default=128
                ),
                activation=broad_activation,
            )(in_dense)
            in_dense = out_dense

        final = Dense(1, activation=None)(out_dense)

        return final

    return local_network_function


def fmri_volume_optimized_denoiser(path_json):
    def local_network_function(input_img):

        # encoder
        conv1 = Conv3D(256, (2, 2, 2), activation="relu", padding="same")(input_img)
        pool1 = MaxPool3D(pool_size=(2, 2, 2))(conv1)
        conv2 = Conv3D(128, (2, 2, 2), activation="relu", padding="same")(pool1)
        pool2 = MaxPool3D(pool_size=(2, 2, 2))(conv2)
        dens1 = Dense(64, activation="relu")(pool2)
        dens2 = Dense(32, activation="relu")(dens1)
        dens3 = Dense(64, activation="relu")(dens2)
        dens4 = Dense(64, activation="relu")(dens3)
        dens5 = Dense(64, activation="relu")(dens4)
        dens6 = Dense(64, activation="relu")(dens5)

        dense_out = Dense(1, activation=None)(dens6)

        return dense_out

    return local_network_function


def fmri_volume_deeper_denoiser(path_json):
    def local_network_function(input_img):

        # encoder
        conv1 = Conv3D(32, (2, 2, 2), activation="relu", padding="same")(input_img)
        pool1 = MaxPool3D(pool_size=(2, 2, 2))(conv1)
        conv2 = Conv3D(64, (2, 2, 2), activation="relu", padding="same")(pool1)
        pool2 = MaxPool3D(pool_size=(2, 2, 2))(conv2)
        dens1 = Dense(128, activation="relu")(pool2)
        dens2 = Dense(128, activation="relu")(dens1)
        dens3 = Dense(128, activation="relu")(dens2)
        dens4 = Dense(128, activation="relu")(dens3)

        dense_out = Dense(1, activation=None)(dens4)

        return dense_out

    return local_network_function


def fmri_volume_dense_denoiser(path_json):
    def local_network_function(input_img):

        # encoder
        conv1 = Conv3D(32, (2, 2, 2), activation="relu", padding="same")(input_img)
        pool1 = MaxPool3D(pool_size=(2, 2, 2))(conv1)
        conv2 = Conv3D(64, (2, 2, 2), activation="relu", padding="same")(pool1)
        pool2 = MaxPool3D(pool_size=(2, 2, 2))(conv2)
        dens1 = Dense(128, activation="relu")(pool2)
        dens2 = Dense(128, activation="relu")(dens1)

        dense_out = Dense(1, activation=None)(dens2)

        return dense_out

    return local_network_function


def fmri_volume_denoiser(path_json):
    def local_network_function(input_img):

        # encoder
        conv1 = Conv3D(32, (2, 2, 2), activation="relu", padding="same")(input_img)
        pool1 = MaxPool3D(pool_size=(2, 2, 2))(conv1)
        conv2 = Conv3D(64, (2, 2, 2), activation="relu", padding="same")(pool1)
        pool2 = MaxPool3D(pool_size=(2, 2, 2))(conv2)
        conv3 = Conv3D(128, (2, 2, 2), activation="relu", padding="same")(pool2)
        dens1 = Dense(128, activation="relu")(conv3)
        dens2 = Dense(128, activation="relu")(dens1)

        dense_out = Dense(1, activation=None)(dens2)

        return dense_out

    return local_network_function


def unet_single_ephys_1024(path_json):
    def local_network_function(input_img):

        # encoder
        # input = 512 x 512 x number_img_in (wide and thin)
        conv1 = Conv2D(64, (2, 2), activation="relu", padding="same")(
            input_img
        )  # 512 x 512 x 32
        # CHANGE pool_size from (2,2) to (2,1)
        pool1 = MaxPooling2D(pool_size=(2, 1))(conv1)  # 14 x 14 x 32
        conv2 = Conv2D(128, (3, 1), activation="relu", padding="same")(
            pool1
        )  # 256 x 256 x 64
        pool2 = MaxPooling2D(pool_size=(2, 1))(conv2)  # 7 x 7 x 64#
        conv3 = Conv2D(256, (3, 1), activation="relu", padding="same")(
            pool2
        )  # 128 x 128 x 128 (small and thick)
        pool3 = MaxPooling2D(pool_size=(2, 1))(conv3)
        conv4 = Conv2D(512, (3, 1), activation="relu", padding="same")(
            pool3
        )  # 128 x 128 x 128 (small and thick)
        pool4 = MaxPooling2D(pool_size=(2, 1))(conv4)

        conv5 = Conv2D(1024, (3, 1), activation="relu", padding="same")(
            pool4
        )  # 128 x 128 x 128 (small and thick)

        # decoder
        up1 = UpSampling2D((2, 1))(conv5)  # 14 x 14 x 128
        conc_up_1 = Concatenate()([up1, conv4])
        conv7 = Conv2D(512, (3, 1), activation="relu", padding="same")(
            conc_up_1
        )  # 256 x 256 x 64
        up2 = UpSampling2D((2, 1))(conv7)  # 28 x 28 x 64
        conc_up_2 = Concatenate()([up2, conv3])
        conv8 = Conv2D(256, (3, 1), activation="relu", padding="same")(
            conc_up_2
        )  # 512 x 512 x 1
        up3 = UpSampling2D((2, 1))(conv8)  # 28 x 28 x 64
        conc_up_3 = Concatenate()([up3, conv2])
        conv9 = Conv2D(128, (3, 1), activation="relu", padding="same")(
            conc_up_3
        )  # 512 x 512 x 1
        up4 = UpSampling2D((2, 2))(conv9)  # 28 x 28 x 64
        conc_up_4 = Concatenate()([up4, conv1])
        conv10 = Conv2D(64, (2, 2), activation="relu", padding="same")(
            conc_up_4
        )  # 512 x 512 x 1
        decoded = Conv2D(1, (1, 1), activation=None, padding="same")(
            conv10
        )  # 512 x 512 x 1

        return decoded

    return local_network_function

def unet_tetrode(path_json):
    def local_network_function(input_img):

        # encoder
        # input = batches x frames x chans
        print(f'input_img shape: {input_img.shape}')

        conv1 = Conv2D(64, 2, activation="relu", padding="same", input_shape=input_img.shape[1:])(
            input_img)
        print(f'conv1 shape: {conv1.shape}')
 
         # 512 x 512 x 32
        # CHANGE pool_size from (2,2) to (2,1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
        print(f'pool1 shape: {pool1.shape}')
        conv2 = Conv2D(128, (3, 1), activation="relu", padding="same")(
            pool1
        )  # 256 x 256 x 64
        print(f'conv2 shape: {conv2.shape}')
        pool2 = MaxPooling2D(pool_size=(2, 1))(conv2)  # 7 x 7 x 64#
        print(f'pool2 shape: {pool2.shape}')

        conv3 = Conv2D(256, (3, 1), activation="relu", padding="same")(
            pool2
        )  # 128 x 128 x 128 (small and thick)
        print(f'conv3 shape: {conv3.shape}')

        pool3 = MaxPooling2D(pool_size=(2, 1))(conv3)
        print(f'pool3 shape: {pool3.shape}')

        conv4 = Conv2D(512, (2, 1), activation="relu", padding="same")(
            pool3
        )  # 128 x 128 x 128 (small and thick)
        print(f'conv4 shape: {conv4.shape}')

        pool4 = MaxPooling2D(pool_size=(2, 1))(conv4)
        print(f'pool4 shape: {pool4.shape}')

        conv5 = Conv2D(1024, (2, 1), activation="relu", padding="same")(
            pool4
        )  # 128 x 128 x 128 (small and thick)
        print(f'conv5 shape: {conv5.shape}')

        # decoder
        up1 = UpSampling2D((2, 1))(conv5)  # 14 x 14 x 128
        print(f'up1 shape: {up1.shape}')
        conc_up_1 = Concatenate()([up1, conv4])
        conv7 = Conv2D(512, (3, 1), activation="relu", padding="same")(
            conc_up_1
        )  # 256 x 256 x 64
        up2 = UpSampling2D((2, 1))(conv7)  # 28 x 28 x 64
        conc_up_2 = Concatenate()([up2, conv3])
        conv8 = Conv2D(256, (3, 1), activation="relu", padding="same")(
            conc_up_2
        )  # 512 x 512 x 1
        up3 = UpSampling2D((2, 1))(conv8)  # 28 x 28 x 64
        conc_up_3 = Concatenate()([up3, conv2])
        conv9 = Conv2D(128, (3, 1), activation="relu", padding="same")(
            conc_up_3
        )  # 512 x 512 x 1
        up4 = UpSampling2D((2, 2))(conv9)  # 28 x 28 x 64
        conc_up_4 = Concatenate()([up4, conv1])
        conv10 = Conv2D(64, (2, 2), activation="relu", padding="same")(
            conc_up_4
        )  # 512 x 512 x 1
        decoded = Conv2D(1, (1, 1), activation=None, padding="same")(
            conv10
        )  # 512 x 512 x 1

        return decoded

    return local_network_function

def unet_tetrode2(path_json):
    def local_network_function(input_img):

        # encoder
        # input = batches x frames x chans
        print(f'input_img shape: {input_img.shape}')

        conv1 = Conv2D(64, 4, activation="relu", padding="same", input_shape=input_img.shape[1:])(
            input_img)
        print(f'first layer (conv1) shape: {conv1.shape}')
 
        pool1 = MaxPooling2D(pool_size=(4, 2))(conv1)  # 14 x 14 x 32
        print(f'second layer (pool1) shape: {pool1.shape}')
        conv2 = Conv2D(128, (4, 1), activation="relu", padding="same")(
            pool1
        ) 
        print(f'third layer (conv2) shape: {conv2.shape}')
        pool2 = MaxPooling2D(pool_size=(4, 1))(conv2) 
        print(f'pool2 shape: {pool2.shape}')

        # conv3 = Conv2D(256, (3, 1), activation="relu", padding="same")(
        #     pool2
        # )  # 128 x 128 x 128 (small and thick)
        # print(f'conv3 shape: {conv3.shape}')

        # pool3 = MaxPooling2D(pool_size=(2, 1))(conv3)
        # print(f'pool3 shape: {pool3.shape}')

        # conv4 = Conv2D(512, (2, 1), activation="relu", padding="same")(
        #     pool3
        # )  # 128 x 128 x 128 (small and thick)
        # print(f'conv4 shape: {conv4.shape}')

        # pool4 = MaxPooling2D(pool_size=(2, 1))(conv4)
        # print(f'pool4 shape: {pool4.shape}')

        conv5 = Conv2D(256, (2, 1), activation="relu", padding="same")(
            pool2
        )  # 128 x 128 x 128 (small and thick)
        print(f'conv5 shape: {conv5.shape}')

        # print(f'conv5 shape: {conv5.shape}')

        # decoder
        # up1 = UpSampling2D((2, 1))(conv5)  # 14 x 14 x 128
        # print(f'up1 shape: {up1.shape}')
        # conc_up_1 = Concatenate()([up1, conv4])
        # conv7 = Conv2D(512, (3, 1), activation="relu", padding="same")(
        #     conc_up_1
        # )  # 256 x 256 x 64
        # up2 = UpSampling2D((2, 1))(conv7)  # 28 x 28 x 64
        # conc_up_2 = Concatenate()([up2, conv3])
        # conv8 = Conv2D(256, (3, 1), activation="relu", padding="same")(
        #     conc_up_2
        # )  # 512 x 512 x 1
        up3 = UpSampling2D((4, 1))(conv5)  # 28 x 28 x 64
        print(f'up3 shape: {up3.shape}')

        conc_up_3 = Concatenate()([up3, conv2])
        print(f'conc_up_3 shape: {conc_up_3.shape}')

        conv9 = Conv2D(128, (4, 1), activation="relu", padding="same")(
            conc_up_3
        )  # 512 x 512 x 1
        print(f'conv9 shape: {conv9.shape}')

        up4 = UpSampling2D((4, 2))(conv9)  # 28 x 28 x 64
        print(f'up4 shape: {up4.shape}')

        conc_up_4 = Concatenate()([up4, conv1])
        print(f'conc_up_4 shape: {conc_up_4.shape}')

        conv10 = Conv2D(64, (4, 4), activation="relu", padding="same")(
            conc_up_4
        )  # 512 x 512 x 1
        print(f'conv10 shape: {conv10.shape}')

        decoded = Conv2D(1, (1, 1), activation=None, padding="same")(
            conv10
        )  # 512 x 512 x 1
        print(f'decoded shape: {decoded.shape}')

        return decoded

    return local_network_function

def padding_unet_single_1024(path_json):
    def local_network_function(input_img):

        # encoder
        # input = 512 x 512 x number_img_in (wide and thin)

        conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(
            input_img
        )  # 512 x 512 x 32
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
        conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(
            pool1
        )  # 256 x 256 x 64
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64#
        conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(
            pool2
        )  # 128 x 128 x 128 (small and thick)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(
            pool3
        )  # 128 x 128 x 128 (small and thick)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        conv5 = Conv2D(1024, (3, 3), activation="relu", padding="same")(
            pool4
        )  # 128 x 128 x 128 (small and thick)

        # decoder
        up1 = UpSampling2D((2, 2))(conv5)  # 14 x 14 x 128

        def pad_layers_before_concatenate(smaller_layer, larger_layer):
            add_dim_width = int(larger_layer.shape[1]) - int(smaller_layer.shape[1])
            add_dim_height = int(larger_layer.shape[1]) - int(smaller_layer.shape[1])
            smaller_layer = ZeroPadding2D(
                padding=((0, add_dim_width), (0, add_dim_height))
            )(smaller_layer)

            return smaller_layer

        up1 = pad_layers_before_concatenate(up1, conv4)
        conc_up_1 = Concatenate()([up1, conv4])
        conv7 = Conv2D(512, (3, 3), activation="relu", padding="same")(
            conc_up_1
        )  # 256 x 256 x 64
        up2 = UpSampling2D((2, 2))(conv7)  # 28 x 28 x 64

        up2 = pad_layers_before_concatenate(up2, conv3)
        conc_up_2 = Concatenate()([up2, conv3])
        conv8 = Conv2D(256, (3, 3), activation="relu", padding="same")(
            conc_up_2
        )  # 512 x 512 x 1
        up3 = UpSampling2D((2, 2))(conv8)  # 28 x 28 x 64

        up3 = pad_layers_before_concatenate(up3, conv2)
        conc_up_3 = Concatenate()([up3, conv2])
        conv9 = Conv2D(128, (3, 3), activation="relu", padding="same")(
            conc_up_3
        )  # 512 x 512 x 1
        up4 = UpSampling2D((2, 2))(conv9)  # 28 x 28 x 64

        up4 = pad_layers_before_concatenate(up4, conv1)
        conc_up_4 = Concatenate()([up4, conv1])
        conv10 = Conv2D(64, (3, 3), activation="relu", padding="same")(
            conc_up_4
        )  # 512 x 512 x 1
        decoded = Conv2D(1, (1, 1), activation=None, padding="same")(
            conv10
        )  # 512 x 512 x 1

        return decoded

    return local_network_function


def unet_1024_search(path_json):
    local_json_loader = JsonLoader(path_json)
    local_json_loader.load_json()
    json_data = local_json_loader.json_data

    def local_network_function(input_img):

        # encoder
        local_input = input_img
        for local_depth in range(json_data["network_depth"]):
            local_conv = Conv2D(
                2 ** local_depth * json_data["nb_features_scale"],
                (3, 3),
                activation="relu",
                padding="same",
            )(local_input)
            local_output = MaxPooling2D(pool_size=(2, 2))(local_conv)
            if json_data["unet"]:
                if local_depth == 0:
                    u_net_shortcut = []
                u_net_shortcut.append(local_conv)
            local_input = local_output

        # Deep CONV
        deep_conv = Conv2D(
            2 ** json_data["network_depth"] * json_data["nb_features_scale"],
            (3, 3),
            activation="relu",
            padding="same",
        )(local_input)

        # decoder
        local_input = deep_conv
        for local_depth in range(json_data["network_depth"] - 1, -1, -1):
            local_up = UpSampling2D((2, 2))(local_input)
            if json_data["unet"]:
                local_conc = Concatenate()([local_up, u_net_shortcut[local_depth]])
            else:
                local_conc = local_up

            local_output = Conv2D(
                2 ** local_depth * json_data["nb_features_scale"],
                (3, 3),
                activation="relu",
                padding="same",
            )(local_conc)
            local_input = local_output

        # output layer
        final = Conv2D(1, (1, 1), activation=None, padding="same")(local_output)

        return final

    return local_network_function


def unet_single_1024(path_json):
    def local_network_function(input_img):

        # encoder
        # input = 512 x 512 x number_img_in (wide and thin)
        conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(
            input_img
        )  # 512 x 512 x 32
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
        conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(
            pool1
        )  # 256 x 256 x 64
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64#
        conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(
            pool2
        )  # 128 x 128 x 128 (small and thick)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(
            pool3
        )  # 128 x 128 x 128 (small and thick)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        conv5 = Conv2D(1024, (3, 3), activation="relu", padding="same")(
            pool4
        )  # 128 x 128 x 128 (small and thick)

        # decoder
        up1 = UpSampling2D((2, 2))(conv5)  # 14 x 14 x 128
        conc_up_1 = Concatenate()([up1, conv4])
        conv7 = Conv2D(512, (3, 3), activation="relu", padding="same")(
            conc_up_1
        )  # 256 x 256 x 64
        up2 = UpSampling2D((2, 2))(conv7)  # 28 x 28 x 64
        conc_up_2 = Concatenate()([up2, conv3])
        conv8 = Conv2D(256, (3, 3), activation="relu", padding="same")(
            conc_up_2
        )  # 512 x 512 x 1
        up3 = UpSampling2D((2, 2))(conv8)  # 28 x 28 x 64
        conc_up_3 = Concatenate()([up3, conv2])
        conv9 = Conv2D(128, (3, 3), activation="relu", padding="same")(
            conc_up_3
        )  # 512 x 512 x 1
        up4 = UpSampling2D((2, 2))(conv9)  # 28 x 28 x 64
        conc_up_4 = Concatenate()([up4, conv1])
        conv10 = Conv2D(64, (3, 3), activation="relu", padding="same")(
            conc_up_4
        )  # 512 x 512 x 1
        decoded = Conv2D(1, (1, 1), activation=None, padding="same")(
            conv10
        )  # 512 x 512 x 1

        return decoded

    return local_network_function


def segmentation_net(path_json):
    def local_network_function(input_img):

        # encoder
        conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(input_img)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool3)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        conv5 = Conv2D(1024, (3, 3), activation="relu", padding="same")(pool4)

        latent_layer = Dense(100)(conv5)
        latent_activation = Dense(100)(latent_layer)

        # neuronal projection

        up1 = UpSampling2D((2, 2))(conv5)  # 14 x 14 x 128
        conc_up_1 = Concatenate()([up1, conv4])
        conv7 = Conv2D(512, (3, 3), activation="relu", padding="same")(
            conc_up_1
        )  # 256 x 256 x 64
        conv7 = Conv2D(512, (3, 3), activation="relu", padding="same")(
            conv7
        )  # 256 x 256 x 64

        up2 = UpSampling2D((2, 2))(conv7)  # 28 x 28 x 64
        conc_up_2 = Concatenate()([up2, conv3])
        conv8 = Conv2D(256, (3, 3), activation="relu", padding="same")(
            conc_up_2
        )  # 512 x 512 x 1
        conv8 = Conv2D(256, (3, 3), activation="relu", padding="same")(
            conv8
        )  # 512 x 512 x 1

        up3 = UpSampling2D((2, 2))(conv8)  # 28 x 28 x 64
        conc_up_3 = Concatenate()([up3, conv2])
        conv9 = Conv2D(128, (3, 3), activation="relu", padding="same")(
            conc_up_3
        )  # 512 x 512 x 1
        conv9 = Conv2D(128, (3, 3), activation="relu", padding="same")(
            conv9
        )  # 512 x 512 x 1

        up4 = UpSampling2D((2, 2))(conv9)  # 28 x 28 x 64
        conc_up_4 = Concatenate()([up4, conv1])
        conv10 = Conv2D(64, (3, 3), activation="relu", padding="same")(
            conc_up_4
        )  # 512 x 512 x 1
        conv10 = Conv2D(64, (3, 3), activation="relu", padding="same")(
            conv10
        )  # 512 x 512 x 1

        upsample = Conv2D(
            1, (1, 1), activation=None, padding="same", name="output_upsample"
        )(
            conv10
        )  # 512 x 512 x 1
        decoded = Conv2D(
            1,
            (10, 10),
            activation=None,
            padding="same",
            name="output_raw",
            use_bias=False,
            kernel_regularizer=regularizers.l2(0.01),
        )(
            upsample
        )  # 512 x 512 x 1

        return [upsample, decoded]

    return local_network_function


def unet_single_1p_1024(path_json):
    def local_network_function(input_img):

        # encoder
        # input = 512 x 512 x number_img_in (wide and thin)
        conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(
            input_img
        )  # 512 x 512 x 32
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32
        conv2 = Conv2D(128, (3, 3), activation="relu", padding="same")(
            pool1
        )  # 256 x 256 x 64
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x 7 x 64#
        conv3 = Conv2D(256, (3, 3), activation="relu", padding="same")(
            pool2
        )  # 128 x 128 x 128 (small and thick)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, (3, 3), activation="relu", padding="same")(
            pool3
        )  # 128 x 128 x 128 (small and thick)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        conv5 = Conv2D(1024, (3, 3), activation="relu", padding="same")(
            pool4
        )  # 128 x 128 x 128 (small and thick)

        # decoder
        up1 = UpSampling2D((2, 2))(conv5)  # 14 x 14 x 128

        up1 = ZeroPadding2D(padding=((1, 0), (1, 0)))(up1)

        conc_up_1 = Concatenate()([up1, conv4])
        conv7 = Conv2D(512, (3, 3), activation="relu", padding="same")(
            conc_up_1
        )  # 256 x 256 x 64
        up2 = UpSampling2D((2, 2))(conv7)  # 28 x 28 x 64

        up2 = ZeroPadding2D(padding=((1, 0), (1, 0)))(up2)

        conc_up_2 = Concatenate()([up2, conv3])
        conv8 = Conv2D(256, (3, 3), activation="relu", padding="same")(
            conc_up_2
        )  # 512 x 512 x 1
        up3 = UpSampling2D((2, 2))(conv8)  # 28 x 28 x 64

        up3 = ZeroPadding2D(padding=((1, 0), (1, 0)))(up3)

        conc_up_3 = Concatenate()([up3, conv2])
        conv9 = Conv2D(128, (3, 3), activation="relu", padding="same")(
            conc_up_3
        )  # 512 x 512 x 1
        up4 = UpSampling2D((2, 2))(conv9)  # 28 x 28 x 64

        conc_up_4 = Concatenate()([up4, conv1])
        conv10 = Conv2D(64, (3, 3), activation="relu", padding="same")(
            conc_up_4
        )  # 512 x 512 x 1
        decoded = Conv2D(1, (1, 1), activation=None, padding="same")(
            conv10
        )  # 512 x 512 x 1

        return decoded

    return local_network_function


def unet_double_1024(path_json):
    def local_network_function(input_img):

        # encoder
        # input = 512 x 512 x number_img_in (wide and thin)
        conv1 = Conv2D(64, (3, 3), activation="relu", padding="same")(
            input_img
        )  # 512 x 512 x 32
        conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(
            conv1
        )  # 512 x 512 x 32

        pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 14 x 14 x 32

        conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(
            pool1
        )  # 256 x 256 x 64
        conv4 = Conv2D(128, (3, 3), activation="relu", padding="same")(
            conv3
        )  # 256 x 256 x 64

        pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)  # 7 x 7 x 64#

        conv5 = Conv2D(256, (3, 3), activation="relu", padding="same")(
            pool2
        )  # 128 x 128 x 128 (small and thick)
        conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(
            conv5
        )  # 128 x 128 x 128 (small and thick)

        pool3 = MaxPooling2D(pool_size=(2, 2))(conv6)

        conv7 = Conv2D(512, (3, 3), activation="relu", padding="same")(
            pool3
        )  # 128 x 128 x 128 (small and thick)
        conv8 = Conv2D(512, (3, 3), activation="relu", padding="same")(
            conv7
        )  # 256 x 256 x 64

        up2 = UpSampling2D((2, 2))(conv8)  # 28 x 28 x 64

        conc_up_2 = Concatenate()([up2, conv6])

        conv9 = Conv2D(256, (3, 3), activation="relu", padding="same")(
            conc_up_2
        )  # 512 x 512 x 1
        conv10 = Conv2D(256, (3, 3), activation="relu", padding="same")(
            conv9
        )  # 512 x 512 x 1

        up3 = UpSampling2D((2, 2))(conv10)  # 28 x 28 x 64

        conc_up_3 = Concatenate()([up3, conv4])

        conv11 = Conv2D(128, (3, 3), activation="relu", padding="same")(
            conc_up_3
        )  # 512 x 512 x 1
        conv12 = Conv2D(128, (3, 3), activation="relu", padding="same")(
            conv11
        )  # 512 x 512 x 1

        up4 = UpSampling2D((2, 2))(conv12)  # 28 x 28 x 64

        conc_up_4 = Concatenate()([up4, conv2])

        conv13 = Conv2D(64, (3, 3), activation="relu", padding="same")(
            conc_up_4
        )  # 512 x 512 x 1
        decoded = Conv2D(1, (1, 1), activation=None, padding="same")(
            conv13
        )  # 512 x 512 x 1

        return decoded

    return local_network_function


def dense_thick_units(path_json):
    def local_network_function(input_data, nb_units=60, nb_layers=10):

        current_input = input_data

        for depth in np.range(nb_layers):
            current_output = Dense(nb_units, activation="relu", padding="same")(
                current_input
            )

        final_output = Dense(1, activation="relu", padding="same")(current_output)

        return final_output

    return local_network_function
