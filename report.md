## Overview

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. We need to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

## Results

-   Data Preprocessing

    -   What variable(s) are the target(s) for your model?

        -   The rest that aren't mentioned below

    -   What variable(s) are the features for your model?

        -   Application Type and Classification in `AlphabetSoupCharity.ipynb`
        -   Name and Classification in `AlphabetSoupCharity_Optimization`

    -   What variable(s) should be removed from the input data because they are neither targets nor features?

        -   EIN and Name in `AlphabetSoupCharity.ipynb`
        -   EIN in `AlphabetSoupCharity_Optimization`

-   Compiling, Training, and Evaluating the Model

    -   How many neurons, layers, and activation functions did you select for your neural network model, and why?

        -   `AlphabetSoupCharity.ipynb`

            ```py
            # Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
            input_features = X_train_scaled.shape[1]
            hidden_nodes1 = 80
            hidden_nodes2 = 30

            nn = tf.keras.models.Sequential()

            # First hidden layer
            nn.add(
                tf.keras.layers.Dense(
                    units=hidden_nodes1, input_dim=input_features, activation="relu"
                )
            )

            # Second hidden layer
            nn.add(tf.keras.layers.Dense(units=hidden_nodes2, activation="relu"))

            # Output layer
            nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

            # Check the structure of the model
            nn.summary()
            ```

            ```sh
            Model: "sequential"
            _________________________________________________________________
            Layer (type)                Output Shape              Param #
            =================================================================
            dense (Dense)               (None, 80)                6080

            dense_1 (Dense)             (None, 30)                2430

            dense_2 (Dense)             (None, 1)                 31

            =================================================================
            Total params: 8541 (33.36 KB)
            Trainable params: 8541 (33.36 KB)
            Non-trainable params: 0 (0.00 Byte)
            _________________________________________________________________
            ```

        -   `AlphabetSoupCharity_Optimization`

            ```py
            # Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
            input_features = X_train_scaled.shape[1]
            hidden_nodes1 = 7
            hidden_nodes2 = 14
            hidden_nodes3 = 21

            nn = tf.keras.models.Sequential()

            # First hidden layer
            nn.add(
                tf.keras.layers.Dense(
                    units=hidden_nodes1, input_dim=input_features, activation="relu"
                )
            )

            # Second hidden layer
            nn.add(tf.keras.layers.Dense(units=hidden_nodes2, activation="relu"))

            # Third hidden layer
            nn.add(tf.keras.layers.Dense(units=hidden_nodes3, activation="relu"))

            # Output layer
            nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

            # Check the structure of the model
            nn.summary()
            ```

            ```sh
            Model: "sequential_1"
            _________________________________________________________________
            Layer (type)                Output Shape              Param #
            =================================================================
            dense_3 (Dense)             (None, 7)                 2149

            dense_4 (Dense)             (None, 14)                112

            dense_5 (Dense)             (None, 21)                315

            dense_6 (Dense)             (None, 1)                 22

            =================================================================
            Total params: 2598 (10.15 KB)
            Trainable params: 2598 (10.15 KB)
            Non-trainable params: 0 (0.00 Byte)
            _________________________________________________________________
            ```

    -   Were you able to achieve the target model performance?

        -   Yes, using 3 layers in `AlphabetSoupCharity_Optimization` allowed me to achieve an accuracy above 75%.

    -   What steps did you take in your attempts to increase model performance?

        -   Kept Name in the model and applied Name as a feature
        -   Kept Classification as a feature
        -   Added a third layer
        -   Increased epochs to 200

## Summary

Keeping Name in the model, applying it as a feature, adding a third layer and increasing the epochs to 200 allowed me to achieve an accuracy above 75% which was the target model performance.
