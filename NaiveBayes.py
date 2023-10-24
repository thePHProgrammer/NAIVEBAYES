df = pd.read_csv(r"C:\Users\Music\water_datasheet.csv")
X = df[['pH', 'turbidity']]
Y = df['is_clean']

# Data augmentation
augmented_df = pd.concat([df, df.sample(frac=0.5, random_state=42)])
X_augmented = augmented_df[['pH', 'turbidity']]
Y_augmented = augmented_df['is_clean']

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_augmented, Y_augmented, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_normalized, Y_train, epochs=200)

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
