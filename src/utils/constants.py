NUM_FEATURES=['Air temperature [K]', 'Process temperature [K]',
       'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
CAT_FEATURES=['Type']
TARGET=['Target','Failure Type']

#validation and test set
VAL_SIZE, TEST_SIZE = 0.2, 0.2

# Model training parameters
LEARNING_RATE=1e-4
PATIENCE=20
MIN_DELTA=1e-9
START_FROM_EPOCH=200
EPOCS=1000
BATCH_SIZE=32