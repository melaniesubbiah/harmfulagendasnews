local berts = import '../berts.libsonnet'; 

{
  dataset_reader : {
    type : "base_reader",
    token_indexers : {
      bert : berts.indexer,
    }
  },
  validation_dataset_reader: {
    type : "base_reader",
    token_indexers : {
      bert : berts.indexer,
    },
  },
  train_data_path: std.extVar('TRAIN_DATA_PATH'),
  validation_data_path: std.extVar('DEV_DATA_PATH'),
  test_data_path: std.extVar('TEST_DATA_PATH'),
  model: berts.classifier,
  data_loader : {
    batch_size: std.parseInt(std.extVar('BSIZE')),
    shuffle: true,
  },
  trainer: {
    num_epochs: std.parseInt(std.extVar('EPOCHS')),

    patience: 15,

    grad_norm: 5.0,
    validation_metric: "+validation_metric",
    checkpointer: {num_serialized_models_to_keep: 1,},
    cuda_device: std.parseInt(std.extVar("CUDA_DEVICE")),
    optimizer: {
      type: "adamw",
      lr: 1e-5
    }
  },
  random_seed:  std.parseInt(std.extVar("SEED")),
  pytorch_seed: std.parseInt(std.extVar("SEED")),
  numpy_seed: std.parseInt(std.extVar("SEED")),
  evaluate_on_test: true
}
