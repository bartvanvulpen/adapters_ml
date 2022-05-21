# Meta-Learning AdapterFusion

### Instructions

#### Meta-Training
You can meta train on tasks with specified adapters using `meta_train.py`. For example:
```
python meta_train.py --adapters mnli qqp sst wgrande boolq --tasks mrpc scitail --k_shot 4 --max_steps 250
```
For the other parameters (like number of inner steps etc) see the argument descriptions of the `ArgumentParser` in `meta_train.py`. 
TODO: add detailed description of params

The models are saved in a folder named based on their adapters+tasks configuration. Each such folder can contain different versions of a model setup (version_1, version_2 etc, each corresponding to a run). For each model, you can see the specific parameters it was trained with in its `hparams.yaml`.  

Note: increasing `--n_workers` may increase training time, but don't set it too high (then memory issues might occur), by default it is set to 0.

#### Meta-Testing
You can meta test a model on a task using `meta_test.py`, by specifying a meta-trained model checkpoint. For example:
```
python meta_test.py --test_task mrpc --ckpt path/to/checkpoint.ckpt --k_values 2 4 8 --max_it 20 --results_file_path unique_file_name.json
```
Again, see the argument descriptions of the `ArgumentParser` in `meta_train.py`. 


