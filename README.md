# Adversarial Logit Separation

## Acknoledgement
This codebase is forked from the wonderful repository of [TRADES](https://github.com/yaodongyu/TRADES.git). The original README.md is renamed as [TRADES_README.md](./TRADES_README.md)

## Abstract

This paper presents a new idea to increase adversarial robustness by forcing model to be diverse. Contrast with former papers which mostly emphasize on pairing logits to be similar, we introduced a notion of logit separation that emphasizes on adapting stronger predictors from a diverse set of jointly trained models. We use the state-of-the-art model TRADES as a base line, and show that our adapted loss function, together with the joint training method, outperform previous model. We find that ensembling diverse logits both theoretically and experimentally yield better results. 

## Testing Environment
- Pytorch
- CUDA
- linux
- autoattack

## File structure
`train_trades_cifar10.py`: main file to train model

`eval.py`: main file to evaluate model

`print_cosine.py`: print the cosine value of models in the ensemble

## Sample use case
The cifar data is saved at `../data`
```python3
# Train
python3 train_trades_cifar10.py --batch-size 2 --beta 1 --lam 1 --model-dir dummy
# Evaluate
python3 eval.py --data_dir ../data --batch_size 1 --model dummy/model-wideres-epoch6-0.pt --single
# Print cosine distance
python3 print_cosine.py --data_dir ../data --batch_size 10 --model model-cifar-separation-final/model-wideres-epoch10
```

## Trained model
The trained models can be accessed at the [Google Drive](https://drive.google.com/drive/folders/1duDKLafpdSINIg7Y_QpYeTXXSiBmXVBr?usp=sharing)