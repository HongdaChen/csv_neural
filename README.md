## 使用pytorch实现一个对表格数据的神经网络实现

you can run the main.py like this:

```
python main.py --optimizer sgd -lr 0.001 --momentum 0.9 --path ./data.csv --batch-size 200 --epochs 10
```

```
python main.py --optimizer adam -lr 0.001 --momentum 0.9 --beta1 0.9 --beta2 0.999 --path ./data.csv --batch-size 200 --epochs 10
```

