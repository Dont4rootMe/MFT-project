PYTHONPATH=/Users/artemon/Library/Mobile\ Documents/com~apple~CloudDocs/Programming/python_projects/MFT-project python policy/model/train_rnn.py \
    --data /Users/artemon/.cache/kagglehub/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/versions/3 \
    --ts_len 300 \
    --batch_size 64 \
    --epochs 1000 \
    --lr 0.001 \
    --hidden_size 128 \
    --num_layers 2 \
    --output /Users/artemon/Library/Mobile\ Documents/com~apple~CloudDocs/Programming/python_projects/MFT-project/output/price_rnn.pt