# bert_study

## Example Code for running train_test.py
`python3 train_test.py --research_subject local_test --research_num 01 --batch_size 16 --max_length 50 --dropout 0.2 --lr 2e-5 --n_epochs 5 | tee ../results/local_test/local_test_01`

## Example Code for running test.py
`python3 test.py --research local_test --research_num 01 --batch_size 16 --max_length 50`
