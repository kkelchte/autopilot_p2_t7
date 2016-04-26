python pilot_train.py --model small --hidden_size 5 >> log_small_h5_both.txt ;
python pilot_train.py --model small --hidden_size 10 >> log_small_h10_both.txt ;
python pilot_train.py --model small --hidden_size 15 >> log_small_h15_app.txt ;
python pilot_train.py --model medium --hidden_size 5 >> log_medium_h5_app.txt ;
python pilot_train.py --model medium --hidden_size 10 >> log_medium_h10_app.txt ;
python pilot_train.py --model medium --hidden_size 15 >> log_medium_h15_app.txt ;
python pilot_train.py --model medium --hidden_size 20 >> log_medium_h20_app.txt ;
