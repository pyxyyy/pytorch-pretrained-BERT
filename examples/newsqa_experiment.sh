
python run_squad.py --bert_model bert-base-cased --output_dir experiment/bert-base-cased-newsqa --train_file data/newsqa/newsQaJSONSquadFormat_7551_oneAnswer.json --do_train --version_2_with_negative --is_newsqa

# for TESTING if the code runs well
python run_squad.py --bert_model bert-base-cased --output_dir experiment/bert-base-cased-newsqa --predict_file data/newsqa/newsQaJSONSquadFormat_7551_oneAnswer.json --do_predict --version_2_with_negative --is_newsqa
python evaluate-v2.0.py data/newsqa/newsQaJSONSquadFormat_7551_oneAnswer.json experiment/bert-base-cased-newsqa/predictions.json
# {
#   "exact": 0.09830908375933936,
#   "f1": 3.1064383286528536,
#   "total": 5086,
#   "HasAns_exact": 0.06523157208088715,
#   "HasAns_f1": 3.391899399766997,
#   "HasAns_total": 4599,
#   "NoAns_exact": 0.4106776180698152,
#   "NoAns_f1": 0.4106776180698152,
#   "NoAns_total": 487
# }
# apparently if the train and predict code is called separately, it used the pretrained bert model instead of trained one. 
# after editing the code
# {
#   "exact": 0.7078254030672434,
#   "f1": 2.617162519573328,
#   "total": 5086,
#   "HasAns_exact": 0.17395085888236572,
#   "HasAns_f1": 2.28547261894976,
#   "HasAns_total": 4599,
#   "NoAns_exact": 5.749486652977413,
#   "NoAns_f1": 5.749486652977413,
#   "NoAns_total": 487
# }

# predict for SQuAD max_seq_length=288?
python run_squad.py --bert_model bert-base-cased --output_dir experiment/bert-base-cased-squad-288 --predict_file data/squad/dev-v2.0.json --do_predict --version_2_with_negative --max_seq_length=288
python evaluate-v2.0.py data/squad/dev-v2.0.json experiment/bert-base-cased-squad-288/predictions.json
# {
#   "exact": 0.22740672113198013,
#   "f1": 4.332064541384656,
#   "total": 11873,
#   "HasAns_exact": 0.16869095816464239,
#   "HasAns_f1": 8.389777715900813,
#   "HasAns_total": 5928,
#   "NoAns_exact": 0.28595458368376786,
#   "NoAns_f1": 0.28595458368376786,
#   "NoAns_total": 5945
# }
# apparently if the train and predict code is called separately, it used the pretrained bert model instead of trained one. 
# after editing the code
# {
#   "exact": 50.07159100480081,
#   "f1": 50.07159100480081,
#   "total": 11873,
#   "HasAns_exact": 0.0,
#   "HasAns_f1": 0.0,
#   "HasAns_total": 5928,
#   "NoAns_exact": 100.0,
#   "NoAns_f1": 100.0,
#   "NoAns_total": 5945
# }

# predict for SQuAD 
python run_squad.py --bert_model bert-base-cased --output_dir experiment/bert-base-cased-squad --predict_file data/squad/dev-v2.0.json --do_predict --version_2_with_negative
python evaluate-v2.0.py data/squad/dev-v2.0.json experiment/bert-base-cased-squad/predictions.json
# {
#   "exact": 0.22740672113198013,
#   "f1": 4.341144842307596,
#   "total": 11873,
#   "HasAns_exact": 0.16869095816464239,
#   "HasAns_f1": 8.407964357745962,
#   "HasAns_total": 5928,
#   "NoAns_exact": 0.28595458368376786,
#   "NoAns_f1": 0.28595458368376786,
#   "NoAns_total": 5945
# }
# apparently if the train and predict code is called separately, it used the pretrained bert model instead of trained one. 
# after editing the code

# the result doesn't make any sense, I'll try to replicate the github evaluation
# Training with the previous hyper-parameters gave us the following results:
# {"f1": 88.52381567990474, "exact_match": 81.22043519394512}
# I'll see if I can replicate this. RESULT: yes, replicated
python run_squad.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file data/squad/train-v1.1.json \
  --predict_file data/squad/dev-v1.1.json \
  --train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir experiment/bert-base-cased-squad-v1.1/
python evaluate-v1.1.py data/squad/dev-v1.1.json experiment/bert-base-cased-squad-v1.1/predictions.json

# apparently if the train and predict code is called separately, it used the pretrained bert model instead of trained one. 
python run_squad.py \
  --bert_model bert-base-cased \
  --output_dir experiment/bert-base-cased-newsqa \
  --do_train \
  --do_predict \
  --train_file data/newsqa/newsQaJSONSquadFormat_7551_oneAnswer.json \
  --predict_file data/newsqa/newsQaJSONSquadFormat_7551_oneAnswer.json \
  --version_2_with_negative \
  --is_newsqa
  
# and also I guess the result is not too good when you don't set any batch size etc.
# https://github.com/huggingface/pytorch-pretrained-BERT/pull/174 
# Let's try for Newsqa dataset
# NOW - is running this to get accuracy result
python run_squad.py \
  --bert_model bert-base-cased \
  --do_train \
  --do_predict \
  --train_file data/newsqa/newsQaJSONSquadFormat_7551_oneAnswer.json \
  --predict_file data/newsqa/newsQaJSONSquadFormat_7551_oneAnswer.json \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir experiment/bert-base-cased-newsqa \
  --train_batch_size 24 \
  --loss_scale 128 \
  --version_2_with_negative \
  --null_score_diff_threshold -1 \
  --is_newsqa
python evaluate-v2.0.py data/newsqa/newsQaJSONSquadFormat_7551_oneAnswer.json experiment/bert-base-cased-newsqa/predictions.json
# {
#   "exact": 1.2780180888714117,
#   "f1": 2.057326434915497,
#   "total": 5086,
#   "HasAns_exact": 0.21743857360295715,
#   "HasAns_f1": 1.079269895190303,
#   "HasAns_total": 4599,
#   "NoAns_exact": 11.293634496919918,
#   "NoAns_f1": 11.293634496919918,
#   "NoAns_total": 487
# }

# Let's try with Squad now
python run_squad.py \
  --bert_model bert-base-cased \
  --do_train \
  --do_predict \
  --train_file data/squad/train-v2.0.json \
  --predict_file data/squad/dev-v2.0.json \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir experiment/bert-base-cased-squad-384-batch-24 \
  --train_batch_size 24 \
  --loss_scale 128 \
  --version_2_with_negative \
  --null_score_diff_threshold -1
python evaluate-v2.0.py data/squad/dev-v2.0.json experiment/bert-base-cased-squad-384-batch-24/predictions.json
# {
#   "exact": 71.75945422386928,
#   "f1": 75.02373600970313,
#   "total": 11873,
#   "HasAns_exact": 71.28879892037787,
#   "HasAns_f1": 77.82672362402222,
#   "HasAns_total": 5928,
#   "NoAns_exact": 72.22876366694702,
#   "NoAns_f1": 72.22876366694702,
#   "NoAns_total": 5945
# }

# Apparently I'm only training with 7.5% of the total data. Now that I have the 80%, let's see the accuracy
python run_squad.py \
  --bert_model bert-base-cased \
  --do_train \
  --do_predict \
  --train_file data/newsqa/training_80.json \
  --predict_file data/newsqa/test_10.json \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir experiment/bert-base-cased-newsqa-80-10 \
  --train_batch_size 24 \
  --loss_scale 128 \
  --version_2_with_negative \
  --null_score_diff_threshold -1 \
  --is_newsqa
python evaluate-v2.0.py data/newsqa/test_10.json experiment/bert-base-cased-newsqa-80-10/predictions.json
# {
#   "exact": 10.745098039215685,
#   "f1": 10.745098039215685,
#   "total": 1275,
#   "HasAns_exact": 0.0,
#   "HasAns_f1": 0.0,
#   "HasAns_total": 1138,
#   "NoAns_exact": 100.0,
#   "NoAns_f1": 100.0,
#   "NoAns_total": 137
# }

# The result is as bad. Is it because of max_seq_length that is too short? How long is the CNN articles?
# Changing max_seq_length from 384 to 512, cannot used 600, max is 512 for this Bert model. It will result in indexing error
python run_squad.py \
  --bert_model bert-base-cased \
  --do_train \
  --do_predict \
  --train_file data/newsqa/training_80.json \
  --predict_file data/newsqa/test_10.json \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 512 \
  --doc_stride 128 \
  --output_dir experiment/bert-base-cased-newsqa-80-10-msl-600 \
  --train_batch_size 24 \
  --loss_scale 128 \
  --version_2_with_negative \
  --null_score_diff_threshold -1 \
  --is_newsqa

'''
#python run_squad.py --bert_model bert-base-cased --output_dir experiment/bert-base-cased-newsqa --predict_file dev-newsqa.json --do_predict --version_2_with_negative

/home/user/fiona/pytorch-pretrained-BERT/examples
newsQaJSONSquadFormat_complete.json
python run_squad.py --bert_model bert-base-cased --output_dir experiment/bert-base-cased-newsqa --train_file newsQaJSONSquadFormat_complete.json --do_train --version_2_with_negative

# python run_squad.py --bert_model bert-base-uncased --output_dir experiment/bert-base-uncased-newsqa --train_file train-newsqa.json --do_train --version_2_with_negative --do_lower_case
# python run_squad.py --bert_model bert-base-uncased --output_dir experiment/bert-base-uncased-newsqa --predict_file dev-newsqa.json --do_predict --version_2_with_negative --do_lower_case
'''